import argparse
import json
import sys
import time
import pickle
import torch
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.core.scheduler import AllocStatus
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.utils import PacketType, get_packet_type, RecvKVCacheCoordinator
from vllm.sequence import SequenceStatus, Sequence
from vllm.worker.model_runner import _make_tensor_with_pad
from safetensors.torch import load

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    prefix_pos = request_dict.pop("prefix_pos", None)
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt,
                                        sampling_params,
                                        request_id,
                                        prefix_pos=prefix_pos)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    ret = {"text": text_outputs}
    return JSONResponse(ret)

@app.websocket("/decode")
async def decode(ws: WebSocket):
    if not engine.is_running:
        engine.start_background_loop()

    await ws.accept()
    print("prefill worker connected")

    _engine = engine.engine
    scheduler = _engine.scheduler
    block_size = scheduler.block_manager.block_size
    request_tracker = engine._request_tracker
    gpu_cache = _engine.driver_worker.gpu_cache
    cpu_cache = _engine.driver_worker.cpu_cache
    cache_engine = _engine.driver_worker.cache_engine
    num_layers = _engine.model_config.get_num_layers(_engine.parallel_config)

    current_block_tables = []
    src_to_dst = {}

    # r_kvc = RecvKVCacheCoordinator(num_layers, gpu_cache, scheduler.without_kv, scheduler.with_kv, request_tracker)



    # batch_frames = num_layers * 2 + 1
    is_query = True
    ith = 0
    checked_events = 0

    is_k = True
    while True:
        ith %= num_layers
        packet = await ws.receive_bytes()
        start = time.perf_counter()
        if len(packet) > 1000000:
            packet_type = PacketType.KV_CACHE
        else:
            if is_query:
                packet_type = PacketType.QUERY
                is_query = False
            else:
                packet_type = PacketType.DECODE
                is_query = True

        if packet_type ==  PacketType.QUERY:
            seq_groups = pickle.loads(packet)

            if scheduler.block_manager.can_allocates(seq_groups) == AllocStatus.OK:
                slot_mapping: List[List[int]] = []
                max_prompt_len = 0
                for seq_group in seq_groups:
                    seq: Sequence = seq_group.get_seqs()[0]
                    prompt_len = len(seq.data.get_token_ids())
                    if prompt_len > max_prompt_len:
                        max_prompt_len = prompt_len
                    slot_mapping.append([])
                    for seq in seq_group.get_seqs():
                        seq.status = SequenceStatus.WAITING
                    scheduler._allocate(seq_group)
                    block_table = scheduler.block_manager.get_block_table(seq)
                    current_block_tables.extend(block_table)

                print(f"prove {len(seq_groups)} requests")
                await ws.send_text("yes")
            else:
                is_query = True
                print(f"reject {len(seq_groups)} requests")
                await ws.send_text("no")

        elif packet_type ==  PacketType.KV_CACHE:
            if ith % 16 == 0:
                s = time.perf_counter()
                for event in cache_engine.events[checked_events:]:
                    if event.query():
                        checked_events += 1

                if checked_events == num_layers:
                    if scheduler.without_kv:
                        scheduler.with_kv.extend(scheduler.without_kv.pop(0))
                        request_tracker.new_requests_event.set()
                    checked_events == 0

                print(f"check takes: {1000 * (time.perf_counter() - s)} ms", file=sys.stderr)

            kv_cpu = torch.frombuffer(packet, dtype=torch.float16)

            if is_k:
                kv_cpu_reshape = kv_cpu.reshape(-1, 8, 16, 16, 8)
                if ith == 0:
                    bs = kv_cpu_reshape.shape[0]
                    src_to_dst = {i: current_block_tables[i] for i in range(bs)}
                cpu_cache[ith][0][:bs].copy_(kv_cpu_reshape)
                is_k = False
            else:
                kv_cpu_reshape = kv_cpu.reshape(-1, 8, 128, 16)
                bs = kv_cpu_reshape.shape[0]
                cpu_cache[ith][1][:bs].copy_(kv_cpu_reshape)

                s = time.perf_counter()
                cache_engine.swap_in_layerwise(ith, src_to_dst)
                print(f"swap in {ith} takes: {1000 * (time.perf_counter() - s)} ms", file=sys.stderr)

                is_k = True
                ith += 1

        elif packet_type ==  PacketType.DECODE:
            s = time.perf_counter()
            seq_groups = pickle.loads(packet)
            print(f"received {len(seq_groups)} requests")
            scheduler.without_kv.append(seq_groups)
            print(f"niubi takes: {1000 * (time.perf_counter() - s)} ms", file=sys.stderr)

        print(f"process {packet_type} takes: {1000 * (time.perf_counter() - start)} ms", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    app.root_path = args.root_path
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ws_max_size=1024*1024*1024,
                ws_max_queue=512)
