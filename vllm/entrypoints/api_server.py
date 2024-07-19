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
    num_layers = _engine.model_config.get_num_layers(_engine.parallel_config)

    current_slot_mapping = None

    r_kvc = RecvKVCacheCoordinator(num_layers, gpu_cache, scheduler.without_kv, scheduler.with_kv, request_tracker)


    is_k = True
    last_k_cpu_pin = None
    last_k_gpu = None
    while True:
        packet = await ws.receive_bytes()
        packet_type = get_packet_type(packet)
        if packet_type ==  PacketType.QUERY:
            seq_groups = pickle.loads(packet[1:])

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

                    start_idx = 0
                    for i in range(prompt_len):
                        if i < start_idx:
                            slot_mapping[-1].append(-1)
                            continue

                        block_number = block_table[i // block_size]
                        block_offset = i % block_size
                        slot = block_number * block_size + block_offset
                        slot_mapping[-1].append(slot)

                current_slot_mapping = _make_tensor_with_pad(slot_mapping,
                                                    max_prompt_len,
                                                    pad=-1,
                                                    dtype=torch.long,
                                                    device='cuda')
                # print(f"current slot mapping {current_slot_mapping}", file=sys.stderr)

                print(f"prove {len(seq_groups)} requests")
                await ws.send_text("yes")
            else:
                print(f"reject {len(seq_groups)} requests")
                await ws.send_text("no")

        if packet_type ==  PacketType.KV_CACHE:
            m0 = time.perf_counter()
            ith = packet[1]
            kv_bytes = packet[2:]
            l = len(kv_bytes)//2

            # print(f"received {ith} {kv_dict['k'].shape} {kv_dict['v'].shape}")

            print(f"----------------check time: {1000 * (time.perf_counter()-m0)} ms", file=sys.stderr)

            # k_bytes = kv_bytes[:l]
            # v_bytes = kv_bytes[l:]

            r_kvc.check()
            m1 = time.perf_counter()

            kv_cpu = torch.frombuffer(kv_bytes, dtype=torch.float16)
            # v_cpu = torch.frombuffer(v_bytes, dtype=torch.float16)

            m7 = time.perf_counter()
            print(f"----------------de-s time: {1000 * (m7-m1)} ms", file=sys.stderr)

            kv_cpu = kv_cpu.reshape(-1, 8, 128)
            # if is_k:
            #     kv_cpu = kv_cpu.reshape(-1, 8, 16, 16, 8)
            # else:
            #     kv_cpu = kv_cpu.reshape(-1, 8, 128, 16)
            # k_cpu = v_cpu.reshape(-1, 8, 128)

            m2 = time.perf_counter()
            print(f"----------------reshape time: {1000 * (m2-m7)} ms", file=sys.stderr)

            kv_cpu_pin = kv_cpu.pin_memory()
            # v_cpu_pin = v_cpu.pin_memory()

            m3 = time.perf_counter()
            print(f"----------------pin time: {1000 * (m3-m2)} ms", file=sys.stderr)
            with torch.cuda.stream(engine.async_io_stream):
                kv_gpu = kv_cpu_pin.to('cuda', non_blocking=True)
            # v_gpu = v_cpu_pin.to('cuda', non_blocking=True)
                if is_k:
                    last_k_cpu_pin = kv_cpu_pin
                    last_k_gpu = kv_gpu
                    is_k = False
                else:
                    event = torch.cuda.Event()
                    event.record()
                    r_kvc.submit(last_k_cpu_pin, kv_cpu_pin, last_k_gpu, kv_gpu, ith, current_slot_mapping, event)
                    is_k = True
            # k.reshape
            # engine.engine.driver_worker.cpu_kv_buffer[i][0].copy_(k)
            # engine.engine.driver_worker.cpu_kv_buffer[i][1].copy_(v)

        if packet_type ==  PacketType.DECODE:
            seq_groups = pickle.loads(packet[1:])
            print(f"received {len(seq_groups)} requests")
            scheduler.without_kv.append(seq_groups)

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
