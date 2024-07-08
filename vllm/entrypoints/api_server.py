import argparse
import json
from typing import AsyncGenerator, Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.utils import marshalToB64String, unmarshalFromB64String, coalesce_blocks
from vllm.sequence import SequenceStatus

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


@app.post("/receive_kv_cache")
async def receive_kv_cache(request: Request) -> Response:
    '''just send back ack for now, but in the future we can use this to update the kv cache with the received blocks'''
    request_dict = await request.json()
    from_rank = request_dict.pop("from_rank")
    seq_groups = unmarshalFromB64String(request_dict.pop("encoded_seq_groups"))

    bts = []
    for seq_group in seq_groups:
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.WAITING
        engine.engine.scheduler._allocate(seq_group)
        block_tables: Dict[int, List[int]] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq_id = seq.seq_id
            block_tables[seq_id] = engine.engine.scheduler.block_manager.get_block_table(seq)
            bts.append(block_tables)

    to_receive = coalesce_blocks([block
                                    for block_tables in bts
                                    for blocks in block_tables.values()
                                    for block in blocks ])

    await engine.create_receive_kv_cache_task(from_rank, to_receive)

    ret = {"output":  "ack"}
    return JSONResponse(ret)

@app.post("/decode")
async def decode(request: Request) -> Response:
    request_dict = await request.json()
    seq_groups = unmarshalFromB64String(request_dict.pop("encoded_seq_groups"))

    engine.pre_running_requests.append(seq_groups) # process later
    if sum(len(seq_groups) for seq_groups in engine.pre_running_requests) >= engine.engine.scheduler_config.max_num_seqs:
        engine._request_tracker.new_requests_event.set()

    ret = {"output":  "ack"}
    return JSONResponse(ret)

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
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
