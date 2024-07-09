"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

def sample_requests(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    prompt_len,
    output_len
) -> List[Tuple[str, int, int]]:
    # Tokenize the prompts and completions.

    prompt = "Hi" * prompt_len
    prompts = [prompt]
    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(50000):
        tokenized_dataset.append((prompts[0], prompt_token_ids[0], output_len))

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    return sampled_requests

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(backend: str, model: str, api_url: str, prompt: str,
                       prompt_len: int, output_len: int, best_of: int,
                       use_beam_search: bool, pbar: tqdm) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
        if model is not None:
            pload["model"] = model
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers,
                                    json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    pbar.update(1)


async def benchmark(
    backend: str,
    model: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    pbar = tqdm(total=len(input_requests))
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(backend, model, api_url, prompt, prompt_len,
                         output_len, best_of, use_beam_search, pbar))
        tasks.append(task)
    await asyncio.gather(*tasks)
    pbar.close()


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"{args.protocol}://{args.host}:{args.port}{args.endpoint}"
    tokenizer = get_tokenizer(args.tokenizer,
                              trust_remote_code=args.trust_remote_code)
    input_requests = sample_requests(args.num_prompts, tokenizer, args.prompt_len, args.output_len)

    benchmark_start_time = time.perf_counter()
    asyncio.run(
        benchmark(args.backend, args.model, api_url, input_requests,
                  args.best_of, args.use_beam_search, args.request_rate))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--backend",
                        type=str,
                        default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--protocol",
                        type=str,
                        default="http",
                        choices=["http", "https"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--endpoint", type=str, default="/generate")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tokenizer",
                        type=str,
                        required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--best-of",
                        type=int,
                        default=1,
                        help="Generates `best_of` sequences per prompt and "
                        "returns the best one.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--prompt-len",
                        type=int,
                        default=100,
                        help="prompt length")
    parser.add_argument("--output-len",
                        type=int,
                        default=20,
                        help="output length")
    parser.add_argument("--request-rate",
                        type=float,
                        default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                        "then all the requests are sent at time 0. "
                        "Otherwise, we use Poisson process to synthesize "
                        "the request arrival times.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    args = parser.parse_args()
    main(args)
