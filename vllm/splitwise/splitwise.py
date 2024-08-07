
import asyncio
import time
from collections import deque
from typing import Callable, List

import aiohttp
import torch
import torch.distributed as dist
from torch import distributed as dist
from transformers import PretrainedConfig

from vllm import _custom_ops as ops
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.outputs import RequestOutput
from vllm.sequence import SamplerOutput, SequenceGroup

from .request import SplitwiseRequest


class KVCacheCoordinator:
    MAX_WIP = 400

    def __init__(self):
        self.pending = deque()  # Pending caused by too many running IOs
        self.wip = deque()
        self.p2p_group = dist.new_group(list(range(dist.get_world_size())))
        self.last_time = time.perf_counter()
        self.n_completed = 0
        self.n_issue = 0

    def _invoke(self, buf, op, cb):
        if len(self.pending) >= KVCacheCoordinator.MAX_WIP * 4:
            tbd = len(self.wip) + len(self.pending)
            t0 = time.perf_counter()
            n_completed0 = self.n_completed
            while len(self.pending) >= KVCacheCoordinator.MAX_WIP * 4:
                assert time.perf_counter() - t0 < 0.1, f'👾 Stall {(time.perf_counter() - t0)*1000:.2f} ms, '\
                    f'completed {self.n_completed-n_completed0}/{tbd}'
                self.complete_io_and_dispatch_pending()
            print(f'👾 Stall {(time.perf_counter() - t0)*1000:.2f} ms, '
                  f'completed {self.n_completed-n_completed0}/{tbd}')

        if len(self.wip) < KVCacheCoordinator.MAX_WIP:
            self.n_issue += 1
            self.wip.append((buf, op(), cb))
        else:
            self.pending.append((buf, op, cb))

    def isend(self, buf, dst, cb=None):
        self._invoke(buf, lambda: dist.isend(
            buf, dst=dst, group=self.p2p_group), cb)

    def irecv(self, buf, src, cb=None):
        self._invoke(buf, lambda: dist.irecv(
            buf, src=src, group=self.p2p_group), cb)

    def complete_io_and_dispatch_pending(self):
        nwip = len(self.wip)
        nbytes = 0
        elapsed = time.perf_counter() - self.last_time
        self.last_time = time.perf_counter()
        while self.wip:
            buf, h, cb = self.wip[0]
            if h.is_completed():
                self.n_completed += 1
                nbytes += buf.numel() * buf.element_size()
                if cb is not None:
                    cb()
                self.wip.popleft()
            else:
                break
        while self.pending:
            if len(self.wip) < KVCacheCoordinator.MAX_WIP:
                buf, op, cb = self.pending.popleft()
                self.n_issue += 1
                self.wip.append((buf, op(), cb))
            else:
                break
        # if nwip != len(self.wip):
        #     print(f'👾completed={nwip-len(self.wip)}, wip={len(self.wip)}, pending={len(self.pending)}, '
        #           f'interval={elapsed*1000:.3f}ms, bandwidth={nbytes/elapsed/1024**3:.3f}GB/s, '
        #           f'accumaleted_issue={self.n_issue}, accumaleted_complete={self.n_completed}')

    def has_running_io(self):
        return len(self.wip) > 0 or len(self.pending) > 0

    def next_id(self):
        return self.n_issue + len(self.pending)


_KVCC = None


def initialize_kvcc():
    global _KVCC
    assert _KVCC is None, "KVCC is already initialized"
    _KVCC = KVCacheCoordinator()


def get_kvcc():
    assert _KVCC is not None, "KVCC is not initialized"
    return _KVCC


def notify_prefill_and_resume_later(model: str, scheduler: Scheduler,
                                    scheduler_outputs: SchedulerOutputs,
                                    out_continuation: Callable[[List[RequestOutput]], bool]):
    assert len(scheduler_outputs.scheduled_seq_groups) == 1
    seq_group: SequenceGroup = scheduler_outputs.scheduled_seq_groups[0].seq_group
    assert seq_group.splitwise_request.prefill_endpoint is not None

    async def notify_prefill():
        data = {
            'model': model,
            'prompt': next(iter(seq_group.seqs_dict.values())).inputs['prompt_token_ids'],
            'max_tokens': 1,
            'decoding_rank': dist.get_rank(),
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(seq_group.splitwise_request.prefill_endpoint, json=data) as response:
                async for _ in response.content.iter_any():
                    pass
    asyncio.create_task(notify_prefill())
    output_future = asyncio.get_event_loop().create_future()
    seq_group.splitwise_request.future = output_future

    async def continuation():
        await output_future
        output = output_future.result()
        scheduler.running.append(seq_group)
        out_continuation(output)
    asyncio.create_task(continuation())


def recv_kv_caches_and_resume_later(
        splitwise_request: SplitwiseRequest,
        kv_caches: List[torch.Tensor], slot_mapping: torch.Tensor, kv_cache_dtype: str,
        config: PretrainedConfig, token_len: int, model_dtype: torch.dtype, device: torch.device,
        vocab_size: int, logits_callback: Callable[[torch.Tensor], SamplerOutput]):

    kv_heads = config.num_attention_heads
    if hasattr(config, 'num_key_value_heads'):
        kv_heads = config.num_key_value_heads
    kv_cache_shape = (token_len, kv_heads, config.hidden_size // config.num_attention_heads)

    def mkcb(i, k_buf, v_buf):
        def cb():
            key_cache = kv_caches[i][0]
            value_cache = kv_caches[i][1]
            ops.reshape_and_cache_flash(
                k_buf,
                v_buf,
                key_cache,
                value_cache,
                slot_mapping.flatten(),
                kv_cache_dtype,
            )
        return cb

    kvcc = get_kvcc()

    for i in range(config.num_hidden_layers):
        k_buf = torch.empty(kv_cache_shape, dtype=model_dtype, device=device)
        v_buf = torch.empty(kv_cache_shape, dtype=model_dtype, device=device)
        kvcc.irecv(k_buf, src=splitwise_request.prefill_rank)
        kvcc.irecv(v_buf, src=splitwise_request.prefill_rank,
                   cb=mkcb(i, k_buf, v_buf))

    logits_buf = torch.empty((1, vocab_size), dtype=model_dtype, device=device)
    assert splitwise_request.future is not None

    def cb():
        output: SamplerOutput = logits_callback(logits_buf)
        splitwise_request.future.set_result([output])
    kvcc.irecv(logits_buf, src=splitwise_request.prefill_rank, cb=cb)
    kvcc.complete_io_and_dispatch_pending()
