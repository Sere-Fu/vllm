import time
from collections import deque

import torch.distributed as dist


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
    if _KVCC is None:
        _KVCC = KVCacheCoordinator()
    return _KVCC


def get_kvcc():
    assert _KVCC is not None, "KVCC is not initialized"
    return _KVCC


class SplitwiseRequest:
    def __init__(self, prefill_endpoint, decoding_rank):
        self.prefill_endpoint = prefill_endpoint
        self.decoding_rank = decoding_rank
