from typing import List, Tuple
import zmq
import pickle
from multiprocessing import Queue

from vllm.worker.cache_engine import KVCache
from vllm.logger import init_logger

logger = init_logger(__name__)

class KVPusher:
    def __init__(self, gpu_cache: List[KVCache], cpu_cache: List[KVCache]):
        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.connect("tcp://127.0.0.1:7777")

    def send_kv(self, ith: int, to_send: List[Tuple[int, int]]):
        logger.info(f"start sending {ith}th layer kv, to_send: {to_send}")
        for start, l in to_send:
            self.cpu_cache[ith][0][start: start+l].copy_(self.gpu_cache[ith][0][start: start+l])
            self.cpu_cache[ith][1][start: start+l].copy_(self.gpu_cache[ith][1][start: start+l])
            self.sock.send(pickle.dumps(self.cpu_cache[ith][0][start: start+l]), copy=False)
            self.sock.send(pickle.dumps(self.cpu_cache[ith][1][start: start+l]), copy=False)
        logger.info(f"end sending {ith}th layer kv")


class KVPuller:
    def __init__(self, num_layers, result_queue: Queue, gpu_cache: List[KVCache], cpu_cache: List[KVCache]):
        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        self.num_layers = num_layers
        self.result_queue = result_queue
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PULL)
        self.sock.bind("tcp://127.0.0.1:7777")

    def receive_kv(self, to_receive: List[Tuple[int, int]]):
        for ith in range(self.num_layers):
            logger.info(f"start receiving {ith}th layer kv, to_receive: {to_receive}")
            for start, l in to_receive:
                self.cpu_cache[ith][0][start: start+l].copy_(pickle.loads(self.sock.recv()))
                self.cpu_cache[ith][1][start: start+l].copy_(pickle.loads(self.sock.recv()))
                self.gpu_cache[ith][0][start: start+l].copy_(self.cpu_cache[ith][0][start: start+l])
                self.gpu_cache[ith][1][start: start+l].copy_(self.cpu_cache[ith][1][start: start+l])
            logger.info(f"end receiving {ith}th layer kv")
        self.result_queue.put("done")
