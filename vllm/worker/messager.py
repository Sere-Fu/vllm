from typing import List, Tuple
import zmq
import sys
import torch
import time

from vllm.logger import init_logger

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

class Messager:
    def __init__(self, role, url, num_layers, result_queue,
                 gpu_cache: List[KVCache], cpu_cache: List[KVCache], kv_buffer: List[KVCache]):
        self.role = role
        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        self.kv_buffer = kv_buffer
        self.num_layers = num_layers
        self.result_queue = result_queue
        self.init_zmq(url)

    def init_zmq(self, url):
        ctx = zmq.Context()
        if self.role == "pusher":
            self.sock = ctx.socket(zmq.PUSH)
            self.sock.connect(url)
        elif self.role == "puller":
            self.sock = ctx.socket(zmq.PULL)
            self.sock.bind(url)

    def send_kv(self, num_slots, ith: int):
        self.sock.send(self.kv_buffer[ith][0][:num_slots].numpy(), copy=False)
        self.sock.send(self.kv_buffer[ith][1][:num_slots].numpy(), copy=False)

        if ith == self.num_layers-1:
            logger.info(f"end sending kv cache")

    def receive_kv(self):
        ith = 0
        while True:
            k = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            v = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            if ith == self.num_layers-1:
                logger.info(f"end receiving kv cache")
                self.result_queue.put(b'0x01')
                return
            ith += 1

    def receive_kv_forever(self):
        ith = 0
        while True:
            k = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            v = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            if ith == self.num_layers-1:
                ith = 0
                logger.info(f"end receiving kv cache")
                self.result_queue.put(b'0x01')
            ith += 1
