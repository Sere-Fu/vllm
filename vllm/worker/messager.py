from typing import List, Tuple
import zmq
import sys
import torch
import time
from vllm._C import cache_ops

from vllm.logger import init_logger

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

class Messager:
    def __init__(self, role, url, num_layers, kv_shape, result_queue,
                 gpu_cache, cpu_cache, gpu_buffer, cpu_buffer: List[KVCache]):
        self.role = role
        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        self.gpu_buffer = gpu_buffer
        self.cpu_buffer = cpu_buffer
        self.num_layers = num_layers
        self.kv_shape = kv_shape
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
        self.sock.send(self.cpu_buffer[ith][0][:num_slots].numpy(), copy=False)
        self.sock.send(self.cpu_buffer[ith][1][:num_slots].numpy(), copy=False)

        if ith == self.num_layers-1:
            logger.info(f"end sending kv cache")

    def send_slot_mapping(self, slot_mapping):
        self.sock.send(slot_mapping.numpy(), copy=False)

    def receive_kv(self):
        slot_mapping = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.int64).to('cuda')
        num_slots = slot_mapping.size(0)

        ith = 0
        while True:
            k_buffer = self.gpu_buffer[ith][0][:num_slots]
            v_buffer = self.gpu_buffer[ith][1][:num_slots]

            k = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16).reshape(-1, *self.kv_shape)
            k_buffer.copy_(k)

            v = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16).reshape(-1, *self.kv_shape)
            v_buffer.copy_(v)

            cache_ops.reshape_and_cache(
                k_buffer,
                v_buffer,
                self.gpu_cache[ith][0],
                self.gpu_cache[ith][1],
                slot_mapping,
                "auto",
            )

            if ith == self.num_layers-1:
                logger.info(f"end receiving kv cache")
                self.result_queue.put(b'0x01')
                return
            ith += 1

    def receive_kv_forever(self):
        current_slot_mapping = None
        ith = 0
        while True:
            k = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            v = torch.frombuffer(memoryview(self.sock.recv(copy=False)), dtype=torch.float16)
            if ith == self.num_layers-1:
                ith = 0
                logger.info(f"end receiving kv cache")
                self.result_queue.put(b'0x01')
            ith += 1
