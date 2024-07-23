import asyncio
import multiprocessing
import os
import sys
import torch
import time
import threading
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.connection import wait
from multiprocessing.process import BaseProcess
from typing import (Any, Callable, Dict, Generic, List, Optional, TextIO, Tuple,
                    TypeVar, Union)

from vllm.worker.messager import KVPusher, KVPuller
from vllm.logger import init_logger

KVCache = Tuple[torch.Tensor, torch.Tensor]

logger = init_logger(__name__)

T = TypeVar('T')

_TERMINATE = "TERMINATE"  # sentinel

# ANSI color codes
CYAN = '\033[1;36m'
RESET = '\033[0;0m'

JOIN_TIMEOUT_S = 2

mp_method = "spawn"
mp = multiprocessing.get_context(mp_method)

@dataclass
class Result(Generic[T]):
    """Result of task dispatched to worker"""

    task_id: uuid.UUID
    value: Optional[T] = None
    exception: Optional[BaseException] = None


class ResultFuture(threading.Event, Generic[T]):
    """Synchronous future for non-async case"""

    def __init__(self):
        super().__init__()
        self.result: Optional[Result[T]] = None

    def set_result(self, result: Result[T]):
        self.result = result
        self.set()

    def get(self) -> T:
        self.wait()
        assert self.result is not None
        if self.result.exception is not None:
            raise self.result.exception
        return self.result.value  # type: ignore[return-value]


def _set_future_result(future: Union[ResultFuture, asyncio.Future],
                       result: Result):
    if isinstance(future, ResultFuture):
        future.set_result(result)
        return
    loop = future.get_loop()
    if not loop.is_closed():
        if result.exception is not None:
            loop.call_soon_threadsafe(future.set_exception, result.exception)
        else:
            loop.call_soon_threadsafe(future.set_result, result.value)


class ResultHandler(threading.Thread):
    """Handle results from all workers (in background thread)"""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.result_queue = mp.Queue()
        self.tasks: Dict[uuid.UUID, Union[ResultFuture, asyncio.Future]] = {}

    def run(self):
        for result in iter(self.result_queue.get, _TERMINATE):
            future = self.tasks.pop(result.task_id)
            _set_future_result(future, result)
        # Ensure that all waiters will receive an exception
        for task_id, future in self.tasks.items():
            _set_future_result(
                future,
                Result(task_id=task_id,
                       exception=ChildProcessError("worker died")))

    def close(self):
        self.result_queue.put(_TERMINATE)


class WorkerMonitor(threading.Thread):
    """Monitor worker status (in background thread)"""

    def __init__(self, workers: List['ProcessWorkerWrapper'],
                 result_handler: ResultHandler):
        super().__init__(daemon=True)
        self.workers = workers
        self.result_handler = result_handler
        self._close = False

    def run(self) -> None:
        # Blocks until any worker exits
        dead_sentinels = wait([w.process.sentinel for w in self.workers])
        if not self._close:
            self._close = True

            # Kill / cleanup all workers
            for worker in self.workers:
                process = worker.process
                if process.sentinel in dead_sentinels:
                    process.join(JOIN_TIMEOUT_S)
                if process.exitcode is not None and process.exitcode != 0:
                    logger.error("Worker %s pid %s died, exit code: %s",
                                 process.name, process.pid, process.exitcode)
            # Cleanup any remaining workers
            logger.info("Killing local vLLM worker processes")
            for worker in self.workers:
                worker.kill_worker()
            # Must be done after worker task queues are all closed
            self.result_handler.close()

        for worker in self.workers:
            worker.process.join(JOIN_TIMEOUT_S)

    def close(self):
        if self._close:
            return
        self._close = True
        logger.info("Terminating local vLLM worker processes")
        for worker in self.workers:
            worker.terminate_worker()
        # Must be done after worker task queues are all closed
        self.result_handler.close()


class MessagerWrapper:
    """Local process wrapper for vllm.worker.Worker,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, role: str, num_layers: int, result_handler: ResultHandler) -> None:
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.process: BaseProcess = mp.Process(  # type: ignore[attr-defined]
            target=_run_worker_process,
            name="messager",
            kwargs=dict(
                task_queue=self._task_queue,
                result_queue=self.result_queue,
                role=role,
                num_layers=num_layers,
            ),
            daemon=True)

        self.process.start()

    def pass_cache(self, gpu_cache: List[KVCache], cpu_cache: List[KVCache], kv_buffer: List[KVCache]):
        try:
            self._task_queue.put(gpu_cache)
            self._task_queue.put(cpu_cache)
            self._task_queue.put(kv_buffer)
        except BaseException as e:
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: str, *args, **kwargs):
        try:
            # s = time.perf_counter()
            self._task_queue.put((method, args, kwargs))
            # print(f"{method} {1000 * (time.perf_counter() - s)} ms", file=sys.stderr)
        except BaseException as e:
            raise ChildProcessError("worker died") from e

    async def execute_method_async(self, method: str, *args, **kwargs):
        future = asyncio.get_running_loop().create_future()
        self._enqueue_task(future, method, args, kwargs)
        return await future

    def terminate_worker(self):
        try:
            self._task_queue.put(_TERMINATE)
        except ValueError:
            self.process.kill()
        self._task_queue.close()

    def kill_worker(self):
        self._task_queue.close()
        self.process.kill()


def _run_worker_process(
    task_queue: Queue,
    result_queue: Queue,
    role: str,
    num_layers: int,
) -> None:
    """Worker process event loop"""

    # Add process-specific prefix to stdout and stderr
    process_name = mp.current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)

    gpu_cache = task_queue.get()
    cpu_cache = task_queue.get()
    kv_buffer = task_queue.get()
    logger.info("gpu and cpu cache initilized in messager")
    # Initialize worker
    if role == 'pusher':
        messager = KVPusher(gpu_cache, cpu_cache, kv_buffer)
    elif role == 'puller':
        messager = KVPuller(num_layers, result_queue, gpu_cache, cpu_cache, kv_buffer)

    # Accept tasks from the engine in task_queue
    # and return task output in result_queue
    logger.info(f"{role} ready; awaiting tasks")
    try:
        for items in iter(task_queue.get, _TERMINATE):
            method, args, kwargs = items
            try:
                executor = getattr(messager, method)
                executor(*args, **kwargs)
            except BaseException as e:
                tb = traceback.format_exc()
                logger.error(
                    "Exception in worker %s while processing method %s: %s, %s",
                    process_name, method, e, tb)
    except KeyboardInterrupt:
        pass
    except Exception:
        logger.exception("Worker failed")

    logger.info("Worker exiting")

def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Prepend each output line with process-specific prefix"""

    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find('\n', idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]
