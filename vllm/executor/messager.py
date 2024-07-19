class ProcessWorkerWrapper:
    """Local process wrapper for vllm.worker.Worker,
    for handling single-node multi-GPU tensor parallel."""

    def __init__(self, result_handler: ResultHandler,
                 worker_factory: Callable[[], Any]) -> None:
        self._task_queue = mp.Queue()
        self.result_queue = result_handler.result_queue
        self.tasks = result_handler.tasks
        self.process: BaseProcess = mp.Process(  # type: ignore[attr-defined]
            target=_run_worker_process,
            name="VllmWorkerProcess",
            kwargs=dict(
                worker_factory=worker_factory,
                task_queue=self._task_queue,
                result_queue=self.result_queue,
            ),
            daemon=True)

        self.process.start()

    def _enqueue_task(self, future: Union[ResultFuture, asyncio.Future],
                      method: str, args, kwargs):
        task_id = uuid.uuid4()
        self.tasks[task_id] = future
        try:
            self._task_queue.put((task_id, method, args, kwargs))
        except BaseException as e:
            del self.tasks[task_id]
            raise ChildProcessError("worker died") from e

    def execute_method(self, method: str, *args, **kwargs):
        future: ResultFuture = ResultFuture()
        self._enqueue_task(future, method, args, kwargs)
        return future

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
