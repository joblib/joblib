import cloudpickle
from multiprocessing.pool import ThreadPool
from .parallel import ParallelBackendBase


class DaskDistributedBackend(ParallelBackendBase):

    def __init__(self, sparkContext):
        self.sparkContext = sparkContext
        self._n_jobs = self.effective_n_jobs(10)

    def effective_n_jobs(self, n_jobs):
        num_slots = 10 # TODO: get real task slots
        return min(n_jobs, num_slots)

    def abort_everything(self, ensure_ready=True):
        raise RuntimeError("unsupported abort")

    def start_call(self):
        pass

    def stop_call(self):
        pass

    def configure(self, n_jobs=10, parallel=None, **backend_args):
        n_jobs = self.effective_n_jobs(n_jobs)
        self._n_jobs = n_jobs
        return n_jobs

    def _get_pool(self):
        """Lazily initialize the thread pool

        The actual pool of worker threads is only initialized at the first
        call to apply_async.
        """
        if self._pool is None:
            self._pool = ThreadPool(self._n_jobs)
        return self._pool

    def apply_async(self, func, callback=None):

        def run_on_worker_and_fetch_result():
            ser_res = self.sparkContext.parallelize([0], 1) \
                .map(lambda _: cloudpickle.dumps(func())) \
                .first()
            return cloudpickle.loads(ser_res)

        self._get_pool().apply_async(
            run_on_worker_and_fetch_result,
            callback=callback
        )
