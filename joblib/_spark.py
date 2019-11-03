import cloudpickle
from multiprocessing.pool import ThreadPool
from .parallel import ParallelBackendBase

from pyspark.sql import SparkSession

from ._parallel_backends import SequentialBackend
from .logger import Logger
import warnings


class SparkDistributedBackend(ParallelBackendBase):

    def __init__(self, **backend_args):
        super(SparkDistributedBackend, self).__init__(**backend_args)
        Logger.__init__(self)
        self._pool = None
        self._n_jobs = None
        self._spark = SparkSession \
            .builder \
            .appName("JoblibSparkBackend") \
            .getOrCreate()

    def effective_n_jobs(self, n_jobs):
        # TODO: get real task slots (i.e maximumn parallel spark jobs number)
        num_slots = 128
        return min(n_jobs, num_slots)

    def abort_everything(self, ensure_ready=True):
        # TODO: Current pyspark has some issue on cancelling jobs, so do not support it
        #       for now
        warnings.warn("Do not implement abort_everything. Spark jobs may not exit.")
        pass

    def start_call(self):
        pass

    def stop_call(self):
        pass

    def terminate(self):
        warnings.warn("Joblib Spark backend stop. Running jobs will be killed.")
        self._spark.stop()

    def configure(self, n_jobs=1, parallel=None, **backend_args):
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

    def compute_batch_size(self):
        """Determine the optimal batch size"""
        return 2

    def apply_async(self, func, callback=None):
        # Note the `func` args is a batch here. (BatchedCalls type)
        # See joblib.parallel.Parallel._dispatch
        def run_on_worker_and_fetch_result():
            ser_res = self._spark.sparkContext.parallelize([0], 1) \
                .map(lambda _: cloudpickle.dumps(func())) \
                .first()
            return cloudpickle.loads(ser_res)

        return self._get_pool().apply_async(
            run_on_worker_and_fetch_result,
            callback=callback
        )

    def get_nested_backend(self):
        """Backend instance to be used by nested Parallel calls.
        """
        nesting_level = getattr(self, 'nesting_level', 0) + 1
        return SequentialBackend(nesting_level=nesting_level), None
