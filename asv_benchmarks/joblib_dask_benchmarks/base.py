import socket

from joblib import Parallel, delayed
from distributed import Client

from .utils import create_dask_cluster


class BenchmarkBase:

    param_names = ["backend", "n_workers", "threads_per_worker"]
    params = (["threading", "loky", "dask"], [4], [1])

    repeat = 1
    number = 1

    def _setup_backend(
        self, backend: str, n_workers: int, threads_per_worker: int
    ):
        if backend == "dask":
            cluster = create_dask_cluster(
                use_slurm="margaret" in socket.gethostname(),
                n_workers=n_workers,
                threads_per_worker=threads_per_worker
            )
            client = Client(cluster)
            self.cluster, self.client = cluster, client

        elif backend == "loky":
            # pre-warm the loky executor
            _ = Parallel(n_jobs=n_workers * threads_per_worker)(
                delayed(id)(i) for i in range(100)
            )
        elif backend == "threading":
            pass
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.backend_kwargs = {
            "backend": backend,
            "n_jobs": n_workers * threads_per_worker,
        }
