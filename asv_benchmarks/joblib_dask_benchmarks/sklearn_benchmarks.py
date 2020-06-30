import numpy as np
from distributed import LocalCluster, Client
from joblib import Parallel, delayed, parallel_backend
from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


class TimeSklearnBenchmarks:

    repeat = 1
    number = 1

    param_names = ["backend", "n_workers", "threads_per_worker"]
    params = (["threading", "loky", "dask"][2:], [4], [1])

    def setup(self, backend, n_workers, threads_per_worker):
        if backend == "dask":
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=True,
            )
            client = Client(cluster)
            self.cluster, self.client = cluster, client
        elif backend == "loky":
            # pre-warm the loky executor
            _ = Parallel(n_jobs=n_workers * threads_per_worker)(
                delayed(id)(i) for i in range(100)
            )

        self.backend_kwargs = {
            "backend": backend,
            "n_jobs": n_workers * threads_per_worker,
        }
        self.large_array = (
            np.ones(int(1e8)).astype(np.int8).reshape(int(1e7), 10)
        )
        X, y = make_classification(n_samples=1000, n_features=10)
        self.X, self.y = X, y

    def time_gridsearch_cv(self, backend, n_workers, threads_per_worker):
        param_grid = {'C': [0.1, 0.5, 1, 5, 10][:1]}
        base_estimator = LogisticRegression()
        clf = GridSearchCV(estimator=base_estimator, param_grid=param_grid,
                           n_jobs=-1, cv=2, verbose=10000)
        with parallel_backend('dask'):
            # Parallel(verbose=10000)(delayed(id)(x) for x in range(100))
            clf.fit(self.X, self.y)
