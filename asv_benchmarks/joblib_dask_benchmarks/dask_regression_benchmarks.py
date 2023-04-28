"""
Collections of reproducers that exhibited bad performance in the past.
"""
import time

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits, fetch_20newsgroups, fetch_openml
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


from joblib import Parallel, delayed, parallel_backend

from .base import BenchmarkBase


class TimeDask5993(BenchmarkBase):
    """Performance bug originally reported in dask/dask#5993"""

    def setup(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        self._setup_backend(backend, n_workers, threads_per_worker)
        digits = load_digits()
        self.X = np.concatenate((digits.data, digits.data), axis=0)
        self.y = np.concatenate((digits.target, digits.target))

    def time_dask_5993(
        self, backend: str, n_workers: int, threads_per_worker: int
    ):
        clf = RandomForestClassifier(n_estimators=2000, verbose=1)
        with parallel_backend(**self.backend_kwargs):
            clf.fit(self.X, self.y)


class TimeJoblib1020(BenchmarkBase):
    """Performance bug originally reported in joblib/joblib#1020"""

    categories = ["alt.atheism", "talk.religion.misc"]

    def setup(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        self._setup_backend(backend, n_workers, threads_per_worker)

        # Scale Up: set categories=None to use all the categories

        print("Loading 20 newsgroups dataset for categories:")
        print(self.categories)

        data = fetch_20newsgroups(subset="train", categories=self.categories)

        pipeline = Pipeline(
            [
                ("vect", HashingVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", SGDClassifier(max_iter=1000)),
            ]
        )

        parameters = {
            "clf__alpha": np.linspace(0.00001, 0.000001, 5),
        }

        # fmt: off
        self.grid_search = GridSearchCV(
            pipeline, parameters, n_jobs=-1, verbose=1, cv=5,
            refit=False, iid=False, pre_dispatch="all"
        )
        # fmt: on
        self.data = data

    def time_joblib_1020(
        self, backend: str, n_workers: int, threads_per_worker: int
    ):
        with parallel_backend(**self.backend_kwargs):
            self.grid_search.fit(self.data.data, self.data.target)


class TimeJoblib957(BenchmarkBase):
    """Reproducer of joblib/joblib#957

    The issue is a race condition during interpreter shutdown, (causing
    ``future.result()``) to fail in a joblib callback.
    """

    def setup(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        self._setup_backend(backend, n_workers, threads_per_worker)

    def time_joblib_957(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        # this issue is not reproducible from the benchmark suite, you need to
        # call this in simple .py script to see errors.
        with parallel_backend(**self.backend_kwargs):
            with Parallel(verbose=1000) as p:
                _ = p(delayed(time.sleep)(1e-5) for _ in range(100))


class TimeJoblib959(BenchmarkBase):
    def setup(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        self._setup_backend(backend, n_workers, threads_per_worker)
        column_transformer = ColumnTransformer(
            transformers=[
                ("NumericalPreprocessing", StandardScaler(), [0, 1, 2, 3],)
            ]
        )
        self.model = GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("Preprocessing", column_transformer),
                    ("Estimator", DecisionTreeClassifier()),
                ],
            ),
            param_grid={"Estimator__min_samples_split": list(range(2, 101))},
        )
        self.X, self.y = fetch_openml(return_X_y=True, data_id=61)

    def time_joblib_959(
        self, backend: str, n_workers: int, threads_per_worker: int
    ) -> None:
        with parallel_backend(**self.backend_kwargs):
            self.model.fit(self.X, self.y)
