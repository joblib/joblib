from .. import Parallel, delayed, parallel_backend
from time import sleep

import pytest


def inc(x):
    return x + 1


def slow_raise_value_error(condition, duration=0.05):
    sleep(duration)
    if condition:
        raise ValueError("condition evaluated to True")


def test_simple():
    with parallel_backend('dask') as (ba, _):
        seq = Parallel(n_jobs=5)(delayed(inc)(i) for i in range(10))
        assert seq == [inc(i) for i in range(10)]

    with pytest.raises(BaseException):
        Parallel(n_jobs=5)(delayed(slow_raise_value_error)(i == 3)
                           for i in range(10))

