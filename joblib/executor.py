"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Copyright: 2017, Thomas Moreau
# License: BSD 3 clause

import os
import random
from ._memmapping_reducer import get_memmapping_reducers, JOBLIB_MMAPS
from .externals.loky.reusable_executor import get_reusable_executor
from .externals.loky.backend import resource_tracker
from .disk import delete_folder


_executor_args = None


def get_memmapping_executor(n_jobs, timeout=300, initializer=None, initargs=(),
                            env=None, **backend_args):
    """Factory for ReusableExecutor with automatic memmapping for large numpy
    arrays.
    """
    global _executor_args
    # Check if we can reuse the executor here instead of deferring the test to
    # loky as the reducers are objects that changes at each call.
    executor_args = backend_args.copy()
    executor_args.update(env if env else {})
    executor_args.update(dict(
        timeout=timeout, initializer=initializer, initargs=initargs))
    reuse = _executor_args is None or _executor_args == executor_args
    _executor_args = executor_args

    id_executor = random.randint(0, int(1e10))
    job_reducers, result_reducers, temp_folder = get_memmapping_reducers(
        id_executor, track_memmap_in_child=True, **backend_args)
    _executor = get_reusable_executor(n_jobs, job_reducers=job_reducers,
                                      result_reducers=result_reducers,
                                      reuse=reuse, timeout=timeout,
                                      initializer=initializer,
                                      initargs=initargs, env=env)
    # If executor doesn't have a _temp_folder, it means it is a new executor
    # and the reducers have been used. Else, the previous reducers are used
    # and we should not change this attibute.
    if not hasattr(_executor, "_temp_folder"):
        _executor._temp_folder = temp_folder
    return _executor


class _TestingMemmappingExecutor():
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.
    """
    def __init__(self, n_jobs, **backend_args):
        self._executor = get_memmapping_executor(n_jobs, **backend_args)
        self._temp_folder = self._executor._temp_folder

    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self._executor.submit(func, *args)
        future.get = future.result
        return future

    def terminate(self):
        self._executor.shutdown()
        if os.path.exists(self._temp_folder):
            for filename in os.listdir(self._temp_folder):
                resource_tracker.maybe_unlink(
                    os.path.join(self._temp_folder, filename), "file"
                )
            JOBLIB_MMAPS.clear()
            try:
                delete_folder(self._temp_folder, allow_non_empty=False)
            except OSError:
                pass

    def map(self, f, *args):
        res = self._executor.map(f, *args)
        return list(res)
