"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Copyright: 2017, Thomas Moreau
# License: BSD 3 clause

import random
from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManagerMixin
from .externals.loky.reusable_executor import _ReusablePoolExecutor


_executor_args = None


def get_memmapping_executor(n_jobs, **kwargs):
    return MemmappingExecutor.get_memmapping_executor(n_jobs, **kwargs)


class MemmappingExecutor(
        _ReusablePoolExecutor, TemporaryResourcesManagerMixin
):

    @classmethod
    def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None,
                                initargs=(), env=None, temp_folder=None,
                                **backend_args):
        """Factory for ReusableExecutor with automatic memmapping for large numpy
        arrays.
        """
        global _executor_args
        # Check if we can reuse the executor here instead of deferring the test
        # to loky as the reducers are objects that changes at each call.
        executor_args = backend_args.copy()
        executor_args.update(env if env else {})
        executor_args.update(dict(
            timeout=timeout, initializer=initializer, initargs=initargs))
        reuse = _executor_args is None or _executor_args == executor_args
        _executor_args = executor_args

        if isinstance(temp_folder, str) or temp_folder is None:
            # backward-compat. joblib codes always uses a
            # TempFolderNameGenerator
            from ._memmapping_reducer import TempFolderNameGenerator
            temp_folder = TempFolderNameGenerator(temp_folder)

        job_reducers, result_reducers = get_memmapping_reducers(
            unlink_on_gc_collect=True, temp_folder_provider=temp_folder,
            **backend_args)
        _executor, _ = super().get_reusable_executor(
            n_jobs, job_reducers=job_reducers, result_reducers=result_reducers,
            reuse=reuse, timeout=timeout, initializer=initializer,
            initargs=initargs, env=env
        )
        # If executor doesn't have a _temp_folder, it means it is a new
        # executor and the reducers have not been used. Else, the previous
        # reducers are used and we should not change this attribute.
        if not hasattr(_executor, "_temp_folder_provider"):
            _executor._temp_folder_provider = temp_folder

        return _executor


class _TestingMemmappingExecutor(TemporaryResourcesManagerMixin):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.
    """
    def __init__(self, n_jobs, **backend_args):
        self._executor = get_memmapping_executor(n_jobs, **backend_args)

    @property
    def _temp_folder(self):
        return self._executor._temp_folder_provider.get_temp_folder_name()

    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self._executor.submit(func, *args)
        future.get = future.result
        return future

    def terminate(self):
        self._executor.shutdown()
        self._unlink_temporary_resources()

    def map(self, f, *args):
        res = self._executor.map(f, *args)
        return list(res)
