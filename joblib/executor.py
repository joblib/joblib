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
                                initargs=(), env=None, **backend_args):
        """Factory for ReusableExecutor with automatic memmapping for large numpy
        arrays.
        """
        global _executor_args
        executor_args = backend_args.copy()
        executor_args.update(env if env else {})
        executor_args.update(dict(
            timeout=timeout, initializer=initializer, initargs=initargs))
        reuse = _executor_args == executor_args
        _executor_args = executor_args
        if reuse:
            return super().get_reusable_executor(
                n_jobs, reuse=reuse, timeout=timeout, initializer=initializer,
                initargs=initargs, env=env
            )
        else:
            # only create reducers (and configure a new temporary folder) if a
            # new executor has to be created
            id_executor = random.randint(0, int(1e10))
            temp_folder, use_shared_mem = cls.get_temp_dir(
                backend_args.pop('temp_folder', None), id_executor
            )
            job_reducers, result_reducers = get_memmapping_reducers(
                unlink_on_gc_collect=True, temp_folder=temp_folder,
                **backend_args)
            _executor = super().get_reusable_executor(
                n_jobs, job_reducers=job_reducers,
                result_reducers=result_reducers, reuse=reuse, timeout=timeout,
                initializer=initializer, initargs=initargs, env=env
            )
            # The whole temporary folder configuration would be less awkward if
            # we:
            # - first create the reducers without any info about the temp
            #   folder
            # - then create the executor
            # - then create a temp folder
            # - then "bind" the temp folder to the executor and the reducers.
            _executor._setup_temp_dir_tracking(
                temp_folder, delete_folder_upon_gc=True
            )
            return _executor


class _TestingMemmappingExecutor(TemporaryResourcesManagerMixin):
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
        self._unlink_temporary_resources(delete_folder=True)

    def map(self, f, *args):
        res = self._executor.map(f, *args)
        return list(res)
