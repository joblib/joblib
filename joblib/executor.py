"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""
# Author: Thomas Moreau <thomas.moreau.2010@gmail.com>
# Copyright: 2017, Thomas Moreau
# License: BSD 3 clause

from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor


_executor_args = None


def get_memmapping_executor(n_jobs, **kwargs):
    return MemmappingExecutor.get_memmapping_executor(n_jobs, **kwargs)


class MemmappingExecutor(_ReusablePoolExecutor):

    @classmethod
    def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None,
                                initargs=(), env=None, temp_folder=None,
                                context_id=None, **backend_args):
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

        manager = TemporaryResourcesManager(
            temp_folder,
            context_id=context_id
        )

        # reducers access the temporary folder in which to store temporary
        # pickles through a call to manager.resolve_temp_folder_name. resolving
        # the folder name dynamically is useful to use different folders across
        # calls of a same reusable executor
        job_reducers, result_reducers = get_memmapping_reducers(
            unlink_on_gc_collect=True,
            temp_folder_resolver=manager.resolve_temp_folder_name,
            **backend_args)
        _executor, executor_is_reused = super().get_reusable_executor(
            n_jobs, job_reducers=job_reducers, result_reducers=result_reducers,
            reuse=reuse, timeout=timeout, initializer=initializer,
            initargs=initargs, env=env
        )

        if not executor_is_reused:
            # if _executor is new, the previously created manager will used by
            # the reducer to resolve temporary folder names. Otherwise, we
            # musn't patch it, because the reducers will use the manager
            # instance created by an older `get_memmaping_exeuctor` call.
            _executor._temp_folder_manager = manager

        return _executor

    def terminate(self, kill_workers=False):
        self.shutdown(kill_workers=kill_workers)
        if kill_workers:
            # When workers are killed in such a brutal manner, they cannot
            # execute the finalizer of their shared memmaps. The refcount of
            # those memmaps may be off by an unknown number, so instead of
            # decref'ing them, we delete the whole temporary folder, and
            # unregister them. There is no risk of PermissionError at folder
            # deletion because because at this point, all child processes are
            # dead, so all references to temporary memmaps are closed.

            # unregister temporary resources from all contexts
            self._temp_folder_manager._unregister_temporary_resources()
            self._temp_folder_manager._try_delete_folder(allow_non_empty=True)
        else:
            self._temp_folder_manager._unlink_temporary_resources()

    @property
    def _temp_folder(self):
        # Legacy property in tests. could be removed if we refactored the
        # memmapping tests.
        return self._temp_folder_manager.resolve_temp_folder_name()


class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """
    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self.submit(func, *args)
        future.get = future.result
        return future

    def map(self, f, *args):
        return list(super().map(f, *args))
