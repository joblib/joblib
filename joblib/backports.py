"""
Backports of fixes for joblib dependencies
"""
import os
import time
import weakref

from distutils.version import LooseVersion

from .numpy_pickle_utils import _get_backing_memmap


def maybe_unlink(filename, rtype):
    from .externals.loky.backend.resource_tracker import _resource_tracker
    print(
        "[FINALIZER CALL] object mapping to {} about to be deleted,"
        " decrementing the refcount of the file (pid: {})\n".format(
            filename, os.getpid()))
    _resource_tracker.maybe_unlink(filename, rtype)


try:
    import numpy as np

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C'):
        """Backport of numpy memmap offset fix.

        See https://github.com/numpy/numpy/pull/8443 for more details.

        The numpy fix will be available in numpy 1.13.
        """
        mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset,
                       shape=shape, order=order)
        if LooseVersion(np.__version__) < '1.13':
            mm.offset = offset
        # TODO: add a verbose parameter or remove these print statements
        # before merging
        print(
            "[MEMMAP READ] reading a memmap (shape {}, filename {}, "
            "pid {})\n".format(shape, filename.split('/')[-1], os.getpid())
        )

        mmap_obj = _get_backing_memmap(mm)
        print(
            "[FINALIZER ADD] about to add a finalizer to a {} (id {}, "
            "filename {}, pid {}, time {})\n".format(
                type(mmap_obj.base), id(mmap_obj.base), mmap_obj.filename,
                os.getpid(), time.time()))

        if mmap_obj.base is None:
            raise ValueError(
                "mmap base of a np.memmap object should not be None")
        weakref.finalize(mmap_obj.base, maybe_unlink, filename, "file")
        return mm
except ImportError:
    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C'):
        raise NotImplementedError(
            "'joblib.backports.make_memmap' should not be used "
            'if numpy is not installed.')


if os.name == 'nt':
    # https://github.com/joblib/joblib/issues/540
    access_denied_errors = (5, 13)
    from os import replace

    def concurrency_safe_rename(src, dst):
        """Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.

        On Windows os.replace can yield permission errors if executed by two
        different processes.
        """
        max_sleep_time = 1
        total_sleep_time = 0
        sleep_time = 0.001
        while total_sleep_time < max_sleep_time:
            try:
                replace(src, dst)
                break
            except Exception as exc:
                if getattr(exc, 'winerror', None) in access_denied_errors:
                    time.sleep(sleep_time)
                    total_sleep_time += sleep_time
                    sleep_time *= 2
                else:
                    raise
        else:
            raise
else:
    from os import replace as concurrency_safe_rename  # noqa
