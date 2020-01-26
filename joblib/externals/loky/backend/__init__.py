import os
import sys

from .context import get_context

if sys.version_info > (3, 4):

    def _make_name():
        name = '/loky-%i-%s' % (os.getpid(), next(synchronize.SemLock._rand))
        return name

    # Handle cases where semaphores are not supported
    try:
        # monkey patch the name creation for multiprocessing
        from multiprocessing import synchronize
        synchronize.SemLock._make_name = staticmethod(_make_name)
    except ImportError as import_error:
        # sys.stderr.write(f"Disabling loky support: {import_error}")
        pass

__all__ = ["get_context"]
