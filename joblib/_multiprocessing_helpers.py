"""Helper module to factorize the conditional multiprocessing import logic

We use a distinct module to simplify import statements and avoid introducing
circular dependencies (for instance for the assert_spawning name).
"""
import os
import sys
import warnings

from ._compat import CompatFileExistsError


# Obtain possible configuration from the environment, assuming 1 (on)
# by default, upon 0 set to None. Should instructively fail if some non
# 0/1 value is set.
mp = int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)) or None
if mp:
    try:
        import multiprocessing as mp
    except ImportError:
        mp = None

# 2nd stage: validate that locking is available on the system and
#            issue a warning if not
if mp is not None:
    try:
        # try to create a named semaphore using SemLock to make sure they are
        # available on this platform. We use the low level object
        # _multiprocessing.SemLock to avoid spawning a resource tracker on
        # Unix system or changing the default backend.
        import tempfile
        from _multiprocessing import SemLock
        if sys.version_info < (3,):
            _SemLock = SemLock

            def SemLock(kind, value, maxvalue, name, unlink):
                return _SemLock(kind, value, maxvalue)

        _rand = tempfile._RandomNameSequence()
        for i in range(100):
            try:
                name = '/joblib-{}-{}' .format(
                    os.getpid(), next(_rand))
                _sem = SemLock(0, 0, 1, name=name, unlink=True)
                del _sem  # cleanup
                break
            except CompatFileExistsError:  # pragma: no cover
                if i >= 99:
                    raise CompatFileExistsError(
                        'cannot find name for semaphore')
    except (CompatFileExistsError, AttributeError, ImportError, OSError) as e:
        mp = None
        warnings.warn('%s.  joblib will operate in serial mode' % (e,))


# 3rd stage: backward compat for the assert_spawning helper
if mp is not None:
    try:
        # Python 3.4+
        from multiprocessing.context import assert_spawning
    except ImportError:
        from multiprocessing.forking import assert_spawning
else:
    assert_spawning = None
