"""Helper module to factorize the conditional multiprocessing import logic

We use a distinct module to simplify import statements and avoid introducing
circular dependencies (for instance for the assert_spawning name).
"""
import os
import sys
import warnings

from ._compat import PY27, CompatFileExistsError


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
        if sys.platform == "win32":
            # Use the spawn context
            if PY27:
                Semaphore = mp.Semaphore
            else:
                # Using mp.Semaphore has a border effect and set the default
                # backend for multiprocessing. To avoid that, we use the
                # 'spawn' context which is available on all supported platforms
                ctx = mp.get_context('spawn')
                Semaphore = ctx.Semaphore
            _sem = Semaphore()
        else:
            # try to create a named semaphore using
            from joblib.externals.loky.backend.semlock import SemLock
            for i in range(100):
                try:
                    name = '/loky-{}-{}' .format(
                        os.getpid(), next(SemLock._rand))
                    _sem = SemLock(0, 1, None, name=name, unlink_now=True)
                except CompatFileExistsError:  # pragma: no cover
                    pass
                else:
                    break
            else:  # pragma: no cover
                raise CompatFileExistsError(
                    'cannot find name for semaphore')
            del _sem  # cleanup
    except (AttributeError, CompatFileExistsError, ImportError, OSError) as e:
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
