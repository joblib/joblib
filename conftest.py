import os

import logging
import faulthandler

import pytest
from _pytest.doctest import DoctestItem

from joblib.parallel import mp
from joblib.backports import LooseVersion
try:
    import lz4
except ImportError:
    lz4 = None
try:
    from distributed.utils_test import loop, loop_in_thread
except ImportError:
    loop = None
    loop_in_thread = None


def pytest_collection_modifyitems(config, items):
    skip_doctests = True

    # We do not want to run the doctests if multiprocessing is disabled
    # e.g. via the JOBLIB_MULTIPROCESSING env variable
    if mp is not None:
        try:
            # numpy changed the str/repr formatting of numpy arrays in 1.14.
            # We want to run doctests only for numpy >= 1.14.
            import numpy as np
            if LooseVersion(np.__version__) >= LooseVersion('1.14'):
                skip_doctests = False
        except ImportError:
            pass

    if skip_doctests:
        skip_marker = pytest.mark.skip(
            reason='doctests are only run for numpy >= 1.14')

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)

    if lz4 is None:
        for item in items:
            if item.name == 'persistence.rst':
                item.add_marker(pytest.mark.skip(reason='lz4 is missing'))


def pytest_configure(config):
    """Setup multiprocessing logging for the tests"""
    if mp is not None:
        log = mp.util.log_to_stderr(logging.DEBUG)
        log.handlers[0].setFormatter(logging.Formatter(
            '[%(levelname)s:%(processName)s:%(threadName)s] %(message)s'))

    # Some CI runs failed with hanging processes that were not terminated
    # with the timeout. To make sure we always get a proper trace, set a large
    # enough dump_traceback_later to kill the process with a report.
    faulthandler.dump_traceback_later(30 * 60, exit=True)

    DEFAULT_BACKEND = os.environ.get(
        "JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND", None
    )
    if DEFAULT_BACKEND is not None:
        print(
            f"Setting joblib parallel default backend to {DEFAULT_BACKEND} "
            "from JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND environment variable"
        )
        from joblib import parallel
        parallel.DEFAULT_BACKEND = DEFAULT_BACKEND


def pytest_unconfigure(config):

    # Setup a global traceback printer callback to debug deadlocks that
    # would happen once pytest has completed: for instance in atexit
    # finalizers. At this point the stdout/stderr capture of pytest
    # should be disabled. Note that we cancel the global dump_traceback_later
    # to waiting for too long.
    faulthandler.cancel_dump_traceback_later()

    # Note that we also use a shorter timeout for the per-test callback
    # configured via the pytest-timeout extension.
    faulthandler.dump_traceback_later(60, exit=True)
