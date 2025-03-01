import faulthandler
import logging
import os
import sys

import pytest
from _pytest.doctest import DoctestItem

from joblib import Memory
from joblib.backports import LooseVersion
from joblib.parallel import ParallelBackendBase, mp

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
            # Only run doctests for numpy >= 2 and Python >= 3.10 to avoid
            # formatting changes
            import numpy as np

            if LooseVersion(np.__version__) >= LooseVersion("2") and sys.version_info[
                :2
            ] >= (3, 10):
                skip_doctests = False
        except ImportError:
            pass

    if skip_doctests:
        reason = (
            "doctests are only run in some conditions, see conftest.py for more details"
        )
        skip_marker = pytest.mark.skip(reason=reason)

        for item in items:
            if isinstance(item, DoctestItem):
                item.add_marker(skip_marker)

    if lz4 is None:
        for item in items:
            if item.name == "persistence.rst":
                item.add_marker(pytest.mark.skip(reason="lz4 is missing"))


def pytest_configure(config):
    """Setup multiprocessing logging for the tests"""
    if mp is not None:
        log = mp.util.log_to_stderr(logging.DEBUG)
        log.handlers[0].setFormatter(
            logging.Formatter(
                "[%(levelname)s:%(processName)s:%(threadName)s] %(message)s"
            )
        )

    # Some CI runs failed with hanging processes that were not terminated
    # with the timeout. To make sure we always get a proper trace, set a large
    # enough dump_traceback_later to kill the process with a report.
    faulthandler.dump_traceback_later(30 * 60, exit=True)

    DEFAULT_BACKEND = os.environ.get("JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND", None)
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


@pytest.fixture(scope="function")
def memory(tmp_path):
    "Fixture to get an independent and self-cleaning Memory"
    mem = Memory(location=tmp_path, verbose=0)
    yield mem
    mem.clear()


@pytest.fixture(scope="function", autouse=True)
def avoid_env_var_leakage():
    "Fixture to avoid MAX_NUM_THREADS env vars leakage between tests"
    yield
    assert all(
        os.environ.get(k) is None for k in ParallelBackendBase.MAX_NUM_THREADS_VARS
    )
