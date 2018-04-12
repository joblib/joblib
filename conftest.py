from distutils.version import LooseVersion

import pytest
from _pytest.doctest import DoctestItem

import logging
from joblib.parallel import mp


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


def pytest_configure(config):
    """Setup multiprocessing logging for the tests"""
    if mp is not None:
        log = mp.util.log_to_stderr(logging.DEBUG)
        log.handlers[0].setFormatter(logging.Formatter(
            '[%(levelname)s:%(processName)s:%(threadName)s] %(message)s'))
