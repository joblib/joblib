"""Fixture module to skip memmaping test if numpy is not installed"""

import os
from nose import SkipTest
from joblib.test.common import setup_autokill
from joblib.test.common import teardown_autokill


def setup_module(module):
    if not int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)):
        raise SkipTest(
            'Skipped as multiprocessing is required to run this doctest.')

    numpy = None
    multiprocessing = int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)) or None
    if multiprocessing:
        try:
            import multiprocessing
        except ImportError:
            multiprocessing = None

    # validate that locking is available on the system and
    # skip test if not
    if multiprocessing:
        try:
            import numpy
            _sem = multiprocessing.Semaphore()
            del _sem  # cleanup
        except (ImportError, OSError):
            multiprocessing = None

    if numpy is None or multiprocessing is None:
        raise SkipTest('Skipped as numpy or multiprocessing is not installed')

    setup_autokill(module.__name__)


def teardown_module(module):
    teardown_autokill(module.__name__)
