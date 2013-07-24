"""Fixture module to skip memmaping test if numpy is not installed"""

import os
from nose import SkipTest
from joblib.test.common import setup_autokill
from joblib.test.common import teardown_autokill


def setup_module(module):
    if not int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)):
        raise SkipTest(
            'Skipped as multiprocessing is required to run this doctest.')
    try:
        import numpy
        import multiprocessing
    except ImportError:
        raise SkipTest('Skipped as numpy or multiprocessing is not installed')

    setup_autokill(module.__name__)


def teardown_module(module):
    teardown_autokill(module.__name__)
