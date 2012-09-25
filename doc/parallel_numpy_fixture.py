"""Fixture module to skip memmaping test if numpy is not installed"""

from nose import SkipTest
from joblib.test.common import setup_autokill
from joblib.test.common import teardown_autokill


def setup_module(module):
    try:
        import numpy as np
        import multiprocessing
    except ImportError:
        raise SkipTest('Skipped as numpy or multiprocessing is not installed')

    setup_autokill(module.__name__)


def teardown_autokill(module):
    teardown_autokill(module.__name__)
