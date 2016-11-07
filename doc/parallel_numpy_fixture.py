"""Fixture module to skip memmaping test if numpy is not installed"""

from joblib.parallel import mp
from joblib.test.common import setup_autokill
from joblib.test.common import teardown_autokill
from joblib.testing import SkipTest


def setup_module(module):
    numpy = None
    try:
        import numpy
    except ImportError:
        pass

    if numpy is None or mp is None:
        raise SkipTest('Skipped as numpy or multiprocessing is not available')

    setup_autokill(module.__name__, timeout=300)


def teardown_module(module):
    teardown_autokill(module.__name__)
