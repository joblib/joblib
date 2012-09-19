"""Fixture module to skip memmaping test if numpy is not installed"""

from nose import SkipTest

def setup_module(module):
    try:
        import numpy as np
        import multiprocessing
    except ImportError:
        raise SkipTest('Skipped as numpy or multiprocessing is not installed')
