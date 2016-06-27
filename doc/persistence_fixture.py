"""Fixture module to skip the persistence doctest with python 2.6."""

from nose import SkipTest
from joblib import _compat


def setup_module(module):
    """Setup module."""
    if _compat.PY26:
        raise SkipTest("Skipping persitence doctest in Python 2.6")
