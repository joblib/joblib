"""Fixture module to skip the persistence doctest with python 2.6."""

from joblib import _compat
from joblib.testing import SkipTest


def setup_module(module):
    """Setup module."""
    if _compat.PY26:
        # gzip.GZipFile and bz2.BZ2File compressor classes cannot be used
        # within a context manager (e.g in a `with` block) in python 2.6 so
        # we skip doctesting of persistence documentation in this case.
        raise SkipTest("Skipping persistence doctest in Python 2.6")
