"""
Test the memoize module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
from tempfile import mkdtemp

from joblib.memoize import memoize

def _assert_equal(a, b):
    """ Our own version of assert, that is not cleaned in -O mode.
    """
    if not a == b:
        raise AssertionError, "%s != %s" % (a, b)


def test_memoize():
    """ Simple tests of the memoize feature.
    """
    cachedir = mkdtemp()
    accumulator_f = list()
    accumulator_g = list()
    accumulator_h = list()

    @memoize(cachedir=cachedir, debug=True)
    def f(l):
        " Function to test simple memoize "
        accumulator_f.append(1)
        return l

    @memoize(cachedir=cachedir, debug=True)
    def g(l=None, m=1):
        " Function to test memoize with keyword arguments "
        accumulator_g.append(1)
        return l

    @memoize(persist=False)
    def h(l):
        " Function to test memoize without persistence "
        accumulator_h.append(1)
        return l

    for i in range(10):
        for j in range(3):
            for accumulator, func in zip(
                    (accumulator_f, accumulator_g, accumulator_h), (f, g, h)):
                _assert_equal(func(i), i)
                _assert_equal(len(accumulator), i + 1)


    # Test for an explicit keyword argument:
    _assert_equal(g(l=30, m=2), 30)

    shutil.rmtree(cachedir)


def test_memoize_unhashable():
    """ Test memoize with some unpickable object.
    """
    cachedir = mkdtemp()
    accumulator = list()

    class A(object):
        " This class cannot be pickled, as it it not importable"
        a = 1

    @memoize(cachedir=cachedir, debug=True)
    def f(l):
        " Function to test simple memoize "
        accumulator.append(1)
        return l

    a = A()
    _assert_equal(f(a), a)
    _assert_equal(f(a), a)
    # Unpickable object, memoize should not have cached, and the function
    # is ran twice.
    _assert_equal(len(accumulator), 2)

    shutil.rmtree(cachedir)

def test_memoize_file_error():
    """ Test memoize while removing its cache in the middle of the its
        work.    
    """
    cachedir = mkdtemp()
    accumulator = list()
    # Be mean, remove the cache directory.
    shutil.rmtree(cachedir)

    @memoize(cachedir=cachedir, debug=True)
    def f(l):
        " Function to test simple memoize "
        accumulator.append(1)
        return l

    for i in range(10):
        _assert_equal(f(i), i)
        _assert_equal(f(i), i)
    
    _assert_equal(len(accumulator), 10)

    shutil.rmtree(cachedir)

