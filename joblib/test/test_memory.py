"""
Test the memory module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
from tempfile import mkdtemp

from ..memory import Memory


def test_memory():
    """ Simple tests of memory features.
    """
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir)
    accumulator_f = list()
    accumulator_g = list()

    @memory.cache
    def f(l):
        " Function to test simple memoize "
        accumulator_f.append(1)
        return l

    @memory.cache
    def g(l=None, m=1):
        " Function to test memoize with keyword arguments "
        accumulator_g.append(1)
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



