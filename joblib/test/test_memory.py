"""
Test the memory module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
import os
from tempfile import mkdtemp

import nose

from ..memory import Memory

################################################################################
# Module-level variables for the tests
def f(x, y=1):
    """ A module-level function for testing purposes.
    """
    return x**2 + y

cachedir = None

################################################################################
# Test fixtures
def setup():
    """ Test setup.
    """
    global cachedir
    cachedir = mkdtemp()
    #cachedir = 'foobar'
    if os.path.exists(cachedir):
        shutil.rmtree(cachedir)
    os.makedirs(cachedir)
    

def teardown():
    """ Test teardown.
    """
    #return True
    shutil.rmtree(cachedir)


################################################################################
# Tests
def test_memory_integration():
    """ Simple tests of memory features.
    """
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
                    (accumulator_f, accumulator_g), (f, g)):
                nose.tools.assert_equal(func(i), i)
                nose.tools.assert_equal(len(accumulator), i + 1)


    # Test for an explicit keyword argument:
    nose.tools.assert_equal(g(l=30, m=2), 30)


def test_func_dir():
    """ Test the creation of the memory cache directory for the function.
    """
    memory = Memory(cachedir=cachedir)
    path = __name__.split('.')
    path.append('f')
    path = os.path.join(cachedir, *path)

    # Test that the function directory is created on demand
    yield nose.tools.assert_equal, memory._get_func_dir(f), path
    
    # Test that the code is stored.
    yield nose.tools.assert_false, \
        memory._check_previous_func_code(f)
    yield nose.tools.assert_true, \
            os.path.exists(os.path.join(path, 'func_code.py'))
    yield nose.tools.assert_true, \
        memory._check_previous_func_code(f)

