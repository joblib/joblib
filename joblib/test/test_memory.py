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
    # We use lists to count if functions are being re-evaluated.
    accumulator_f = list()
    accumulator_g = list()

    # Rmk: this function has the same name than a module-level function,
    # thus it serves as a test to see that both are identified
    # as different.
    @memory.cache
    def f(l):
        " Function to test memory "
        accumulator_f.append(1)
        return l

    @memory.cache
    def g(l=None, m=1):
        " Function to test memory with keyword arguments "
        accumulator_g.append(1)
        return l

    accumulator_l = list()
    def helper(x):
        """ A helper function to define l as a lambda.
        """
        accumulator_l.append(1)
        return x

    l = lambda x: helper(x)
    l = memory.cache(l)

    # Call each function with several arguments, and check that it is
    # evaluated only once per argument.
    for i in range(10):
        for _ in range(3):
            for accumulator, func in zip(
                    (accumulator_f, accumulator_g, accumulator_l), 
                    (f, g, l)):
                nose.tools.assert_equal(func(i), i)
                nose.tools.assert_equal(len(accumulator), i + 1)

    # Now test clearing
    memory.clear()
    current_accumulator = len(accumulator_f)
    f(1)
    yield nose.tools.assert_equal, len(accumulator_f), \
                current_accumulator + 1

    # Test for an explicit keyword argument:
    nose.tools.assert_equal(g(l=30, m=2), 30)


def test_memory_exception():
    """ Smoketest the exception handling of Memory. 
    """
    memory = Memory(cachedir=cachedir)
    class MyException(Exception):
        pass

    @memory.cache
    def h(exc=0):
        if exc:
            raise MyException

    # Call once, to initialise the cache
    h()

    for _ in range(3):
        # Call 3 times, to be sure that the Exception is always raised
        yield nose.tools.assert_raises, MyException, h, 1


def test_memory_eval():
    """ Test memory with functions defined on the fly that do not have
        underlying code.
    """
    return 
    memory = Memory(cachedir=cachedir)
    yield nose.tools.assert_equal, l(0), l(0)
 

def test_func_dir():
    """ Test the creation of the memory cache directory for the function.
    """
    memory = Memory(cachedir=cachedir)
    path = __name__.split('.')
    path.append('f')
    path = os.path.join(cachedir, *path)

    # Test that the function directory is created on demand
    yield nose.tools.assert_equal, memory._get_func_dir(f), path
    yield nose.tools.assert_true, os.path.exists(path)

    # Test that the code is stored.
    yield nose.tools.assert_false, \
        memory._check_previous_func_code(f)
    yield nose.tools.assert_true, \
            os.path.exists(os.path.join(path, 'func_code.py'))
    yield nose.tools.assert_true, \
        memory._check_previous_func_code(f)

