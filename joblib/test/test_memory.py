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
# Helper function for the tests
def check_identity_lazy(func, accumulator):
    """ Given a function and an accumulator (a list that grows every
        time the function is called, check that the function can be
        decorated by memory to be a lazy identity.
    """
    # Call each function with several arguments, and check that it is
    # evaluated only once per argument.
    memory = Memory(cachedir=cachedir)
    memory.clear()
    func = memory.cache(func)
    for i in range(3):
        for _ in range(2):
            yield nose.tools.assert_equal, func(i), i
            yield nose.tools.assert_equal, len(accumulator), i + 1


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
    """ Simple test of memory lazy evaluation.
    """
    accumulator = list()
    # Rmk: this function has the same name than a module-level function,
    # thus it serves as a test to see that both are identified
    # as different.
    def f(l):
        accumulator.append(1)
        return l

    for test in check_identity_lazy(f, accumulator):
        yield test

    # Now test clearing
    memory = Memory(cachedir=cachedir)
    memory.clear()
    current_accumulator = len(accumulator)
    f(1)
    yield nose.tools.assert_equal, len(accumulator), \
                current_accumulator + 1


def test_memory_kwarg():
    " Test memory with a function with keyword arguments."
    accumulator = list()
    def g(l=None, m=1):
        accumulator.append(1)
        return l

    for test in check_identity_lazy(g, accumulator):
        yield test

    memory = Memory(cachedir=cachedir)
    g = memory.cache(g)
    # Smoke test with an explicit keyword argument:
    nose.tools.assert_equal(g(l=30, m=2), 30)


def test_memory_lambda():
    " Test memory with a function with a lambda."
    accumulator = list()
    def helper(x):
        """ A helper function to define l as a lambda.
        """
        accumulator.append(1)
        return x

    l = lambda x: helper(x)

    for test in check_identity_lazy(l, accumulator):
        yield test


def test_memory_eval():
    " Smoke test memory with a function with a function defined in an eval."
    memory = Memory(cachedir=cachedir)

    m = eval('lambda x: x')

    yield nose.tools.assert_equal, 1, m(1)


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


def test_func_dir():
    """ Test the creation of the memory cache directory for the function.
    """
    memory = Memory(cachedir=cachedir)
    path = __name__.split('.')
    path.append('f')
    path = os.path.join(cachedir, *path)

    g = memory.cache(f)
    # Test that the function directory is created on demand
    yield nose.tools.assert_equal, g._get_func_dir(), path
    yield nose.tools.assert_true, os.path.exists(path)

    # Test that the code is stored.
    yield nose.tools.assert_false, \
        g._check_previous_func_code()
    yield nose.tools.assert_true, \
            os.path.exists(os.path.join(path, 'func_code.py'))
    yield nose.tools.assert_true, \
        g._check_previous_func_code()

