"""
Tests for joblib.make.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
from tempfile import mkdtemp
import time

# Local imports
from joblib.make import Serializer, _time_stamp_registry, PickleFile, \
    NumpyFile, make, TimeStamp

def test_serializer():
    """ Test the serializer.

        Check that we can extract non-native type of complex objects.
    """
    serializer = Serializer()
    _assert_equal(serializer.hash((1, 2)), (1, 2))
    a = [1, {'a':None, (1, 2):'foo'}]
    _assert_equal(serializer.hash(a), a)
    _assert_equal(serializer.reference_registry.id_table.keys(),
                                                            [id(None)])

    # Check that time_stamp extraction works.
    class A(object):
        pass
    a = A()
    hash = serializer.hash(a)
    assert serializer.reference_registry.latest_reference().time_stamp \
                                                            < time.time()
    assert serializer.reference_registry.latest_reference().time_stamp > 0
    assert isinstance(hash, TimeStamp)
    # check that we don't have a memory leak
    del a
    _assert_equal(serializer.reference_registry.id_table.keys(),
                                                            [id(None)])
    # Check that a non-serialized object still compares as identical to
    # its serialized version.
    a = [1, {'a':None, (1, 2):'foo'}, A()]
    hash = serializer.hash(a)
    _assert_equal(hash, a)

################################################################################
# Test make
def _assert_equal(a, b):
    """ Our own version of assert, that is not cleaned in -O mode.
    """
    if not a == b:
        raise AssertionError, "%s != %s" % (a, b)

def test_make_non_mutable():
    """ Test the make on non-mutable inputs and outputs.
    """
    cachedir = mkdtemp()
    output_accumulator = list()

    @make(cachedir=cachedir, debug=True)
    def test1(l, m=None):
        output_accumulator.append(l)

    test1(1)
    _assert_equal(output_accumulator, [1, ])
    test1(1)
    _assert_equal(output_accumulator, [1, ])
    test1(2)
    _assert_equal(output_accumulator, [1, 2])
    test1(2, m=1)
    _assert_equal(output_accumulator, [1, 2, 2])

    shutil.rmtree(cachedir)


def test_make_persistence():
    """ Test the simplest possible persistence: using a PickleFile.
    """
    cachedir = mkdtemp()

    @make(cachedir=cachedir, output=PickleFile(cachedir+'/f_output'), 
                                                                debug=True)
    def test3(l):
        return l

    obj_list = (1, (1, 2), 'ae', [1, 2], 1.2)
    for obj in obj_list:
        _assert_equal(test3(obj), test3(obj))

    # And test with a lambda function
    test4 = make(func=lambda x:x, cachedir=cachedir, 
                 output=PickleFile(cachedir+'/f_output'),  debug=True)

    for obj in obj_list:
        _assert_equal(test4(obj), test4(obj))


    shutil.rmtree(cachedir)


def test_make_complexe_persistence():
    """ Test persistence using a nested list of persisters.
    """
    cachedir = mkdtemp()

    @make(cachedir=cachedir, debug=True, raise_errors=True,
          output=(PickleFile(cachedir+'/f_output_1'),
                  (1, PickleFile(cachedir+'/f_output_2')))
          )
    def test3(a, b):
        return (a, (1, b))

    _assert_equal(test3('ab', (1, 2, 3)), test3('ab', (1, 2, 3)))

    shutil.rmtree(cachedir)


def test__numpy():
    """ Check that we can persist, load, and perform operations, with
        numpy arrays.
    """
    try:
        import numpy as np
    except ImportError:
        return
    # FIXME: We should check for leaks
    #_time_stamp_registry.__reset__()
    cachedir = mkdtemp()
    output_accumulator = list()
    test_array = np.vander((0, 1, 0, 2, 3))

    @make(cachedir=cachedir, output=NumpyFile(cachedir+'/test.npy'),
                                     debug=True)
    def test(a=None):
        output_accumulator.append(1)
        return test_array

    # Check that a call produces the right output (this is an exception
    # tracer)
    assert np.all(test() == test_array)
    _assert_equal(output_accumulator, [1, ])
    # Check that a second call does not trigger a second run
    assert np.all(test() == test_array)
    _assert_equal(output_accumulator, [1, ])

    # Check that a different input argument produces another call.
    assert np.all(test(1) == test_array)
    _assert_equal(output_accumulator, [1, 1, ])

    # FIXME: We should check for leaks
    #_time_stamp_registry.__reset__()

    @make(cachedir=cachedir, output=(NumpyFile(cachedir+'/test1_1.npy' ), 
                                     NumpyFile(cachedir+'/test2_2.npy')), 
                                     debug=True)
    def test2(x):
        output_accumulator.append(1)
        return x, x**2

    # Check that we can persist numpy arrays
    output_accumulator = list()
    a, b = test2(2)
    assert np.allclose(a, 2)
    assert np.allclose(b, 4)
    _assert_equal(output_accumulator, [1, ])

    a, b = test2(2)
    assert np.allclose(a, 2)
    assert np.allclose(b, 4)
    _assert_equal(output_accumulator, [1, ])

    # Check that when giving a numpy array generated by a function
    # decorated with a "make", the function only gets re-ran when the
    # first function does.

    # The function "test" always returns the same array. Thus test2
    # should not be re-ran in the following.
    output_accumulator = list()
    a, b =  test2(test(3))
    _assert_equal(output_accumulator, [1, 1])

    a, b =  test2(test(3))
    _assert_equal(output_accumulator, [1, 1])

    x = test(10)
    _assert_equal(output_accumulator, [1, 1, 1, ])
    a, b =  test2(x)
    _assert_equal(output_accumulator, [1, 1, 1,])
    #x = x.copy()
    x = 2#np.array([20])
    a, b =  test2(x)
    _assert_equal(output_accumulator, [1, 1, 1, 1])
    a, b =  test2(x)
    _assert_equal(output_accumulator, [1, 1, 1, 1])


    @make(cachedir=cachedir, output=NumpyFile(cachedir+'/test3.npy' ), 
                                     debug=True)
    def test3(x):
        output_accumulator.append(1)
        return np.array(x)

    output_accumulator = list()
    x = test3(1)
    _assert_equal(output_accumulator, [1, ])
    y = test2(x)
    _assert_equal(output_accumulator, [1, 1])
    y = test2(x)
    _assert_equal(output_accumulator, [1, 1])

    x = test3(2)
    _assert_equal(output_accumulator, [1, 1, 1])
    y = test2(x)
    _assert_equal(output_accumulator, [1, 1, 1, 1])

    shutil.rmtree(cachedir)


def test_time_stamp_tracking():
    """ Check that we can keep trac of timestamps even in complex
        signatures.
    """
    cachedir = mkdtemp()
    output_accumulator = list()

    # FIXME: We should check for leaks
    #_time_stamp_registry.__reset__()

    @make(cachedir=cachedir, output=PickleFile(cachedir+'/test3.pkl'),
          debug=True)
    def test3(x):
        output_accumulator.append(1)
        # We are returning a PickleFile instance, as it is a non-native
        # type that can be pickled.
        return 0, {1: PickleFile('test')}


    @make(cachedir=cachedir, output=PickleFile(cachedir+'/test4.pkl'),
          debug=True)
    def test4(x):
        output_accumulator.append(1)
        return x

    d = test3(3)
    _assert_equal(output_accumulator, [1, ])
    e = test3(3)
    _assert_equal(output_accumulator, [1, ])
    _assert_equal(d[1][1], e[1][1])
    # d[1][1] is a non-native type, we should be keeping track of it.
    assert d[1][1] in _time_stamp_registry
    # Check that we are not refreshing the timestamp when we shouldn't
    # be:
    d = test3(3)
    time_stamp = _time_stamp_registry[d[1][1]]
    _assert_equal(time_stamp, _time_stamp_registry[d[1][1]])
    f = test4(e[1])
    _assert_equal(output_accumulator, [1, 1])
    _assert_equal(d[1][1], f[1])

    # Extra check, probably redundent.
    g = test3(3)
    h = test4(g[1])
    _assert_equal(output_accumulator, [1, 1])
    _assert_equal(h[1], f[1])

    # FIXME: We should check for leaks
    #_time_stamp_registry.__reset__()
    g = test3(3)
    assert g[1][1] in _time_stamp_registry

    shutil.rmtree(cachedir)


def test_lambda():
    """ Test with two different lambda expressions. If the code is not
        able to separate them, it will run them too often.
    """
    cachedir = mkdtemp()

    side_effect = [0]
    f = make(func=lambda x: side_effect[0] + 1, cachedir=cachedir, name='f',
                     output=PickleFile(cachedir+'/f_output'), debug=True)
    g = make(func=lambda x: side_effect[0] + 2, cachedir=cachedir, name='g',
                     output=PickleFile(cachedir+'/g_output'), debug=True)
    _assert_equal(f(2), 1)
    _assert_equal(g(2), 2)
    # If the function is rerun, the side_effect will matter, and the
    # following assert will fail.
    side_effect[0] = 1
    _assert_equal(f(2), 1)

    shutil.rmtree(cachedir)


