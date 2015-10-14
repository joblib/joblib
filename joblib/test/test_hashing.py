"""
Test the hashing module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose
import time
import hashlib
import tempfile
import os
import sys
import gc
import io
import collections
import pickle
import random

from nose.tools import assert_equal

from joblib.hashing import hash, PY3
from joblib.func_inspect import filter_args
from joblib.memory import Memory

from joblib.test.test_memory import env as test_memory_env
from joblib.test.test_memory import setup_module as test_memory_setup_func
from joblib.test.test_memory import teardown_module as test_memory_teardown_func

from joblib.test.common import np, with_numpy

try:
    # Python 2/Python 3 compat
    unicode('str')
except NameError:
    unicode = lambda s: s


def assert_less(a, b):
    if a > b:
        raise AssertionError("%r is not lower than %r")


###############################################################################
# Helper functions for the tests
def time_func(func, *args):
    """ Time function func on *args.
    """
    times = list()
    for _ in range(3):
        t1 = time.time()
        func(*args)
        times.append(time.time() - t1)
    return min(times)


def relative_time(func1, func2, *args):
    """ Return the relative time between func1 and func2 applied on
        *args.
    """
    time_func1 = time_func(func1, *args)
    time_func2 = time_func(func2, *args)
    relative_diff = 0.5 * (abs(time_func1 - time_func2)
                           / (time_func1 + time_func2))
    return relative_diff


class Klass(object):

    def f(self, x):
        return x


class KlassWithCachedMethod(object):

    def __init__(self):
        mem = Memory(cachedir=test_memory_env['dir'])
        self.f = mem.cache(self.f)

    def f(self, x):
        return x


###############################################################################
# Tests

def test_trival_hash():
    """ Smoke test hash on various types.
    """
    obj_list = [1, 2, 1., 2., 1 + 1j, 2. + 1j,
                'a', 'b',
                (1, ), (1, 1, ), [1, ], [1, 1, ],
                {1: 1}, {1: 2}, {2: 1},
                None,
                gc.collect,
                [1, ].append,
               ]
    for obj1 in obj_list:
        for obj2 in obj_list:
            # Check that 2 objects have the same hash only if they are
            # the same.
            yield nose.tools.assert_equal, hash(obj1) == hash(obj2), \
                obj1 is obj2


def test_hash_methods():
    # Check that hashing instance methods works
    a = io.StringIO(unicode('a'))
    nose.tools.assert_equal(hash(a.flush), hash(a.flush))
    a1 = collections.deque(range(10))
    a2 = collections.deque(range(9))
    nose.tools.assert_not_equal(hash(a1.extend), hash(a2.extend))


@with_numpy
def test_hash_numpy():
    """ Test hashing with numpy arrays.
    """
    rnd = np.random.RandomState(0)
    arr1 = rnd.random_sample((10, 10))
    arr2 = arr1.copy()
    arr3 = arr2.copy()
    arr3[0] += 1
    obj_list = (arr1, arr2, arr3)
    for obj1 in obj_list:
        for obj2 in obj_list:
            yield nose.tools.assert_equal, hash(obj1) == hash(obj2), \
                np.all(obj1 == obj2)

    d1 = {1: arr1, 2: arr1}
    d2 = {1: arr2, 2: arr2}
    yield nose.tools.assert_equal, hash(d1), hash(d2)

    d3 = {1: arr2, 2: arr3}
    yield nose.tools.assert_not_equal, hash(d1), hash(d3)

    yield nose.tools.assert_not_equal, hash(arr1), hash(arr1.T)


@with_numpy
def test_numpy_datetime_array():
    # memoryview is not supported for some dtypes e.g. datetime64
    # see https://github.com/joblib/joblib/issues/188 for more details
    dtypes = ['datetime64[s]', 'timedelta64[D]']

    a_hash = hash(np.arange(10))
    arrays = (np.arange(0, 10, dtype=dtype) for dtype in dtypes)
    for array in arrays:
        nose.tools.assert_not_equal(hash(array), a_hash)


@with_numpy
def test_hash_numpy_noncontiguous():
    a = np.asarray(np.arange(6000).reshape((1000, 2, 3)),
                   order='F')[:, :1, :]
    b = np.ascontiguousarray(a)
    nose.tools.assert_not_equal(hash(a), hash(b))

    c = np.asfortranarray(a)
    nose.tools.assert_not_equal(hash(a), hash(c))


@with_numpy
def test_hash_memmap():
    """ Check that memmap and arrays hash identically if coerce_mmap is
        True.
    """
    filename = tempfile.mktemp(prefix='joblib_test_hash_memmap_')
    try:
        m = np.memmap(filename, shape=(10, 10), mode='w+')
        a = np.asarray(m)
        for coerce_mmap in (False, True):
            yield (nose.tools.assert_equal,
                            hash(a, coerce_mmap=coerce_mmap)
                                == hash(m, coerce_mmap=coerce_mmap),
                            coerce_mmap)
    finally:
        if 'm' in locals():
            del m
            # Force a garbage-collection cycle, to be certain that the
            # object is delete, and we don't run in a problem under
            # Windows with a file handle still open.
            gc.collect()
            try:
                os.unlink(filename)
            except OSError as e:
                # Under windows, some files don't get erased.
                if not os.name == 'nt':
                    raise e


@with_numpy
def test_hash_numpy_performance():
    """ Check the performance of hashing numpy arrays:

        In [22]: a = np.random.random(1000000)

        In [23]: %timeit hashlib.md5(a).hexdigest()
        100 loops, best of 3: 20.7 ms per loop

        In [24]: %timeit hashlib.md5(pickle.dumps(a, protocol=2)).hexdigest()
        1 loops, best of 3: 73.1 ms per loop

        In [25]: %timeit hashlib.md5(cPickle.dumps(a, protocol=2)).hexdigest()
        10 loops, best of 3: 53.9 ms per loop

        In [26]: %timeit hash(a)
        100 loops, best of 3: 20.8 ms per loop
    """
    # This test is not stable under windows for some reason, skip it.
    if sys.platform == 'win32':
        raise nose.SkipTest()

    rnd = np.random.RandomState(0)
    a = rnd.random_sample(1000000)
    if hasattr(np, 'getbuffer'):
        # Under python 3, there is no getbuffer
        getbuffer = np.getbuffer
    else:
        getbuffer = memoryview
    md5_hash = lambda x: hashlib.md5(getbuffer(x)).hexdigest()

    relative_diff = relative_time(md5_hash, hash, a)
    assert_less(relative_diff, 0.3)

    # Check that hashing an tuple of 3 arrays takes approximately
    # 3 times as much as hashing one array
    time_hashlib = 3 * time_func(md5_hash, a)
    time_hash = time_func(hash, (a, a, a))
    relative_diff = 0.5 * (abs(time_hash - time_hashlib)
                           / (time_hash + time_hashlib))
    assert_less(relative_diff, 0.3)


def test_bound_methods_hash():
    """ Make sure that calling the same method on two different instances
    of the same class does resolve to the same hashes.
    """
    a = Klass()
    b = Klass()
    nose.tools.assert_equal(hash(filter_args(a.f, [], (1, ))),
                            hash(filter_args(b.f, [], (1, ))))


@nose.tools.with_setup(test_memory_setup_func, test_memory_teardown_func)
def test_bound_cached_methods_hash():
    """ Make sure that calling the same _cached_ method on two different
    instances of the same class does resolve to the same hashes.
    """
    a = KlassWithCachedMethod()
    b = KlassWithCachedMethod()
    nose.tools.assert_equal(hash(filter_args(a.f.func, [], (1, ))),
                            hash(filter_args(b.f.func, [], (1, ))))


@with_numpy
def test_hash_object_dtype():
    """ Make sure that ndarrays with dtype `object' hash correctly."""

    a = np.array([np.arange(i) for i in range(6)], dtype=object)
    b = np.array([np.arange(i) for i in range(6)], dtype=object)

    nose.tools.assert_equal(hash(a),
                            hash(b))


@with_numpy
def test_numpy_scalar():
    # Numpy scalars are built from compiled functions, and lead to
    # strange pickling paths explored, that can give hash collisions
    a = np.float64(2.0)
    b = np.float64(3.0)
    nose.tools.assert_not_equal(hash(a), hash(b))


@nose.tools.with_setup(test_memory_setup_func, test_memory_teardown_func)
def test_dict_hash():
    # Check that dictionaries hash consistently, eventhough the ordering
    # of the keys is not garanteed
    k = KlassWithCachedMethod()

    d = {'#s12069__c_maps.nii.gz': [33],
         '#s12158__c_maps.nii.gz': [33],
         '#s12258__c_maps.nii.gz': [33],
         '#s12277__c_maps.nii.gz': [33],
         '#s12300__c_maps.nii.gz': [33],
         '#s12401__c_maps.nii.gz': [33],
         '#s12430__c_maps.nii.gz': [33],
         '#s13817__c_maps.nii.gz': [33],
         '#s13903__c_maps.nii.gz': [33],
         '#s13916__c_maps.nii.gz': [33],
         '#s13981__c_maps.nii.gz': [33],
         '#s13982__c_maps.nii.gz': [33],
         '#s13983__c_maps.nii.gz': [33]}

    a = k.f(d)
    b = k.f(a)

    nose.tools.assert_equal(hash(a),
                            hash(b))


@nose.tools.with_setup(test_memory_setup_func, test_memory_teardown_func)
def test_set_hash():
    # Check that sets hash consistently, eventhough their ordering
    # is not garanteed
    k = KlassWithCachedMethod()

    s = set(['#s12069__c_maps.nii.gz',
             '#s12158__c_maps.nii.gz',
             '#s12258__c_maps.nii.gz',
             '#s12277__c_maps.nii.gz',
             '#s12300__c_maps.nii.gz',
             '#s12401__c_maps.nii.gz',
             '#s12430__c_maps.nii.gz',
             '#s13817__c_maps.nii.gz',
             '#s13903__c_maps.nii.gz',
             '#s13916__c_maps.nii.gz',
             '#s13981__c_maps.nii.gz',
             '#s13982__c_maps.nii.gz',
             '#s13983__c_maps.nii.gz'])

    a = k.f(s)
    b = k.f(a)

    nose.tools.assert_equal(hash(a), hash(b))


def test_string():
    # Test that we obtain the same hash for object owning several strings,
    # whatever the past of these strings (which are immutable in Python)
    string = 'foo'
    a = {string: 'bar'}
    b = {string: 'bar'}
    c = pickle.loads(pickle.dumps(b))
    assert_equal(hash([a, b]), hash([a, c]))


@with_numpy
def test_dtype():
    # Test that we obtain the same hash for object owning several dtype,
    # whatever the past of these dtypes. Catter for cache invalidation with
    # complex dtype
    a = np.dtype([('f1', np.uint), ('f2', np.int32)])
    b = a
    c = pickle.loads(pickle.dumps(a))
    assert_equal(hash([a, c]), hash([a, b]))


class MyClass(object):
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.c = 'fixed'


def test_hashes_stay_the_same():
    # We want to make sure that hashes don't change with joblib
    # version. For end users, that would mean that they have to
    # regenerate their cache from scratch, which potentially means
    # lengthy recomputations.
    rng = random.Random(42)
    to_hash_list = ['This is a string to hash',
                    u"C'est l\xe9t\xe9",
                    (123456, 54321, -98765),
                    [rng.random() for _ in range(5)],
                    [3, 'abc', None, MyClass(1, 2)],
                    {'abcde': 123, 'sadfas': [-9999, 2, 3]}]

    # These expected results have been generated with joblib 0.9.2
    expected_dict = {
        'py2': ['80436ada343b0d79a99bfd8883a96e45',
                '2ff3a25200eb6219f468de2640913c2d',
                '50d81c80af05061ac4dcdc2d5edee6d6',
                '536af09b66a087ed18b515acc17dc7fc',
                'b5547baee3f205fb763e8a97c130c054',
                'fc9314a39ff75b829498380850447047'],
        'py3': ['71b3f47df22cb19431d85d92d0b230b2',
                '2d8d189e9b2b0b2e384d93c868c0e576',
                'e205227dd82250871fa25aa0ec690aa3',
                '9e4e9bf9b91890c9734a6111a35e6633',
                '731fafc4405a6c192c0a85a58c9e7a93',
                'aeda150553d4bb5c69f0e69d51b0e2ef']}

    py_version_str = 'py3' if PY3 else 'py2'
    expected_list = expected_dict[py_version_str]

    for to_hash, expected in zip(to_hash_list, expected_list):
        yield assert_equal, hash(to_hash), expected


@with_numpy
def test_hashes_stay_the_same_with_numpy_objects():
    # We want to make sure that hashes don't change with joblib
    # version. For end users, that would mean that they have to
    # regenerate their cache from scratch, which potentially means
    # lengthy recomputations.
    rng = np.random.RandomState(42)
    # Being explicit about dtypes in order to avoid
    # architecture-related differences
    to_hash_list = [
        rng.randint(-1000, high=1000, size=50).astype('<i8'),
        tuple(rng.randn(3).astype('<f8') for _ in range(5)),
        [rng.randn(3).astype('<f8') for _ in range(5)],
        [3, 'abc', None, MyClass(1, 2)],
        {
            -3333: rng.randn(3, 5).astype('<f8'),
            0: [
                rng.randint(10, size=20).astype('<i8'),
                rng.randn(10).astype('<f8')
            ]
        }
    ]

    # These expected results have been generated with joblib 0.9.0
    expected_dict = {'py2': ['80f2387e7752abbda2658aafed49e086',
                             '6028954a4c71750d74e66414bbab227b',
                             'd32171e5bcf0e60034a7923bfea59e08',
                             'b5547baee3f205fb763e8a97c130c054',
                             '7989a9bbb5b7334efeda618843ab3ecc'],
                     'py3': ['10a6afc379ca2708acfbaef0ab676eab',
                             'd93ceeb29ba96f03dfa3f846aa05f221',
                             '54786bcda3d81ef61949d7a103168cee',
                             '731fafc4405a6c192c0a85a58c9e7a93',
                             '245de781f3b1948d4af0331d37a1e47f']}

    py_version_str = 'py3' if PY3 else 'py2'
    expected_list = expected_dict[py_version_str]

    for to_hash, expected in zip(to_hash_list, expected_list):
        yield assert_equal, hash(to_hash), expected
