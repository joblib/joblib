from cPickle import loads
from cPickle import dumps

from ..sharedarray import SharedArray
from .common import with_numpy, np

from nose.tools import assert_equal
from nose.tools import assert_true

assert_array_equal = np.testing.assert_array_equal


@with_numpy
def test_anonymous_shared_array():
    a = SharedArray(dtype=np.float32, shape=(3, 5), order='F')

    # check array metadata
    assert_equal(a.shape, (3, 5))
    assert_true(a.flags['F_CONTIGUOUS'])
    assert_equal(a.dtype, np.float32)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)

    # check some basic numpy features
    assert_array_equal(a, np.zeros((3, 5), dtype=np.float32))
    a[:, 2] = 1.0
    assert_array_equal(a[:, 2], np.ones((3,), dtype=np.float32))

    # check pickling
    b = loads(dumps(a))
    assert_true(b is not a)
    assert_equal(b.shape, (3, 5))
    assert_true(b.flags['F_CONTIGUOUS'])
    assert_equal(b.dtype, np.float32)
    assert_equal(b.mode, 'w+')
    assert_equal(b.offset, 0)
    assert_array_equal(a, b)

    # check memory sharing
    a += 1.0
    assert_array_equal(a, b)

    b[:] = 0.0
    assert_array_equal(a, np.zeros((3, 5), dtype=np.float32))


@with_numpy
def test_assharedarray():
    pass
