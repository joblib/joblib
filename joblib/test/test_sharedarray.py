from cPickle import loads
from cPickle import dumps
import tempfile
import os.path
import shutil

from ..sharedarray import SharedArray
from ..sharedarray import assharedarray
from .common import with_numpy, np

from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_true
from nose.tools import assert_raises

assert_array_equal = np.testing.assert_array_equal


TEMP_FOLDER = None


def setup_temp_folder():
    global TEMP_FOLDER
    TEMP_FOLDER = tempfile.mkdtemp("joblib_test_sharedarray_")


def teardown_temp_folder():
    if os.path.exists(TEMP_FOLDER):
        shutil.rmtree(TEMP_FOLDER)


with_temp_folder = with_setup(setup_temp_folder, teardown_temp_folder)


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
    assert_equal(b.filename, None)
    assert_array_equal(a, b)

    # check memory sharing
    a += 1.0
    assert_array_equal(a, b)

    b[:] = 0.0
    assert_array_equal(a, np.zeros((3, 5), dtype=np.float32))


@with_numpy
@with_temp_folder
def test_file_based_sharedarray():
    filename = os.path.join(TEMP_FOLDER, 'some_file.bin')
    with open(filename, 'wb') as f:
        f.write('\0' * 4 * 3 * 4)

    # Check from filename
    a = SharedArray(filename=filename, dtype=np.float32)
    assert_equal(a.shape, (12,))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.float32)
    assert_equal(a.mode, 'r+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, filename)
    assert_array_equal(a, np.zeros(12, dtype=np.float32))

    # Check from readonly, open file descriptor
    with open(filename, 'rb') as f:
        b = SharedArray(filename=f, dtype=np.float32, shape=(3, 4))
        assert_equal(b.shape, (3, 4))
        assert_true(b.flags['C_CONTIGUOUS'])
        assert_equal(b.dtype, np.float32)
        assert_equal(b.mode, 'r')
        assert_equal(b.offset, 0)
        assert_equal(b.filename, filename)

    # Check inconsistent shape
    with open(filename, 'rb') as f:
        assert_raises(ValueError, SharedArray, filename=f, dtype=np.float32,
                      shape=(42,))


@with_numpy
def test_assharedarray_anonymous():
    a = assharedarray(np.zeros((2, 4)))
    assert_equal(a.shape, (2, 4))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.float64)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, None)

    b = assharedarray([1, 2, 3, 4], shape=(2, 2))
    assert_equal(b.shape, (2, 2))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int64)
    assert_equal(b.mode, 'w+')
    assert_equal(b.offset, 0)
    assert_equal(b.filename, None)


@with_temp_folder
@with_numpy
def test_assharedarray_memmap():
    filename = os.path.join(TEMP_FOLDER, 'array.mmap')
    mmap = np.memmap(filename, np.int32, shape=(2, 3), mode='w+')
    a = assharedarray(mmap)
    assert_equal(a.shape, (2, 3))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.int32)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, filename)

    mmap2 = np.memmap(filename, np.int32, shape=(3,), mode='r+', offset=4 * 3)
    b = assharedarray(mmap2)
    assert_equal(b.shape, (3,))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int32)
    assert_equal(b.mode, 'r+')
    assert_equal(b.offset, 12)
    assert_equal(b.filename, filename)

    # change the content of the first array and check that the changes are
    # visible in the others
    mmap[1, 0] = 42
    mmap.flush()
    assert_equal(a[1, 0], 42)
    assert_equal(b[0], 42)
