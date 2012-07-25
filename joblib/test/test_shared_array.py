from cPickle import loads
from cPickle import dumps
import tempfile
import os.path
import shutil

from ..shared_array import SharedArray
from ..shared_array import SharedMemmap
from ..shared_array import as_shared_array
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
def test_file_based_shared_memmap():
    filename = os.path.join(TEMP_FOLDER, 'some_file.bin')
    with open(filename, 'wb') as f:
        f.write('\0' * 4 * 3 * 4)

    # Check from existing file with default parameters
    a = SharedMemmap(filename, dtype=np.float32)
    assert_equal(a.shape, (12,))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.float32)
    assert_equal(a.mode, 'r+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, filename)
    assert_array_equal(a, np.zeros(12, dtype=np.float32))

    # Check from filename with integer shape
    a = SharedMemmap(filename, dtype=np.float32, shape=12)
    assert_equal(a.shape, (12,))

    # Check from readonly, open file descriptor
    with open(filename, 'rb') as f:
        b = SharedMemmap(f, dtype=np.float32, mode='r', shape=(3, 4), order='F')
        assert_equal(b.shape, (3, 4))
        assert_true(b.flags['F_CONTIGUOUS'])
        assert_equal(b.dtype, np.float32)
        assert_equal(b.mode, 'r')
        assert_equal(b.offset, 0)
        assert_equal(b.filename, filename)

    # Changing the content of 'a' is reflected in 'b'
    a[0] = 42.
    assert_equal(b[0, 0], 42.)
    a[0] = 0.
    assert_equal(b[0, 0], 0.)

    # Check from filename, no shape, copy on write and fortran aligned
    c = SharedMemmap(filename, dtype=np.float32, mode='copyonwrite',
                    order='F')
    assert_equal(c.shape, (12,))
    assert_true(c.flags['F_CONTIGUOUS'])
    assert_equal(c.dtype, np.float32)
    assert_equal(c.mode, 'c')
    assert_equal(c.offset, 0)
    assert_equal(c.filename, filename)
    assert_array_equal(a, np.zeros(12, dtype=np.float32))

    # Changing the content of c is not reflected in a
    c[0] = 42.
    assert_equal(c[0], 42.)
    assert_equal(a[0], 0.)

    # check pickling
    d = loads(dumps(a))
    assert_true(d is not a)
    assert_equal(d.shape, (12,))
    assert_true(d.flags['C_CONTIGUOUS'])
    assert_equal(d.dtype, np.float32)
    assert_equal(d.mode, 'r+')
    assert_equal(d.offset, 0)
    assert_equal(d.filename, filename)
    assert_array_equal(a, d)

    # check memory sharing
    a += 1.0
    assert_array_equal(a, d)

    d[:] = 0.0
    assert_array_equal(a, np.zeros((12,), dtype=np.float32))


@with_numpy
@with_temp_folder
def test_file_based_shared_memmap_errors():
    filename = os.path.join(TEMP_FOLDER, 'some_file.bin')
    with open(filename, 'wb') as f:
        f.write('\0' * 4 * 3 * 4)

    # Check inconsistent shape
    with open(filename, 'rb') as f:
        assert_raises(ValueError, SharedMemmap, f, dtype=np.float32,
                      mode='r', shape=(42,))

    # Check inconsistent length, offset and dtype
    assert_raises(ValueError, SharedMemmap, filename,
                  dtype=np.float64, offset=7)

    # Check invalid mode
    assert_raises(ValueError, SharedMemmap, filename, mode='creative')

    # Check mandatory shape in write mode (this will delete the file content
    # though)
    assert_raises(ValueError, SharedMemmap, filename, mode='w+')


@with_numpy
def test_as_shared_array_anonymous():
    a = as_shared_array(np.zeros((2, 4)))
    assert_equal(a.shape, (2, 4))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.float64)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, None)

    b = as_shared_array([1, 2, 3, 4], shape=(2, 2))
    assert_equal(b.shape, (2, 2))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int64)
    assert_equal(b.mode, 'w+')
    assert_equal(b.offset, 0)
    assert_equal(b.filename, None)

    c = as_shared_array(a)
    assert_true(c is a)


@with_temp_folder
@with_numpy
def test_as_shared_array_memmap():
    filename = os.path.join(TEMP_FOLDER, 'array.mmap')
    mmap = np.memmap(filename, np.int32, shape=(2, 3), mode='w+')
    a = as_shared_array(mmap)
    assert_equal(a.shape, (2, 3))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.int32)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, filename)

    mmap2 = np.memmap(filename, np.int32, shape=(3,), mode='r+', offset=4 * 3)
    b = as_shared_array(mmap2)
    assert_equal(b.shape, (3,))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int32)
    assert_equal(b.mode, 'r+')
    assert_equal(b.offset, 12)
    assert_equal(b.filename, filename)

    # Change the content of the first array and check that the changes are
    # visible in the others
    mmap[1, 0] = 42
    mmap.flush()
    assert_equal(a[1, 0], 42)
    assert_equal(mmap2[0], 42)
    assert_equal(b[0], 42)

    # Converse change
    a[1, 0] = 0
    a.flush()
    assert_equal(mmap[1, 0], 0)
    assert_equal(mmap2[0], 0)
    assert_equal(b[0], 0)
