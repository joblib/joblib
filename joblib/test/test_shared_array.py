import gc
from cPickle import loads
from cPickle import dumps
import tempfile
import os.path
import shutil

from ..shared_array import SharedArray
from ..shared_array import SharedMemmap
from ..shared_array import as_shared_array
from ..shared_array import as_shared_memmap
from ..parallel import Parallel
from ..parallel import delayed
from .common import with_numpy, np

from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_true
from nose.tools import assert_false
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
def test_shared_array():
    a = SharedArray((3, 5), dtype=np.float32, order='F')

    # check array metadata
    assert_equal(a.shape, (3, 5))
    assert_true(a.flags['F_CONTIGUOUS'])
    assert_equal(a.dtype, np.float32)
    assert_equal(a.mode, 'r+')

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
    assert_equal(b.mode, 'r+')
    assert_array_equal(a, b)

    # check recursive pickling
    c = loads(dumps(b))
    assert_true(c is not a)
    assert_true(c is not b)
    assert_equal(c.shape, (3, 5))
    assert_true(c.flags['F_CONTIGUOUS'])
    assert_equal(c.dtype, np.float32)
    assert_equal(c.mode, 'r+')
    assert_array_equal(a, c)
    assert_array_equal(b, c)

    # check memory sharing
    a += 1.0
    assert_array_equal(a, b)
    assert_array_equal(a, c)

    b[:] = 0.0
    assert_array_equal(a, np.zeros((3, 5), dtype=np.float32))
    assert_array_equal(c, np.zeros((3, 5), dtype=np.float32))


def inplace_power(a, i):
    assert_equal(a[i], 2)
    a[i] **= i


@with_numpy
def test_shared_array_parallel():
    a = SharedArray(10, dtype=np.int32)
    a.fill(2)
    assert_array_equal(a, np.ones(10) * 2)

    Parallel(n_jobs=2)(delayed(inplace_power)(a, i)
                       for i in range(a.shape[0]))

    assert_array_equal(a, [2 ** i for i in range(10)])


@with_numpy
def test_shared_array_parallel_on_pickled_shared_array():
    a = SharedArray(10, dtype=np.int32)
    a.fill(2)
    assert_array_equal(a, np.ones(10) * 2)

    # Make a clone of a but don't garbage collect the original
    b = loads(dumps(a))

    # Use b in a parallel setting instead of the initially allocated shared
    # array
    Parallel(n_jobs=2)(delayed(inplace_power)(b, i)
                       for i in range(b.shape[0]))

    assert_array_equal(b, [2 ** i for i in range(10)])


    # Garbage collect a and continue using b
    # XXX: should we try to handle this case? check multiprocessing.Array
    # behavior and try to replicate it or document limitations in docstring
    #del a
    #gc.collect()

    ## Use b in a parallel setting instead of the initially allocated shared
    ## array
    #b.fill(2)
    #Parallel(n_jobs=2)(delayed(inplace_power)(b, i)
    #                   for i in range(b.shape[0]))

    #assert_array_equal(b, [2 ** i for i in range(10)])


@with_numpy
def test_shared_array_parallel_copyonwrite():
    a = SharedArray(10, dtype=np.int32, mode='c')
    a.fill(2)
    assert_array_equal(a, np.ones(10) * 2)

    Parallel(n_jobs=2)(delayed(inplace_power)(a, i)
                       for i in range(a.shape[0]))

    assert_array_equal(a, np.ones(10) * 2)


@with_numpy
def test_shared_array_errors():
    # Check invalid mode
    assert_raises(ValueError, SharedArray, 10, mode='creative')

    # Only readwrite and copyonwrite are supported for anonymous shared arrays
    assert_raises(ValueError, SharedArray, 10, mode='r')
    assert_raises(ValueError, SharedArray, 10, mode='w+')


@with_numpy
@with_temp_folder
def test_file_based_shared_memmap():
    filename = os.path.join(TEMP_FOLDER, 'some_file.bin')
    f = open(filename, 'wb')
    f.write('\0' * 4 * 3 * 4)
    f.close()

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
    f = open(filename, 'rb')
    b = SharedMemmap(f, dtype=np.float32, mode='r', shape=(3, 4), order='F')
    assert_equal(b.shape, (3, 4))
    assert_true(b.flags['F_CONTIGUOUS'])
    assert_equal(b.dtype, np.float32)
    assert_equal(b.mode, 'r')
    assert_equal(b.offset, 0)
    assert_equal(b.filename, filename)
    f.close()

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
    f = open(filename, 'wb')
    f.write('\0' * 4 * 3 * 4)
    f.close()

    # Check inconsistent shape
    f = open(filename, 'rb')
    assert_raises(ValueError, SharedMemmap, f, dtype=np.float32,
                  mode='r', shape=(42,))
    f.close()

    # Check inconsistent length, offset and dtype
    assert_raises(ValueError, SharedMemmap, filename,
                  dtype=np.float64, offset=7)

    # Check invalid mode
    assert_raises(ValueError, SharedMemmap, filename, mode='creative')

    # Check mandatory shape in write mode (this will delete the file content
    # though)
    assert_raises(ValueError, SharedMemmap, filename, mode='w+')


@with_numpy
def test_as_shared_array():
    a = as_shared_array(np.zeros((2, 4)))
    assert_equal(a.shape, (2, 4))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.float64)
    assert_equal(a.mode, 'r+')

    b = as_shared_array([1, 2, 3, 4], shape=(2, 2))
    assert_equal(b.shape, (2, 2))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int_)
    assert_equal(b.mode, 'r+')

    c = as_shared_array(a)
    assert_true(c is a)


@with_numpy
def test_shared_array_operators():
    """Check that operators trigger regular array allocations"""
    a = as_shared_array(np.ones((2, 5)))
    b = a * 2
    assert_array_equal(b, np.ones((2, 5)) * 2)
    # XXX: fix me!
    #assert_false(isinstance(b, SharedArray))


@with_temp_folder
@with_numpy
def test_as_shared_memmap():
    filename = os.path.join(TEMP_FOLDER, 'array.mmap')
    array_mmap = np.memmap(filename, np.int32, shape=(2, 3), mode='w+')
    a = as_shared_memmap(array_mmap)
    assert_equal(a.shape, (2, 3))
    assert_true(a.flags['C_CONTIGUOUS'])
    assert_equal(a.dtype, np.int32)
    assert_equal(a.mode, 'w+')
    assert_equal(a.offset, 0)
    assert_equal(a.filename, filename)

    mmap2 = np.memmap(filename, np.int32, shape=(3,), mode='r+', offset=4 * 3)
    b = as_shared_memmap(mmap2)
    assert_equal(b.shape, (3,))
    assert_true(b.flags['C_CONTIGUOUS'])
    assert_equal(b.dtype, np.int32)
    assert_equal(b.mode, 'r+')
    assert_equal(b.offset, 12)
    assert_equal(b.filename, filename)

    # Change the content of the first array and check that the changes are
    # visible in the others
    array_mmap[1, 0] = 42
    array_mmap.flush()
    assert_equal(a[1, 0], 42)
    assert_equal(mmap2[0], 42)
    assert_equal(b[0], 42)

    # Converse change
    a[1, 0] = 0
    a.flush()
    assert_equal(array_mmap[1, 0], 0)
    assert_equal(mmap2[0], 0)
    assert_equal(b[0], 0)

    c = as_shared_memmap(a)
    assert_true(c is a)
