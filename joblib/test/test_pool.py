import os
import mmap

from joblib.test.common import with_numpy, np
from joblib.test.common import setup_autokill
from joblib.test.common import teardown_autokill
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises
from joblib.backports import make_memmap

from joblib.pool import MemmapingPool
from joblib.pool import has_shareable_memory
from joblib.pool import ArrayMemmapReducer
from joblib.pool import reduce_memmap
from joblib.pool import _strided_from_memmap
from joblib.pool import _get_backing_memmap


def setup_module():
    setup_autokill(__name__, timeout=300)


def teardown_module():
    teardown_autokill(__name__)


def check_array(args):
    """Dummy helper function to be executed in subprocesses

    Check that the provided array has the expected values in the provided
    range.

    """
    data, position, expected = args
    np.testing.assert_array_equal(data[position], expected)


def inplace_double(args):
    """Dummy helper function to be executed in subprocesses


    Check that the input array has the right values in the provided range
    and perform an inplace modification to double the values in the range by
    two.

    """
    data, position, expected = args
    assert data[position] == expected
    data[position] *= 2
    np.testing.assert_array_equal(data[position], 2 * expected)


@with_numpy
@with_multiprocessing
def test_memmap_based_array_reducing(tmpdir):
    """Check that it is possible to reduce a memmap backed array"""
    assert_array_equal = np.testing.assert_array_equal
    filename = tmpdir.join('test.mmap').strpath

    # Create a file larger than what will be used by a
    buffer = np.memmap(filename, dtype=np.float64, shape=500, mode='w+')

    # Fill the original buffer with negative markers to detect over of
    # underflow in case of test failures
    buffer[:] = - 1.0 * np.arange(buffer.shape[0], dtype=buffer.dtype)
    buffer.flush()

    # Memmap a 2D fortran array on a offseted subsection of the previous
    # buffer
    a = np.memmap(filename, dtype=np.float64, shape=(3, 5, 4),
                  mode='r+', order='F', offset=4)
    a[:] = np.arange(60).reshape(a.shape)

    # Build various views that share the buffer with the original memmap

    # b is an memmap sliced view on an memmap instance
    b = a[1:-1, 2:-1, 2:4]

    # c and d are array views
    c = np.asarray(b)
    d = c.T

    # Array reducer with auto dumping disabled
    reducer = ArrayMemmapReducer(None, tmpdir.strpath, 'c')

    def reconstruct_array(x):
        cons, args = reducer(x)
        return cons(*args)

    def reconstruct_memmap(x):
        cons, args = reduce_memmap(x)
        return cons(*args)

    # Reconstruct original memmap
    a_reconstructed = reconstruct_memmap(a)
    assert has_shareable_memory(a_reconstructed)
    assert isinstance(a_reconstructed, np.memmap)
    assert_array_equal(a_reconstructed, a)

    # Reconstruct strided memmap view
    b_reconstructed = reconstruct_memmap(b)
    assert has_shareable_memory(b_reconstructed)
    assert_array_equal(b_reconstructed, b)

    # Reconstruct arrays views on memmap base
    c_reconstructed = reconstruct_array(c)
    assert not isinstance(c_reconstructed, np.memmap)
    assert has_shareable_memory(c_reconstructed)
    assert_array_equal(c_reconstructed, c)

    d_reconstructed = reconstruct_array(d)
    assert not isinstance(d_reconstructed, np.memmap)
    assert has_shareable_memory(d_reconstructed)
    assert_array_equal(d_reconstructed, d)

    # Test graceful degradation on fake memmap instances with in-memory
    # buffers
    a3 = a * 3
    assert not has_shareable_memory(a3)
    a3_reconstructed = reconstruct_memmap(a3)
    assert not has_shareable_memory(a3_reconstructed)
    assert not isinstance(a3_reconstructed, np.memmap)
    assert_array_equal(a3_reconstructed, a * 3)

    # Test graceful degradation on arrays derived from fake memmap instances
    b3 = np.asarray(a3)
    assert not has_shareable_memory(b3)

    b3_reconstructed = reconstruct_array(b3)
    assert isinstance(b3_reconstructed, np.ndarray)
    assert not has_shareable_memory(b3_reconstructed)
    assert_array_equal(b3_reconstructed, b3)


@with_numpy
@with_multiprocessing
def test_high_dimension_memmap_array_reducing(tmpdir):
    assert_array_equal = np.testing.assert_array_equal

    filename = tmpdir.join('test.mmap').strpath

    # Create a high dimensional memmap
    a = np.memmap(filename, dtype=np.float64, shape=(100, 15, 15, 3),
                  mode='w+')
    a[:] = np.arange(100 * 15 * 15 * 3).reshape(a.shape)

    # Create some slices/indices at various dimensions
    b = a[0:10]
    c = a[:, 5:10]
    d = a[:, :, :, 0]
    e = a[1:3:4]

    def reconstruct_memmap(x):
        cons, args = reduce_memmap(x)
        res = cons(*args)
        return res

    a_reconstructed = reconstruct_memmap(a)
    assert has_shareable_memory(a_reconstructed)
    assert isinstance(a_reconstructed, np.memmap)
    assert_array_equal(a_reconstructed, a)

    b_reconstructed = reconstruct_memmap(b)
    assert has_shareable_memory(b_reconstructed)
    assert_array_equal(b_reconstructed, b)

    c_reconstructed = reconstruct_memmap(c)
    assert has_shareable_memory(c_reconstructed)
    assert_array_equal(c_reconstructed, c)

    d_reconstructed = reconstruct_memmap(d)
    assert has_shareable_memory(d_reconstructed)
    assert_array_equal(d_reconstructed, d)

    e_reconstructed = reconstruct_memmap(e)
    assert has_shareable_memory(e_reconstructed)
    assert_array_equal(e_reconstructed, e)


@with_numpy
@with_multiprocessing
def test_pool_with_memmap(tmpdir):
    """Check that subprocess can access and update shared memory memmap"""
    assert_array_equal = np.testing.assert_array_equal

    # Fork the subprocess before allocating the objects to be passed
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = MemmapingPool(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:
        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)

        p.map(inplace_double, [(a, (i, j), 1.0)
                               for i in range(a.shape[0])
                               for j in range(a.shape[1])])

        assert_array_equal(a, 2 * np.ones(a.shape))

        # Open a copy-on-write view on the previous data
        b = np.memmap(filename, dtype=np.float32, shape=(5, 3), mode='c')

        p.map(inplace_double, [(b, (i, j), 2.0)
                               for i in range(b.shape[0])
                               for j in range(b.shape[1])])

        # Passing memmap instances to the pool should not trigger the creation
        # of new files on the FS
        assert os.listdir(pool_temp_folder) == []

        # the original data is untouched
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(b, 2 * np.ones(b.shape))

        # readonly maps can be read but not updated
        c = np.memmap(filename, dtype=np.float32, shape=(10,), mode='r',
                      offset=5 * 4)

        with raises(AssertionError):
            p.map(check_array, [(c, i, 3.0) for i in range(c.shape[0])])

        # depending on the version of numpy one can either get a RuntimeError
        # or a ValueError
        with raises((RuntimeError, ValueError)):
            p.map(inplace_double, [(c, i, 2.0) for i in range(c.shape[0])])
    finally:
        # Clean all filehandlers held by the pool
        p.terminate()
        del p


@with_numpy
@with_multiprocessing
def test_pool_with_memmap_array_view(tmpdir):
    """Check that subprocess can access and update shared memory array"""
    assert_array_equal = np.testing.assert_array_equal

    # Fork the subprocess before allocating the objects to be passed
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = MemmapingPool(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:

        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)

        # Create an ndarray view on the memmap instance
        a_view = np.asarray(a)
        assert not isinstance(a_view, np.memmap)
        assert has_shareable_memory(a_view)

        p.map(inplace_double, [(a_view, (i, j), 1.0)
                               for i in range(a.shape[0])
                               for j in range(a.shape[1])])

        # Both a and the a_view have been updated
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(a_view, 2 * np.ones(a.shape))

        # Passing memmap array view to the pool should not trigger the
        # creation of new files on the FS
        assert os.listdir(pool_temp_folder) == []

    finally:
        p.terminate()
        del p


@with_numpy
@with_multiprocessing
def test_memmaping_pool_for_large_arrays(tmpdir):
    """Check that large arrays are not copied in memory"""

    # Check that the tempfolder is empty
    assert os.listdir(tmpdir.strpath) == []

    # Build an array reducers that automaticaly dump large array content
    # to filesystem backed memmap instances to avoid memory explosion
    p = MemmapingPool(3, max_nbytes=40, temp_folder=tmpdir.strpath)
    try:
        # The temporary folder for the pool is not provisioned in advance
        assert os.listdir(tmpdir.strpath) == []
        assert not os.path.exists(p._temp_folder)

        small = np.ones(5, dtype=np.float32)
        assert small.nbytes == 20
        p.map(check_array, [(small, i, 1.0) for i in range(small.shape[0])])

        # Memory has been copied, the pool filesystem folder is unused
        assert os.listdir(tmpdir.strpath) == []

        # Try with a file larger than the memmap threshold of 40 bytes
        large = np.ones(100, dtype=np.float64)
        assert large.nbytes == 800
        p.map(check_array, [(large, i, 1.0) for i in range(large.shape[0])])

        # The data has been dumped in a temp folder for subprocess to share it
        # without per-child memory copies
        assert os.path.isdir(p._temp_folder)
        dumped_filenames = os.listdir(p._temp_folder)
        assert len(dumped_filenames) == 1

        # Check that memory mapping is not triggered for arrays with
        # dtype='object'
        objects = np.array(['abc'] * 100, dtype='object')
        results = p.map(has_shareable_memory, [objects])
        assert not results[0]

    finally:
        # check FS garbage upon pool termination
        p.terminate()
        assert not os.path.exists(p._temp_folder)
        del p


@with_numpy
@with_multiprocessing
def test_memmaping_pool_for_large_arrays_disabled(tmpdir):
    """Check that large arrays memmaping can be disabled"""
    # Set max_nbytes to None to disable the auto memmaping feature
    p = MemmapingPool(3, max_nbytes=None, temp_folder=tmpdir.strpath)
    try:

        # Check that the tempfolder is empty
        assert os.listdir(tmpdir.strpath) == []

        # Try with a file largish than the memmap threshold of 40 bytes
        large = np.ones(100, dtype=np.float64)
        assert large.nbytes == 800
        p.map(check_array, [(large, i, 1.0) for i in range(large.shape[0])])

        # Check that the tempfolder is still empty
        assert os.listdir(tmpdir.strpath) == []

    finally:
        # Cleanup open file descriptors
        p.terminate()
        del p


@with_numpy
@with_multiprocessing
@with_dev_shm
def test_memmaping_on_dev_shm():
    """Check that MemmapingPool uses /dev/shm when possible"""
    p = MemmapingPool(3, max_nbytes=10)
    try:
        # Check that the pool has correctly detected the presence of the
        # shared memory filesystem.
        pool_temp_folder = p._temp_folder
        folder_prefix = '/dev/shm/joblib_memmaping_pool_'
        assert pool_temp_folder.startswith(folder_prefix)
        assert os.path.exists(pool_temp_folder)

        # Try with a file larger than the memmap threshold of 10 bytes
        a = np.ones(100, dtype=np.float64)
        assert a.nbytes == 800
        p.map(id, [a] * 10)
        # a should have been memmaped to the pool temp folder: the joblib
        # pickling procedure generate one .pkl file:
        assert len(os.listdir(pool_temp_folder)) == 1

        # create a new array with content that is different from 'a' so that
        # it is mapped to a different file in the temporary folder of the
        # pool.
        b = np.ones(100, dtype=np.float64) * 2
        assert b.nbytes == 800
        p.map(id, [b] * 10)
        # A copy of both a and b are now stored in the shared memory folder
        assert len(os.listdir(pool_temp_folder)) == 2

    finally:
        # Cleanup open file descriptors
        p.terminate()
        del p

    # The temp folder is cleaned up upon pool termination
    assert not os.path.exists(pool_temp_folder)


@with_numpy
@with_multiprocessing
def test_memmaping_pool_for_large_arrays_in_return(tmpdir):
    """Check that large arrays are not copied in memory in return"""
    assert_array_equal = np.testing.assert_array_equal

    # Build an array reducers that automaticaly dump large array content
    # but check that the returned datastructure are regular arrays to avoid
    # passing a memmap array pointing to a pool controlled temp folder that
    # might be confusing to the user

    # The MemmapingPool user can always return numpy.memmap object explicitly
    # to avoid memory copy
    p = MemmapingPool(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        res = p.apply_async(np.ones, args=(1000,))
        large = res.get()
        assert not has_shareable_memory(large)
        assert_array_equal(large, np.ones(1000))
    finally:
        p.terminate()
        del p


def _worker_multiply(a, n_times):
    """Multiplication function to be executed by subprocess"""
    assert has_shareable_memory(a)
    return a * n_times


@with_numpy
@with_multiprocessing
def test_workaround_against_bad_memmap_with_copied_buffers(tmpdir):
    """Check that memmaps with a bad buffer are returned as regular arrays

    Unary operations and ufuncs on memmap instances return a new memmap
    instance with an in-memory buffer (probably a numpy bug).
    """
    assert_array_equal = np.testing.assert_array_equal

    p = MemmapingPool(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        # Send a complex, large-ish view on a array that will be converted to
        # a memmap in the worker process
        a = np.asarray(np.arange(6000).reshape((1000, 2, 3)),
                       order='F')[:, :1, :]

        # Call a non-inplace multiply operation on the worker and memmap and
        # send it back to the parent.
        b = p.apply_async(_worker_multiply, args=(a, 3)).get()
        assert not has_shareable_memory(b)
        assert_array_equal(b, 3 * a)
    finally:
        p.terminate()
        del p


@with_numpy
def test__strided_from_memmap(tmpdir):
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    # This line creates the mmap file that is reused later
    memmap_obj = np.memmap(fname, mode='w+', shape=size + offset)
    # filename, dtype, mode, offset, order, shape, strides, total_buffer_len
    memmap_obj = _strided_from_memmap(fname, dtype='uint8', mode='r',
                                      offset=offset, order='C', shape=size,
                                      strides=None, total_buffer_len=None)
    assert isinstance(memmap_obj, np.memmap)
    assert memmap_obj.offset == offset
    memmap_backed_obj = _strided_from_memmap(fname, dtype='uint8', mode='r',
                                             offset=offset, order='C',
                                             shape=(size // 2,), strides=(2,),
                                             total_buffer_len=size)
    assert _get_backing_memmap(memmap_backed_obj).offset == offset


def identity(arg):
    return arg


@with_numpy
@with_multiprocessing
def test_pool_memmap_with_big_offset(tmpdir):
    # Test that numpy memmap offset is set correctly if greater than
    # mmap.ALLOCATIONGRANULARITY, see
    # https://github.com/joblib/joblib/issues/451 and
    # https://github.com/numpy/numpy/pull/8443 for more details.
    fname = tmpdir.join('test.mmap').strpath
    size = 5 * mmap.ALLOCATIONGRANULARITY
    offset = mmap.ALLOCATIONGRANULARITY + 1
    obj = make_memmap(fname, mode='w+', shape=size, dtype='uint8',
                      offset=offset)

    p = MemmapingPool(2, temp_folder=tmpdir.strpath)
    result = p.apply_async(identity, args=(obj,)).get()
    assert isinstance(result, np.memmap)
    assert result.offset == offset
    np.testing.assert_array_equal(obj, result)
