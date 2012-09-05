# TODO: make it possible to skip the tests if multiprocessing is not there
# TODO: make it possible to skip the tests if numpy is not there

import os
import shutil
import tempfile
try:
    import numpy as np
    from numpy.testing import assert_array_equal
except ImportError:
    np = None

try:
    import multiprocessing
    from ..pool import PicklingPool
    from ..pool import DEFAULT_REDUCERS
    from ..pool import make_array_to_memmap_reducer
except:
    multiprocessing = None

from nose import SkipTest
from nose.tools import with_setup
from nose.tools import assert_equal
from nose.tools import assert_raises


TEMP_FOLDER = None


def check_multiprocessing():
    if multiprocessing is None:
        raise SkipTest('Need multiprocessing to run')

with_multiprocessing = with_setup(check_multiprocessing)


def check_numpy():
    if np is None:
        raise SkipTests('Need numpy to run')


with_numpy = with_setup(check_numpy)


def setup_temp_folder():
    global TEMP_FOLDER
    TEMP_FOLDER = tempfile.mkdtemp(prefix='joblib_test_pool_')


def teardown_temp_folder():
    global TEMP_FOLDER
    if TEMP_FOLDER is not None:
        shutil.rmtree(TEMP_FOLDER)
        TEMP_FOLDER = None


with_temp_folder = with_setup(setup_temp_folder, teardown_temp_folder)


def double(input):
    """Dummy helper function to be executed in subprocesses"""
    data, position, expected = input
    if expected is not None:
        assert_equal(data[position], expected)
    data[position] *= 2
    if expected is not None:
        assert_array_equal(data[position], 2 * expected)


@with_numpy
@with_multiprocessing
@with_temp_folder
def test_pool_with_memmap():
    """Check that subprocess can access and update shared memory"""
    # fork the subprocess before allocating the objects to be passed
    p = PicklingPool(10)

    filename = os.path.join(TEMP_FOLDER, 'test.mmap')
    a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
    a.fill(1.0)

    p.map(double, [(a, (i, j), 1.0)
                   for i in range(a.shape[0])
                   for j in range(a.shape[1])])

    assert_array_equal(a, 2 * np.ones(a.shape))

    # open a copy-on-write view on the previous data
    b = np.memmap(filename, dtype=np.float32, shape=(5, 3), mode='c')

    p.map(double, [(b, (i, j), 2.0)
                   for i in range(b.shape[0])
                   for j in range(b.shape[1])])

    # the original data is untouched
    assert_array_equal(a, 2 * np.ones(a.shape))
    assert_array_equal(b, 2 * np.ones(b.shape))


    # readonly maps can be read but not updated
    c = np.memmap(filename, dtype=np.float32, shape=(10,), mode='r',
                  offset=5 * 4)

    assert_raises(AssertionError, p.map, double,
                  [(c, i, 3.0) for i in range(c.shape[0])])

    assert_raises(RuntimeError, p.map, double,
                  [(c, i, 2.0) for i in range(c.shape[0])])


@with_numpy
@with_multiprocessing
@with_temp_folder
def test_pool_with_large_arrays():
    """Check that large arrays are not copied in memory"""
    # Check that the tempfolder is empty
    assert_equal(os.listdir(TEMP_FOLDER), [])

    # Build an array reducers that automaticaly dump large array content
    # to filesystem backed memmap instances to avoid memory explosion
    array_reducer = make_array_to_memmap_reducer(40, TEMP_FOLDER)
    reducers = DEFAULT_REDUCERS + [(np.ndarray, array_reducer)]
    p = PicklingPool(10, reducers=reducers)

    small = np.ones(5, dtype=np.float32)
    assert_equal(small.nbytes, 20)
    p.map(double, [(small, i, 1.0) for i in range(small.shape[0])])

    # memory has been copied, the filesystem folder is unused
    assert_equal(os.listdir(TEMP_FOLDER), [])

    # try with a file larger than the memmap threshold of 40 bytes
    large = np.ones(20, dtype=np.float64)
    assert_equal(large.nbytes, 160)
    p.map(double, [(large, i, 1.0) for i in range(large.shape[0])])
    dumped_filenames = os.listdir(TEMP_FOLDER)
    assert_equal(len(dumped_filenames), 2)
