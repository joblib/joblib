# TODO: make it possible to skip the tests if multiprocessing is not there
# TODO: make it possible to skip the tests if numpy is not there

import os
import shutil
import tempfile
import numpy as np
from numpy.testing import assert_array_equal

from nose.tools import with_setup
from nose.tools import assert_equal
from ..pool import PicklingPool


TEMPFOLDER = None


def setup_temp_folder():
    global TEMPFOLDER
    TEMPFOLDER = tempfile.mkdtemp(prefix='joblib_test_pool_')


def teardown_temp_folder():
    global TEMPFOLDER
    if TEMPFOLDER is not None:
        shutil.rmtree(TEMPFOLDER)
        TEMPFOLDER = None


with_temp_folder = with_setup(setup_temp_folder, teardown_temp_folder)


def double(input):
    data, position, expected = input
    if expected is not None:
        assert_equal(data[position], expected)
    data[position] *= 2


@with_temp_folder
def test_pool_with_memmap():
    # fork the subprocess before allocating the objects to be passed
    def reduce_memmap(a):
        mode = a.mode
        if mode == 'w+':
            # Do not make the subprocess erase the data from the parent memmap
            # inadvertently
            mode = 'r+'
        order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
        return (np.memmap, (a.filename, a.dtype, mode, a.offset, a.shape,
                            order))
    p = PicklingPool(10, reducers=[(np.memmap, reduce_memmap)])

    mmap1 = os.path.join(TEMPFOLDER, 'mmap1')
    a = np.memmap(mmap1, dtype=np.float32, shape=(3, 5), mode='w+')
    a.fill(1.0)

    p.map(double, [(a, (i, 0), 1.0) for i in range(a.shape[0])])

    assert_array_equal(a[:, 0], 2 * np.ones(a.shape[0]))
