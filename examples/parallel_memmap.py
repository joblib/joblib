"""
==========================================
Usage of numpy.memmap with joblib.Parallel
==========================================

This example shows how to preallocate data in memmap arrays both for input and
output of the parallel worker processes.

"""
import tempfile
import shutil
import os
import numpy as np

from joblib import Parallel, delayed
from joblib import load, dump


def sum_row(input, output, i):
    """Compute the sum of a row in input and store it in output"""
    sum_ = input[i, :].sum()
    print("[Worker %d] Sum for row %d is %f" % (os.getpid(), i, sum_))
    output[i] = sum_


rng = np.random.RandomState(42)
folder = tempfile.mkdtemp()
samples_name = os.path.join(folder, 'samples')
sums_name = os.path.join(folder, 'sums')
try:
    # Generate some data and an allocate an output buffer
    samples = rng.normal(size=(10, int(1e6)))

    # Pre-allocate a writeable shared memory map as a container for the
    # results of the parallel computation
    sums = np.memmap(sums_name, dtype=samples.dtype,
                     shape=samples.shape[0], mode='w+')

    # Dump the input data to disk to free the memory
    dump(samples, samples_name)

    # Release the reference on the original in memory array and replace it
    # by a reference to the memmap array so that the garbage collector can
    # release the memory before forking. gc.collect() is internally called
    # in Parallel just before forking.
    samples = load(samples_name, mmap_mode='r')

    # Fork the worker processes to perform computation concurrently
    Parallel(n_jobs=4)(delayed(sum_row)(samples, sums, i)
                       for i in range(samples.shape[0]))

    # Compare the results from the output buffer with the ground truth
    print("Expected sums computed in the parent process:")
    expected_result = samples.sum(axis=1)
    print(expected_result)

    print("Actual sums computed by the worker processes:")
    print(sums)

    assert np.allclose(expected_result, sums)
finally:
    try:
        shutil.rmtree(folder)
    except:  # noqa
        print("Failed to delete: " + folder)
