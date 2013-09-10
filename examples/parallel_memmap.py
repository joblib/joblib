"""Demonstrate the usage of numpy.memmap with joblib.Parallel

This example shows how to preallocate data in memmap arrays both for input and
output of the parallel worker processes.

Sample output for this program::

    [Worker 93486] Sum for row 0 is -1599.756454
    [Worker 93487] Sum for row 1 is -243.253165
    [Worker 93488] Sum for row 3 is 610.201883
    [Worker 93489] Sum for row 2 is 187.982005
    [Worker 93489] Sum for row 7 is 326.381617
    [Worker 93486] Sum for row 4 is 137.324438
    [Worker 93489] Sum for row 8 is -198.225809
    [Worker 93487] Sum for row 5 is -1062.852066
    [Worker 93488] Sum for row 6 is 1666.334107
    [Worker 93486] Sum for row 9 is -463.711714
    Expected sums computed in the parent process:
    [-1599.75645426  -243.25316471   187.98200458   610.20188337   137.32443803
     -1062.85206633  1666.33410715   326.38161713  -198.22580876  -463.71171369]
    Actual sums computed by the worker processes:
    [-1599.75645426  -243.25316471   187.98200458   610.20188337   137.32443803
     -1062.85206633  1666.33410715   326.38161713  -198.22580876  -463.71171369]

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

if __name__ == "__main__":
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
        except:
            print("Failed to delete: " + folder)
