"""
==========================================
Usage of numpy.memmap with joblib.Parallel
==========================================

This example shows how to preallocate data in memmap arrays both for input and
output of the parallel worker processes. When using a memmap as output, it is
then possible to write in the same buffer.

"""

###############################################################################
# ``sum_row`` will compute the sum for a row of ``input`` and will write the
# results in ``output``.


def sum_row(input, output, i):
    """Compute the sum of a row in input and store it in output"""
    sum_ = input[i, :].sum()
    print("[Worker %d] Sum for row %d is %f" % (os.getpid(), i, sum_))
    output[i] = sum_


###############################################################################
# We create a large 2D array containing some data.

import numpy as np

rng = np.random.RandomState(42)
samples = rng.normal(size=(10, int(1e6)))

###############################################################################
# Prepare the folder where the memmap will be dumped.

import os

folder = './joblib_memmap'
samples_name = os.path.join(folder, 'samples')
sums_name = os.path.join(folder, 'sums')

###############################################################################
# Pre-allocate a writeable shared memory map as a container for the results of
# the parallel computation and dump the samples buffer.

from joblib import dump

sums = np.memmap(sums_name, dtype=samples.dtype,
                 shape=samples.shape[0], mode='w+')
dump(samples, samples_name)

###############################################################################
# Release the reference on the original in memory array and replace it by a
# reference to the memmap array so that the garbage collector can release the
# memory before forking. gc.collect() is internally called in Parallel just
# before forking.

from joblib import load

samples = load(samples_name, mmap_mode='r')

###############################################################################
# Fork the worker processes to perform computation concurrently

from joblib import Parallel, delayed

Parallel(n_jobs=4)(delayed(sum_row)(samples, sums, i)
                   for i in range(samples.shape[0]))

###############################################################################
# Compare the results from the output buffer with the ground truth

print("Expected sums computed in the parent process:\n {}"
      .format(samples.sum(axis=1)))
print("Actual sums computed by the worker processes:\n {}"
      .format(sums))

###############################################################################

import shutil
try:
    shutil.rmtree(folder)
except:  # noqa
    print("Failed to delete: " + folder)
