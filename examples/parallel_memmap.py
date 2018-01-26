"""
===============================
NumPy memmap in joblib.Parallel
===============================

This example illustrates some features enabled by using :class:`numpy.memmap`
within :class:`joblib.Parallel`. First, we show that dumping a huge data array
ahead of passing it to :class:`joblib.Parallel` speed-up computation. Then, we
show the possibility to provide write access to original data.

"""

##############################################################################
# Speed-up processing with large data array
##############################################################################
# 
# We create a large data array from which an average is computed for several
# slices.

import numpy as np

data = np.random.random((int(1e7),))
window_size = int(5e5)
slices = [slice(start, start + window_size)
          for start in range(0, data.size - window_size, int(1e5))]

###############################################################################
# The ``slow_mean`` function introduce a :func:`time.sleep` call to simulate a
# more expensive computation cost for which parallel computing is beneficial.
# Otherwise, for very fast sequential processing, parallel processing is not
# adapted due some extra overheads (workers creations, communication, etc.).

import time


def slow_mean(data, sl):
    """Simulate a larger processing"""
    time.sleep(0.005)
    return data[sl].mean()


###############################################################################
# First, we will evaluate the sequential computing on our problem.

tic = time.time()
results = [slow_mean(data, sl) for sl in slices]
toc = time.time()
print('\nElapsed time computing the average on couple of slices {:.2f} s'
      .format(toc - tic))

###############################################################################
# :class:`joblib.Parallel` is used to compute in parallel the average on each
# slices using 2 workers.

from joblib import Parallel, delayed


tic = time.time()
results = Parallel(n_jobs=2)(delayed(slow_mean)(data, sl) for sl in slices)
toc = time.time()
print('\nElapsed time computing the average on couple of slices {:.2f} s'
      .format(toc - tic))

###############################################################################
# Surprisingly (sic) the parallel processing is slower than the sequential
# processing. Indeed, ``data`` is hashed at each call of ``slow_mean``, leading
# to an important time overhead. A solution is to dump this array to a memmap
# and pass the memmap to :class:`joblib.Parallel`.

from joblib import dump, load

filename_memmap = 'data.pkl'
dump(data, filename_memmap)
data = load(filename_memmap, mmap_mode='r')

tic = time.time()
results = Parallel(n_jobs=2)(delayed(slow_mean)(data, sl) for sl in slices)
toc = time.time()
print('\nElapsed time computing the average on couple of slices {:.2f} s\n'
      .format(toc - tic))

###############################################################################
# Therefore, dumping large ``data`` array ahead of calling
# :class:`joblib.Parallel` can speed-up significantly the processing by
# removing some overheads.

###############################################################################
# Output write access within :class:`joblib.Parallel`
###############################################################################
# 
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
try:
    os.mkdir(folder)
except FileExistsError:
    pass
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

Parallel(n_jobs=2)(delayed(sum_row)(samples, sums, i)
                   for i in range(samples.shape[0]))

###############################################################################
# Compare the results from the output buffer with the ground truth

print("\nExpected sums computed in the parent process:\n {}"
      .format(samples.sum(axis=1)))
print("\nActual sums computed by the worker processes:\n {}"
      .format(sums))

###############################################################################
# Clean-up the memmap
###############################################################################

import shutil
import os

try:
    os.remove(filename_memmap)
    shutil.rmtree(folder)
except:  # noqa
    print('Could not clean-up automatically.')
