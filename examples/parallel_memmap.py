"""
===============================
NumPy memmap in joblib.Parallel
===============================

This example illustrates some features enabled by using a memory map
(:class:`numpy.memmap`) within :class:`joblib.Parallel`. First, we show that
dumping a huge data array ahead of passing it to :class:`joblib.Parallel`
speeds up computation. Then, we show the possibility to provide write access to
original data.

"""

##############################################################################
# Speed up processing of a large data array
##############################################################################
#
# We create a large data array for which the average is computed for several
# slices.

import numpy as np

data = np.random.random((int(1e7),))
window_size = int(5e5)
slices = [slice(start, start + window_size)
          for start in range(0, data.size - window_size, int(1e5))]

###############################################################################
# The ``slow_mean`` function introduces a :func:`time.sleep` call to simulate a
# more expensive computation cost for which parallel computing is beneficial.
# Parallel may not be beneficial for very fast operation, due to extra overhead
# (workers creations, communication, etc.).

import time


def slow_mean(data, sl):
    """Simulate a time consuming processing."""
    time.sleep(0.01)
    return data[sl].mean()


###############################################################################
# First, we will evaluate the sequential computing on our problem.

tic = time.time()
results = [slow_mean(data, sl) for sl in slices]
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s'
      .format(toc - tic))

###############################################################################
# :class:`joblib.Parallel` is used to compute in parallel the average of all
# slices using 2 workers.

from joblib import Parallel, delayed


tic = time.time()
results = Parallel(n_jobs=2)(delayed(slow_mean)(data, sl) for sl in slices)
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s'
      .format(toc - tic))

###############################################################################
# Parallel processing is already faster than the sequential processing. It is
# also possible to remove a bit of overhead by dumping the ``data`` array to a
# memmap and pass the memmap to :class:`joblib.Parallel`.

import os
from joblib import dump, load

folder = './joblib_memmap'
try:
    os.mkdir(folder)
except FileExistsError:
    pass

data_filename_memmap = os.path.join(folder, 'data_memmap')
dump(data, data_filename_memmap)
data = load(data_filename_memmap, mmap_mode='r')

tic = time.time()
results = Parallel(n_jobs=2)(delayed(slow_mean)(data, sl) for sl in slices)
toc = time.time()
print('\nElapsed time computing the average of couple of slices {:.2f} s\n'
      .format(toc - tic))

###############################################################################
# Therefore, dumping large ``data`` array ahead of calling
# :class:`joblib.Parallel` can speed up the processing by removing some
# overhead.

###############################################################################
# Writable memmap for shared memory :class:`joblib.Parallel`
###############################################################################
#
# ``slow_mean_write_output`` will compute the mean for some given slices as in
# the previous example. However, the resulting mean will be directly written on
# the output array.


def slow_mean_write_output(data, sl, output, idx):
    """Simulate a time consuming processing."""
    time.sleep(0.005)
    res_ = data[sl].mean()
    print("[Worker %d] Mean for slice %d is %f" % (os.getpid(), idx, res_))
    output[idx] = res_


###############################################################################
# Prepare the folder where the memmap will be dumped.

output_filename_memmap = os.path.join(folder, 'output_memmap')

###############################################################################
# Pre-allocate a writable shared memory map as a container for the results of
# the parallel computation.

output = np.memmap(output_filename_memmap, dtype=data.dtype,
                   shape=len(slices), mode='w+')

###############################################################################
# ``data`` is replaced by its memory mapped version. Note that the buffer has
# already been dumped in the previous section.

data = load(data_filename_memmap, mmap_mode='r')

###############################################################################
# Fork the worker processes to perform computation concurrently

Parallel(n_jobs=2)(delayed(slow_mean_write_output)(data, sl, output, idx)
                   for idx, sl in enumerate(slices))

###############################################################################
# Compare the results from the output buffer with the expected results

print("\nExpected means computed in the parent process:\n {}"
      .format(np.array(results)))
print("\nActual means computed by the worker processes:\n {}"
      .format(output))

###############################################################################
# Clean-up the memmap
###############################################################################
#
# Remove the different memmap that we created. It might fail in Windows due
# to file permissions.

import shutil

try:
    shutil.rmtree(folder)
except:  # noqa
    print('Could not clean-up automatically.')
