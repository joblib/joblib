"""
==================================================
Checkpoint using joblib.Memory and joblib.Parallel
==================================================

We illustrate how to cache intermediate computing results using
:class:`joblib.Memory` within :class:`joblib.Parallel`.

"""

###############################################################################
# Embed caching within parallel processing
###############################################################################
# 
# It is possible to cache a computationally expensive function executing during
# a parallel process. ``costly_column`` will emulate such function.

import time


def costly_column(data, column):
    """Emulate a costly function by sleeping and returning a column."""
    time.sleep(2)
    return data[column]


def data_processing_mean(data, column):
    """Compute the mean of a column."""
    return costly_column(data, column).mean()


###############################################################################
# Create some data

import numpy as np
data = np.random.randn(int(1e4), 4)

###############################################################################
# It is first possible to make the processing without caching or parallel
# processing.

start = time.time()
results = [data_processing_mean(data, col) for col in range(data.shape[1])]
stop = time.time()

print('\nSequential processing')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# ``costly_column`` is expensive to compute and it is used as an intermediate
# step in ``data_processing_mean``. Therefore, it is interesting to store the
# intermediate results from ``costly_column`` using :class:`joblib.Memory`.

from joblib import Memory

cachedir = './cachedir'
memory = Memory(cachedir=cachedir, verbose=0)
costly_column = memory.cache(costly_column)

###############################################################################
# Then, we execute the same processing in parallel and caching the intermediate
# results.

from joblib import Parallel, delayed

start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_mean)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('\nFirst round - caching the data')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# By using 2 workers, we get a x2 speed-up compare to the sequential case. By
# executing again the same process, the intermediate results obtained by
# calling ``costly_column`` will be loaded from the cache instead of executing
# the function.

start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_mean)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('\nSecond round - reloading from the cache')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# Reuse intermediate checkpoints
###############################################################################
# 
# Having cached the intermediate results of the ``costly_column`` function, we
# are able to easily reuse them by calling the function. We define a new
# processing which will take the maximum of the array returned by
# ``costly_column`` instead of previously the mean.


def data_processing_max(data, column):
    """Compute the max of a column."""
    return costly_column(data, column).max()


start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_max)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('\nReusing intermediate checkpoints')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# The processing time only corresponds to the execution of the ``max``
# function. The internal call to ``costly_column`` is reloading the results
# from the cache.

###############################################################################
# Clean-up the cache folder
###############################################################################

memory.clear()
