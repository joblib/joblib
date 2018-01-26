"""
=================================
Checkpoint within joblib.Parallel
=================================

We illustrate how to cache intermediate computing results using
:class:`joblib.Memory` within :class:`joblib.Parallel`.

"""

###############################################################################
# Embed caching within parallel processing
###############################################################################
#
# It is possible to store intermediate results of costly function

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

start = time.time()
results = [data_processing_mean(data, col) for col in range(data.shape[1])]
stop = time.time()

print('\nSequential processing')
print('\nElapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# ``costly_column`` is expensive to compute and it is used as an intermediate step in ``data_processing_mean``. Therefore, it is interesting to store the intermediate results from ``costly_column`` using :class:`joblib.Memory`.

from joblib import Memory

cachedir = './cachedir'
memory = Memory(cachedir=cachedir, verbose=0)
costly_column = memory.cache(costly_column)

from joblib import Parallel, delayed

start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_mean)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('First round')
print('\nElapsed time for the entire processing: {:.2f} s'
      .format(stop - start))


start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_mean)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('Second round')
print('\nElapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# Reuse intermediate checkpoints
###############################################################################
#
# Having cached the intermediate results of the function ``costly_column``, we
# can easily reuse them by calling the function. We define a new processing
# which will take the maximum of the array returned by ``costly_column``
# instead of the mean.


def data_processing_max(data, column):
    """Compute the max of a column."""
    return costly_column(data, column).max()


start = time.time()
results = Parallel(n_jobs=2)(delayed(
    data_processing_max)(data, col) for col in range(data.shape[1]))
stop = time.time()

print('Reusing intermediate checkoints')
print('\nElapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# We can see that the processing time only corresponds to the computation of
# the max since that the call to ``costly_column`` directly loaded the
# intermediate results from the cache.

###############################################################################
# Clean-up the cache folder
###############################################################################

memory.clear()
