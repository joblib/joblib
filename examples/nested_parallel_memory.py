"""
==================================================
Checkpoint using joblib.Memory and joblib.Parallel
==================================================

This example illustrates how to cache intermediate computing results using
:class:`joblib.Memory` within :class:`joblib.Parallel`.

"""

###############################################################################
# Embed caching within parallel processing
###############################################################################
#
# It is possible to cache a computationally expensive function executed during
# a parallel process. ``costly_compute`` emulates such time consuming function.

import time


def costly_compute(data, column):
    """Emulate a costly function by sleeping and returning a column."""
    time.sleep(2)
    return data[column]


def data_processing_mean(data, column):
    """Compute the mean of a column."""
    return costly_compute(data, column).mean()


###############################################################################
# Create some data. The random seed is fixed to generate deterministic data
# across Python session. Note that this is not necessary for this specific
# example since the memory cache is cleared at the end of the session.

import numpy as np
rng = np.random.RandomState(42)
data = rng.randn(int(1e4), 4)

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
# ``costly_compute`` is expensive to compute and it is used as an intermediate
# step in ``data_processing_mean``. Therefore, it is interesting to store the
# intermediate results from ``costly_compute`` using :class:`joblib.Memory`.

from joblib import Memory

location = './cachedir'
memory = Memory(location, verbose=0)
costly_compute_cached = memory.cache(costly_compute)


###############################################################################
# Now, we define ``data_processing_mean_using_cache`` which benefits from the
# cache by calling ``costly_compute_cached``

def data_processing_mean_using_cache(data, column):
    """Compute the mean of a column."""
    return costly_compute_cached(data, column).mean()


###############################################################################
# Then, we execute the same processing in parallel and caching the intermediate
# results.

from joblib import Parallel, delayed

start = time.time()
results = Parallel(n_jobs=2)(
    delayed(data_processing_mean_using_cache)(data, col)
    for col in range(data.shape[1]))
stop = time.time()

print('\nFirst round - caching the data')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# By using 2 workers, the parallel processing gives a x2 speed-up compared to
# the sequential case. By executing again the same process, the intermediate
# results obtained by calling ``costly_compute_cached`` will be loaded from the
# cache instead of executing the function.

start = time.time()
results = Parallel(n_jobs=2)(
    delayed(data_processing_mean_using_cache)(data, col)
    for col in range(data.shape[1]))
stop = time.time()

print('\nSecond round - reloading from the cache')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# Reuse intermediate checkpoints
###############################################################################
#
# Having cached the intermediate results of the ``costly_compute_cached``
# function, they are reusable by calling the function. We define a new
# processing which will take the maximum of the array returned by
# ``costly_compute_cached`` instead of previously the mean.


def data_processing_max_using_cache(data, column):
    """Compute the max of a column."""
    return costly_compute_cached(data, column).max()


start = time.time()
results = Parallel(n_jobs=2)(
    delayed(data_processing_max_using_cache)(data, col)
    for col in range(data.shape[1]))
stop = time.time()

print('\nReusing intermediate checkpoints')
print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

###############################################################################
# The processing time only corresponds to the execution of the ``max``
# function. The internal call to ``costly_compute_cached`` is reloading the
# results from the cache.

###############################################################################
# Clean-up the cache folder
###############################################################################

memory.clear(warn=False)
