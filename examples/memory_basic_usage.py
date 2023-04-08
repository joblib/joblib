"""
========================
How to use joblib.Memory
========================

This example illustrates the usage of :class:`joblib.Memory` with both
functions and methods.

"""

###############################################################################
# Without :class:`joblib.Memory`
###############################################################################
#
# ``costly_compute`` emulates a computationally expensive process which later
# will benefit from caching using :class:`joblib.Memory`.

import time
import numpy as np


def costly_compute(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(5)
    return data[column_index]


###############################################################################
# Be sure to set the random seed to generate deterministic data. Indeed, if the
# data is not deterministic, the :class:`joblib.Memory` instance will not be
# able to reuse the cache from one run to another.

rng = np.random.RandomState(42)
data = rng.randn(int(1e5), 10)
start = time.time()
data_trans = costly_compute(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# Caching the result of a function to avoid recomputing
###############################################################################
#
# If we need to call our function several time with the same input data, it is
# beneficial to avoid recomputing the same results over and over since it is
# expensive. :class:`joblib.Memory` enables to cache results from a function
# into a specific location.

from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)


def costly_compute_cached(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(5)
    return data[column_index]


costly_compute_cached = memory.cache(costly_compute_cached)
start = time.time()
data_trans = costly_compute_cached(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the first call, the results will be cached. Therefore, the computation
# time corresponds to the time to compute the results plus the time to dump the
# results into the disk.

start = time.time()
data_trans = costly_compute_cached(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the second call, the computation time is largely reduced since the results
# are obtained by loading the data previously dumped to the disk instead of
# recomputing the results.

###############################################################################
# Using :class:`joblib.Memory` with a method
###############################################################################
#
# :class:`joblib.Memory` is designed to work with functions with no side
# effects. When dealing with class, the computationally expensive part of a
# method has to be moved to a function and decorated in the class method.


def _costly_compute_cached(data, column):
    time.sleep(5)
    return data[column]


class Algorithm(object):
    """A class which is using the previous function."""

    def __init__(self, column=0):
        self.column = column

    def transform(self, data):
        costly_compute = memory.cache(_costly_compute_cached)
        return costly_compute(data, self.column)


transformer = Algorithm()
start = time.time()
data_trans = transformer.transform(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################

start = time.time()
data_trans = transformer.transform(data)
end = time.time()

print('\nThe function took {:.2f} s to compute.'.format(end - start))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# As expected, the second call to the ``transform`` method load the results
# which have been cached.

###############################################################################
# Clean up cache directory
###############################################################################

memory.clear(warn=False)
