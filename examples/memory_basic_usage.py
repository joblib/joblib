"""
========================
How to use joblib.Memory
========================

This example illustrates the usage of :class:`joblib.Memory`. We illustrate
usages with a function and a method.

"""

###############################################################################
# Without :class:`joblib.Memory`
###############################################################################
# 
# To show the benefit of using :class:`joblib.Memory`, we will implement a
# function which is computationally expensive to execute. First, we will check
# the time required to perform the decomposition on a large array of data.

import time
import numpy as np


def expensive_computation(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(5)
    return data[column_index]


rng = np.random.RandomState(42)
data = rng.randn(int(1e5), 10)
tic = time.time()
data_trans = expensive_computation(data)
toc = time.time()

print('\nOur function took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# Caching the result of a function avoiding recomputing
###############################################################################
# 
# In the case that we would need to call our function several time with the
# same input data, it is beneficial to avoid recomputing the same results over
# and over since it is expensive. We can use :class:`joblib.Memory` to avoid
# such recomputing. A :class:`joblib.Memory` instance can be created by passing
# a directory in which we want to store the results.

from joblib import Memory
cachedir = './cachedir'
memory = Memory(cachedir=cachedir, verbose=0)


def expensive_computation_cached(data, column_index=0):
    """Simulate an expensive computation"""
    time.sleep(5)
    return data[column_index]


expensive_computation_cached = memory.cache(expensive_computation_cached)
tic = time.time()
data_trans = expensive_computation_cached(data)
toc = time.time()

print('\nOur function took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the first call, the results will be cached. Therefore, the computation
# time correspond to the time to compute the results plus the time to dump the
# results into the disk.

tic = time.time()
data_trans = expensive_computation_cached(data)
toc = time.time()

print('\nOur function took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the second call, the computation time is largely reduced since that the
# results are obtained by loading the data previously dumped on the disk
# instead of recomputing the results.

###############################################################################
# Using :class:`joblib.Memory` with a method
###############################################################################
# 
# :class:`joblib.Memory` is designed to work with pure functions. When you want
# to cache a method within a class, you need to create and cache a pure
# function and use it inside the class.


def _expensive_computation_cached(data, column):
    time.sleep(5)
    return data[column]


class Algorithm(object):
    """A class which is using our pure function."""

    def __init__(self, column=0):
        self.column = column

    def transform(self, data):
        expensive_computation = memory.cache(_expensive_computation_cached)
        return expensive_computation(data, self.column)


transformer = Algorithm()
tic = time.time()
data_trans = transformer.transform(data)
toc = time.time()

print('\nOur function took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################

tic = time.time()
data_trans = transformer.transform(data)
toc = time.time()

print('\nOur function took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# As expected, the second call to the ``transform`` method load the results
# which have been cached.

###############################################################################
# Clean up cache directory
###############################################################################

import shutil
try:
    shutil.rmtree(cachedir)
except:  # noqa
    print('We could not remove the folder containing the cached data.')
