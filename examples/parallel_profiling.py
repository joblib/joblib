"""
================================================
Memory and CPU consumption using joblib.Parallel
================================================

This example illustrates some important aspects to keep in mind when using
:class:`joblib.Parallel`. We will focus on different patterns resulting in the
same outcomes but which trigger different internal behaviors. Those differences
will impact the memory consumption and computation time.

To illustrate those differences, the moving average from a large array will be
computed. Therefore, the average can be computed in parallel at different
locations of the data array.

"""

###############################################################################
# ``time_memory_profiling`` and ``print_info`` are two helper functions being
# used extensively later on. The former function allows to track the memory
# consumption and computation time during the call of a function while the
# latter outputs profiling information. To have a more accurate estimate of
# those statistics, the given function is executed several times.

import gc
import time
from memory_profiler import memory_usage


def time_memory_profiling(func, *args, **kwargs):
    """Helper function to profile computation time and memory usage.

    ``func`` is executed for ``n_iter`` and the average memory consumption and
    elapsed time are returned.

    Parameters
    ----------
    func : callable
        The function to profile.

    args : args
        The arguments used in ``func``.

    kwargs : kwargs
        The keywords arguments used in ``func``.

    Returns
    -------
    memory_used : float
        The average memory used by ``func`` across the iterations in MiB.

    elapsed_time : float
        The average elapsed time to execute ``func`` across the iterations in
        seconds.

    """
    gc.collect()
    n_iter = 3
    tic = time.time()
    mem_use = []
    for _ in range(n_iter):
        gc.collect()
        mem_use.append(
            memory_usage((func, args, kwargs), interval=.001,
                         multiprocess=True, include_children=True))
    toc = time.time()
    return (np.mean([max(mem_use_it) - min(mem_use_it)
                     for mem_use_it in mem_use]),
            (toc - tic) / n_iter)


def print_info(mem_used_avg, elapsed_time, func_name):
    print('Memory used by {}: {:.2f} MiB'.format(func_name, mem_used_avg))
    print('Elasped processing time: {:.2f} s\n'.format(elapsed_time))


###############################################################################
# As explained in the introduction, we will create a large array
# (i.e. ``data``) and take its average at different locations (i.e. ``slices``)
# using a specific window size (i.e. ``window_size``). To speed-up the
# execution, we under-sample the number of slices to evaluate.

import numpy as np

data = np.random.random((int(1e7),))
window_size = int(5e5)
slices = [slice(start, start + window_size)
          for start in range(0, data.size - window_size, int(5e4))]


###############################################################################
# In the first strategy, the function passed to :class:`joblib.Parallel` will
# take the portion of the data for which we need to compute the
# sum. Therefore, the function passed to parallel only computes the sum of the
# array passed as argument. Therefore, all arrays passed to the
# ``average`` function will always be different and it will not be able
# to share memory between the workers.

from joblib import Parallel, delayed


def average_slicing_parent(data, window_size, slices):
    def average(data):
        return data.mean()

    Parallel(n_jobs=2, max_nbytes=0)(delayed(average)(data[sl])
                                     for sl in slices)


mem_used, elapsed_time = time_memory_profiling(average_slicing_parent, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time, average_slicing_parent.__name__)


###############################################################################
# The second strategy differs from the former since the original large array
# will be passed to the ``average`` function and the slicing will occur
# directly in child processes. Therefore, the original array can be shared
# between the different workers and will lead to a more efficient use of the
# memory.


def average_slicing_children(data, window_size, slices):
    def average(data, sl):
        return data[sl].mean()

    Parallel(n_jobs=2, max_nbytes=0)(delayed(average)(data, sl)
                                     for sl in slices)


mem_used, elapsed_time = time_memory_profiling(average_slicing_children, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time, average_slicing_children.__name__)

###############################################################################
# As a conclusion, the amount of memory used is reduced by passing the large
# array and sharing it between the workers, even if the coding pattern seems
# less intuitive. However, when looking at the execution time, this strategy is
# more time consuming. This due to the fact that internally, a hash of the
# shareable array will be computed to ensure that the data can be
# shared. However, computing a hash on a large amount of data is costly. The
# solution to avoid such checking is to manually memmap the input data and give
# it directly to :class:`joblib.Parallel`.

import os
import joblib

filename_memmap = 'data.pkl'
joblib.dump(data, filename_memmap)
data = joblib.load(filename_memmap, mmap_mode='r')

mem_used, elapsed_time = time_memory_profiling(average_slicing_parent, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time, average_slicing_parent.__name__)

###############################################################################

mem_used, elapsed_time = time_memory_profiling(average_slicing_children, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time, average_slicing_children.__name__)

os.remove(filename_memmap)

###############################################################################
# The trend regarding the memory consumption is similar to the previous
# case. However, the computation time is largely reduced due to the usage of
# the memmap. In addition, the memory usage is decreasing by dumping the data
# array into a memmap before to pass it to :class:`Parallel`, avoiding to make
# an unnecessary copy of the array.
