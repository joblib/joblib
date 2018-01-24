"""
================================================
Memory and CPU consumption using joblib.Parallel
================================================

This example illustrates some important aspects to keep in mind when using
:class:`joblib.Parallel`. We will focus on different patterns resulting to the
same outcomes but which trigger different internal behaviors. Those differences
will impact the memory consumption and computation time.

To illustrate those differences, the moving average from a large array will be
computed. Therefore, the average can be computed in parallel at different
location of the data array.

"""

###############################################################################
# ``time_memory_profiling`` and ``print_info`` are two helper functions being
# used extensively later on. The former function allows to track the memory
# consumption and computation time during the call of a function while the
# latter output profiling information. To have a more accurate estimate of
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
    mem_use = [memory_usage((func, args, kwargs), interval=.00001,
                            multiprocess=True, include_children=True)
               for _ in range(n_iter)]
    toc = time.time()
    return (np.mean([max(mem_use_it) - min(mem_use_it)
                     for mem_use_it in mem_use]),
            (toc - tic) / n_iter)


def print_info(mem_used_avg, elapsed_time):
    print('Memory used by slicing the data in the parent'
          ' function: {:.2f} MiB'.format(mem_used_avg))
    print('Elasped processing time: {:.2f} s\n'.format(elapsed_time))


###############################################################################
# As explained in the introduction, we will create a large array
# (i.e. ``data``) and take is average at different locations (i.e. ``slices``)
# using a specific window size (i.e. ``window_size``). To speed-up the
# execution, we under-sample the number of slices to evaluate.

import numpy as np

data = np.random.random((10000000,))
window_size = 500000
slices = [slice(start, start + window_size)
          for start in range(0, data.size - window_size, 50000)]


###############################################################################
# In the first strategy, the function passed to :class:`joblib.Parallel` will
# take the portion of the data from which we need to compute the
# sum. Therefore, the function passed to parallel only compute the sum of the
# array passed as argument. Therefore, all arrays passed to the
# ``moving_average`` function will always be different and it will not be able
# to share memory between the workers.

from joblib import Parallel, delayed


def process_slicing_parent(data, window_size, slices):
    def moving_average(data):
        return data.sum()

    Parallel(n_jobs=2, max_nbytes=0)(delayed(
        moving_average)(data[sl]) for sl in slices)


mem_used, elapsed_time = time_memory_profiling(process_slicing_parent, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time)


###############################################################################
# The second strategy differs from the former since the original large array
# will be passed to the ``moving_average`` function and the slicing will occur
# directly in child processes. Therefore, the original array can be shared
# between the different workers and will lead to a more efficient use of the
# memory.


def process_slicing_children(data, window_size, slices):
    def moving_average(data, sl):
        return data[sl].sum()

    Parallel(n_jobs=2, max_nbytes=0)(delayed(
        moving_average)(data, sl) for sl in slices)


mem_used, elapsed_time = time_memory_profiling(process_slicing_children, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time)

###############################################################################
# As a conclusion, the amount of memory used is reduced by passing the large
# array and sharing it between the workers, even if the coding pattern seems
# less intuitive. However, when looking at the execution time, this strategy is
# more time consuming. This due to the fact that internally, a hash of the
# shareable array will be computed to ensure that the data can be
# shared. However, computing an hash on large amount of data is costly. The
# solution to avoid such checking is to manually memmap the input data and give
# directly it to :class:`joblib.Parallel`.

import os
import joblib

filename_memmap = 'data.pkl'
joblib.dump(data, filename_memmap)
data = joblib.load(filename_memmap, mmap_mode='r')

mem_used, elapsed_time = time_memory_profiling(process_slicing_parent, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time)

###############################################################################

mem_used, elapsed_time = time_memory_profiling(process_slicing_children, data,
                                               window_size, slices)
print_info(mem_used, elapsed_time)

os.remove(filename_memmap)

###############################################################################
# The trend regarding the memory consumption is similar to the previous
# case. However, the computation time is largely reduced due to the usage of
# the memmap. In addition, the memory usage is decreasing by dumping the data
# array into a memmap before to pass it to :class:`Parallel`, avoiding to make
# an unnecessary copy of the array.
