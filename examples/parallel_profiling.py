"""================================================
Memory and CPU consumption using joblib.Parallel
================================================

This example illustrates some important aspects to keep in mind when using
:class:`joblib.Parallel`. We will focus on different patterns resulting to the
same outcomes but which trigger different internal behaviors. Those differences
will impact the memory consumption and computation time.

To illustrate those differences, the moving average from a large array will be
computed. Therefore, the average can be computed in parallel at different
location of the signal.

"""
from __future__ import division

import os
import time
import gc
import numpy as np
from memory_profiler import memory_usage

import joblib
from joblib import Parallel, delayed

print(__doc__)

###############################################################################
# `memory_used` and `print_info` are two helper functions being used
# extensively later on. The former function allows to track the memory
# consumption during the call of a function while the latter output profiling
# information.


def memory_used(func, *args, **kwargs):
    gc.collect()
    mem_use = memory_usage((func, args, kwargs), interval=.00001,
                           multiprocess=True, include_children=True)
    return max(mem_use) - min(mem_use)


def print_info(mem_used_avg, elapsed_time):
    print('-' * 79)
    print('Memory used by cropping the signal in the parent'
          ' function: {:.2f} MiB'.format(mem_used_avg))
    print('Elasped processing time: {:.2f} s'.format(elapsed_time))
    print('-' * 79)
    print('\n')


###############################################################################
# As explained in the introduction, we will create a large array
# (i.e. `signal`) and take is average at different locations (i.e. `slices`)
# using a specific window size (i.e. `window_size`). To speed-up the execution,
# we under-sample the number of slices to evaluate.


signal = np.random.random((10000000,))
window_size = 500000
slices = [slice(start, start + window_size)
          for start in range(0, signal.size - window_size, 50000)]
n_iter = 10


###############################################################################
# In the first strategy, the function passed to :class:`joblib.Parallel` will
# take the portion of the signal from which we need to compute the
# sum. Therefore, the function passed to parallel only compute the sum of the
# array passed as argument. Therefore, all arrays passed to the
# `moving_average` function will always be different and it will not be able to
# share memory between the workers.


def process_cropping(signal, window_size, slices):
    def moving_average(signal):
        return signal.sum()

    Parallel(n_jobs=4, max_nbytes=0)(delayed(
        moving_average)(signal[sl]) for sl in slices)


tic = time.time()
mem_used = [memory_used(process_cropping, signal, window_size, slices)
            for _ in range(n_iter)]
toc = time.time()
print_info(np.mean(mem_used), (toc - tic) / n_iter)


###############################################################################
# The second strategy differs from the former since the original large array
# will be passed to the `moving_average` function and the slicing will occur
# directly in this function. Therefore, the original array can be shared
# between the different workers and will lead to a more efficient use of the
# memory.


def process_slicing(signal, window_size, slices):
    def moving_average(signal, sl):
        return signal[sl].sum()

    Parallel(n_jobs=4, max_nbytes=0)(delayed(
        moving_average)(signal, sl) for sl in slices)


tic = time.time()
mem_used = [memory_used(process_slicing, signal, window_size, slices)
            for _ in range(n_iter)]
toc = time.time()
print_info(np.mean(mem_used), (toc - tic) / n_iter)

###############################################################################
# As a conclusion, the amount of memory used is reduced by passing the large
# array and sharing it between the workers, even if the coding pattern seems
# less intuitive. However, when looking at the execution time, this strategy is
# more time consuming. This due to the fact that internally, an hash of the
# shareable array will be computed to ensure that the data can be
# shared. However, computing an hash on large amount of data is costly. The
# solution to avoid such checking is to manually memmap the input data and give
# directly it directly to :class:`joblib.Parallel`.

filename_memmap = 'signal.pkl'
joblib.dump(signal, filename_memmap)
signal = joblib.load(filename_memmap, mmap_mode='r')

tic = time.time()
mem_used = [memory_used(process_cropping, signal, window_size, slices)
            for _ in range(n_iter)]
toc = time.time()
print_info(np.mean(mem_used), (toc - tic) / n_iter)


tic = time.time()
mem_used = [memory_used(process_slicing, signal, window_size, slices)
            for _ in range(n_iter)]
toc = time.time()
print_info(np.mean(mem_used), (toc - tic) / n_iter)

os.remove(filename_memmap)

###############################################################################
# The trend regarding the memory consumption is similar to the previous
# case. However, the computation time is largely reduced due to the usage of
# the memmap.
