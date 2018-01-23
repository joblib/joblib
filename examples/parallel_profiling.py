"""
===========================================================
Memory consumption and  consideration using joblib.Parallel
===========================================================

This example illustrates some notable differences to keep in mind when using
:class:`joblib.Parallel`. More precisely, we will focus on the memory
consumption.

"""
import time
import gc
from memory_profiler import memory_usage
import numpy as np

import joblib
from joblib import Parallel, delayed

print(__doc__)


def memory_used(func, *args, **kwargs):
    """Compute memory usage when executing func."""
    gc.collect()
    mem_use = memory_usage((func, args, kwargs), interval=.00001,
                           multiprocess=True, include_children=True)
    return max(mem_use) - min(mem_use)


def process_cropping(signal, window_size, slices, n_jobs):
    def moving_average(signal):
        return signal.sum()

    Parallel(n_jobs=n_jobs, max_nbytes=0)(delayed(
        moving_average)(signal[sl]) for sl in slices)


def process_slicing(signal, window_size, slices, n_jobs):
    def moving_average(signal, sl):
        return signal[sl].sum()

    Parallel(n_jobs=n_jobs, max_nbytes=0)(delayed(
        moving_average)(signal, sl) for sl in slices)


signal = np.random.random((10000000,))
joblib.dump(signal, 'tmp.pkl')
signal = joblib.load('tmp.pkl', mmap_mode='r')
window_size = 500000
slices = [slice(start, start + window_size)
          for start in range(0, signal.size - window_size, 50000)]

n_jobs = 1
print('-' * 79)
print('Number of workers: {}'.format(n_jobs))
print('\n')

tic = time.time()
print('Memory used by cropping the signal in the parent function: {:.2f} MiB'
      .format(memory_used(process_cropping, signal, window_size, slices,
                          n_jobs)))
toc = time.time()
print('Elasped processing time: {:.2f} s'.format(toc - tic))

tic = time.time()
print('Memory used by slicing the signal in children functions: {:.2f} MiB'
      .format(memory_used(process_slicing, signal, window_size, slices,
                          n_jobs)))
toc = time.time()
print('Elasped processing time: {:.2f} s'.format(toc - tic))

n_jobs = 4
print('-' * 79)
print('Number of workers: {}'.format(n_jobs))
print('\n')

tic = time.time()
print('Memory used by cropping the signal in the parent function: {:.2f} MiB'
      .format(memory_used(process_cropping, signal, window_size, slices,
                          n_jobs)))
toc = time.time()
print('Elasped processing time: {:.2f} s'.format(toc - tic))

tic = time.time()
print('Memory used by slicing the signal in children functions: {:.2f} MiB'
      .format(memory_used(process_slicing, signal, window_size, slices,
                          n_jobs)))
toc = time.time()
print('Elasped processing time: {:.2f} s'.format(toc - tic))
print('-' * 79)
