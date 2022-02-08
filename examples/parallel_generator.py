"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
:class:`joblib.Parallel` to get a generator on the outputs of parallel jobs.
We first create tasks that return results with large memory footprints.
If we call :class:`~joblib.Parallel` for several of these tasks directly, we
observe a high memory usage, as all the results are stacked in RAM.

Using the ``return_generator=True`` option allows to progressively consumes
the outputs as they arrive and keeps the memory at an acceptable level.

Note that the exact RAM usage can depend on the behavior of the garbage
collector, whose behavior is hard to predict.
"""

##############################################################################
# ``MemoryMonitor`` helper
##############################################################################

##############################################################################
# The following class is an helper to monitor the memory of the process and its
# children in another thread, so we can display it afterward.
#
# For this example, we will use ``psutil`` to monitor the memory usage in the
# code. Make sure it is installed with ``pip install psutil`` for this example.
#

import gc
import time
from psutil import Process
from threading import Thread


class MemoryMonitor(Thread):
    """Monitor the memory usage in MB."""
    def __init__(self):
        super().__init__()
        self.stop = False
        self.memory_buffer = []
        gc.collect()
        self.start()

    def get_memory(self):
        "Get memory of a process and its children."
        p = Process()
        memory = p.memory_info().rss
        for c in p.children():
            memory += c.memory_info().rss
        return memory

    def run(self):
        memory_start = self.get_memory()
        while not self.stop:
            self.memory_buffer.append(self.get_memory() - memory_start)
            time.sleep(0.2)
        gc.collect()

    def join(self):
        self.stop = True
        super().join()


##############################################################################
# Save memory by consuming the outputs of the tasks as fast as possible
##############################################################################

##############################################################################
# We create a task whose output takes about 15MB of RAM.
#

import numpy as np


def return_big_object(i):
    time.sleep(.1)
    return i * np.ones((10000, 200), dtype=np.float64)


##############################################################################
# We create a reduce step. The input will be a generator on big objects
# generated in parallel by several instances of :func:`return_big_object`.

def accumulator_sum(generator):
    result = 0
    for value in generator:
        result += value
        print(".", end="", flush=True)
    print("")
    return result


##############################################################################
# We process many of the tasks in parallel. If `return_generator=False`
# (default), we should expect a usage of more than 2GB in RAM. Indeed, all the
# results are computed and stored in ``res`` before being accumulated and
# collected by the gc.

from joblib import Parallel, delayed

monitor = MemoryMonitor()
print("Running tasks with return_generator=False...")
res = Parallel(n_jobs=2, return_generator=False)(
    delayed(return_big_object)(i) for i in range(150)
)
print("Accumulate results:", end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')

# Report memory usage
del res  # we clean the result to avoid memory border effects
monitor.join()
peak = max(monitor.memory_buffer) / 1e9
print(f"Peak memory usage: {peak:.2f}GB")


##############################################################################
# If we use ``return_generator=True``, ``res`` is simply a generator with the
# results that are ready. Here we consume the results as soon as they arrive
# with the ``accumulator_sum`` and once they have been used, they are collected
# by the gc. The memory footprint is thus less than 300MB.

monitor_gen = MemoryMonitor()
print("Create result generator with return_generator=True...")
res = Parallel(n_jobs=2, return_generator=True)(
    delayed(return_big_object)(i) for i in range(150)
)
print("Accumulate results:", end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')

# Report memory usage
del res  # we clean the result to avoid memory border effects
monitor_gen.join()
peak = max(monitor_gen.memory_buffer) / 1e6
print(f"Peak memory usage: {peak:.2f}MB")


##############################################################################
# We can then report the memory usage accross time of the two runs using the
# MemoryMonitor.
#
# In the first case, as the results accumulate in ``res``, the memory grows
# linearly and it is freed once the ``accumulator_sum`` function finishes.
#
# In the second case, the results are processed by the accumulator as soon as
# they arrive, and the memory does not need to be able to contain all
# the results.

import matplotlib.pyplot as plt
plt.semilogy(monitor.memory_buffer, label='return_generator=False')
plt.semilogy(monitor_gen.memory_buffer, label='return_generator=True')
plt.xlabel("Time")
plt.xticks([], [])
plt.ylabel("Memory usage")
plt.yticks([1e7, 1e8, 1e9], ['10MB', '100MB', '1GB'])
plt.legend()
plt.show()

##############################################################################
# It is important to note that with ``return_generator``, the results are
# still accumulated in RAM after computation. But as we asynchronously process
# them, they can be freed sooner. However, if the generator is not consomated,
# the memory still grows linearly.
