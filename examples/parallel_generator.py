"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
:class:`joblib.Parallel` to get a generator on the outputs of parallel jobs.
We first create tasks that return results with large memory footprints.
If we call :class:`~joblib.Parallel` for several of these tasks directly, we
observe a high memory usage, as all the results are held in RAM before being
processed

Using ``return_as='generator'`` allows to progressively consume the outputs
as they arrive and keeps the memory at an acceptable level.

In this case, the output of the `Parallel` call is a generator that yields the
results in the order the tasks have been submitted with. Future releases are
also planned to support the ``return_as="unordered_generator"`` parameter to
have the generator yield results as soon as available.

"""

##############################################################################
# ``MemoryMonitor`` helper
##############################################################################

##############################################################################
# The following class is an helper to monitor the memory of the process and its
# children in another thread, so we can display it afterward.
#
# We will use ``psutil`` to monitor the memory usage in the code. Make sure it
# is installed with ``pip install psutil`` for this example.


import time
from psutil import Process
from threading import Thread


class MemoryMonitor(Thread):
    """Monitor the memory usage in MB in a separate thread.

    Note that this class is good enough to highlight the memory profile of
    Parallel in this example, but is not a general purpose profiler fit for
    all cases.
    """
    def __init__(self):
        super().__init__()
        self.stop = False
        self.memory_buffer = []
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
# generated in parallel by several instances of ``return_big_object``.

def accumulator_sum(generator):
    result = 0
    for value in generator:
        result += value
        print(".", end="", flush=True)
    print("")
    return result


##############################################################################
# We process many of the tasks in parallel. If ``return_as="list"`` (default),
# we should expect a usage of more than 2GB in RAM. Indeed, all the results
# are computed and stored in ``res`` before being processed by
# `accumulator_sum` and collected by the gc.

from joblib import Parallel, delayed

monitor = MemoryMonitor()
print("Running tasks with return_as='list'...")
res = Parallel(n_jobs=2, return_as="list")(
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
# If we use ``return_as="generator"``, ``res`` is simply a generator on the
# results that are ready. Here we consume the results as soon as they arrive
# with the ``accumulator_sum`` and once they have been used, they are collected
# by the gc. The memory footprint is thus reduced, typically around 300MB.

monitor_gen = MemoryMonitor()
print("Create result generator with return_as='generator'...")
res = Parallel(n_jobs=2, return_as="generator")(
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
plt.figure(0)
plt.semilogy(
    np.maximum.accumulate(monitor.memory_buffer),
    label='return_as="list"'
)
plt.semilogy(
    np.maximum.accumulate(monitor_gen.memory_buffer),
    label='return_as="generator"'
)
plt.xlabel("Time")
plt.xticks([], [])
plt.ylabel("Memory usage")
plt.yticks([1e7, 1e8, 1e9], ['10MB', '100MB', '1GB'])
plt.legend()
plt.show()

##############################################################################
# It is important to note that with ``return_as="generator"``, the results are
# still accumulated in RAM after computation. But as we asynchronously process
# them, they can be freed sooner. However, if the generator is not consumed
# the memory still grows linearly.
#
# NB: the example uses `batch_size="auto"` rather than `batch_size="auto"` to
# prevent auto-batching from grouping together fast tasks and delayed tasks
# in the same batch, which makes the example takwaways less consistently
# reproducible.

##############################################################################
# Now let's add some complexity to the problem and assume that some of the
# tasks will complete much slowly than others.


def return_big_object_delayed(i):
    if (i + 20) % 60:
        time.sleep(0.1)
    else:
        time.sleep(5)
    return i * np.ones((10000, 200), dtype=np.float64)


##############################################################################
# There's the same noticeably high RAM usage when using `return_as="list"`...

monitor_delayed = MemoryMonitor()
print("Running delayed tasks with return_as='list'...")
res = Parallel(n_jobs=2, return_as="list", batch_size=1)(
    delayed(return_big_object_delayed)(i) for i in range(150)
)
print("Accumulate results:", end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')

# Report memory usage
del res  # we clean the result to avoid memory border effects
monitor_delayed.join()
peak = max(monitor_delayed.memory_buffer) / 1e9
print(f"Peak memory usage: {peak:.2f}GB")

##############################################################################
# But now using ``return_as="generator"`` does not provide as much of a relief
# on memory allocation. The reason is that, because the generator respects the
# order the tasks has been submitted with, the tasks that are slower than the
# other tasks will delay the corresponding iteration of the generator, and
# during this time subsequent shorter tasks will be done by other processes
# and have time to accumulate in RAM.

monitor_delayed_gen = MemoryMonitor()
print("Create result generator on delayed tasks with return_as='generator'...")
res = Parallel(n_jobs=2, return_as="generator", batch_size=1)(
    delayed(return_big_object_delayed)(i) for i in range(150)
)
print("Accumulate results:", end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')

# Report memory usage
del res  # we clean the result to avoid memory border effects
monitor_delayed_gen.join()
peak = max(monitor_delayed_gen.memory_buffer) / 1e6
print(f"Peak memory usage: {peak:.2f}MB")

##############################################################################
# If we use ``return_as="generator_unordered"``, ``res`` will not enforce any
# order when returning the results, and will simply enable iterating on the
# results as soon as it's available. The peak memory usage is controlled again
# to a lower level, since that results can be comsumed immediately rather than
# being delayed by the compute of a slower task that has been submitted
# earlier.
# Beware that the downstream consumer of the results must not expect them to
# be returned with the order the tasks have been submitted with, neither with
# any deterministic order, since the tasks completion can now depend on the
# availability of the workers, which can be affected by external events, such
# as system load, implementation details in the backend, etc. In this example,
# it is not required to enforce an order, since the accumulator use a
# commutative operation (sum), so we can safely use ``generator_unordered``
# mode.

monitor_delayed_gen_unordered = MemoryMonitor()
print(
  "Create result generator on delayed tasks with "
  "return_as='generator_unordered'..."
)
res = Parallel(n_jobs=2, return_as="generator_unordered", batch_size=1)(
    delayed(return_big_object_delayed)(i) for i in range(150)
)
print("Accumulate results:", end='')
res = accumulator_sum(res)
print('All tasks completed and reduced successfully.')

# Report memory usage
del res  # we clean the result to avoid memory border effects
monitor_delayed_gen_unordered.join()
peak = max(monitor_delayed_gen_unordered.memory_buffer) / 1e6
print(f"Peak memory usage: {peak:.2f}MB")


##############################################################################
# Notice how the plot for ``'return_as="generator'`` now show a peaks where
# slow jobs resulted in a congestion of tasks and an accumulation of results
# in RAM, but it's smoothed out when using
# ``'return_as="generator_unordered"``.

plt.figure(1)
plt.semilogy(
    np.maximum.accumulate(monitor_delayed.memory_buffer),
    label='return_as="list"'
)
plt.semilogy(
    np.maximum.accumulate(monitor_delayed_gen.memory_buffer),
    label='return_as="generator"'
)
plt.semilogy(
    np.maximum.accumulate(monitor_delayed_gen_unordered.memory_buffer),
    label='return_as="generator_unordered"'
)
plt.xlabel("Time")
plt.xticks([], [])
plt.ylabel("Memory usage")
plt.yticks([1e7, 1e8, 1e9], ['10MB', '100MB', '1GB'])
plt.legend()
plt.show()
