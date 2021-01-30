"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
 :class:`joblib.Parallel` to get a generator on the outputs of parallel jobs.
 We first create tasks that return results with large memory footprints.
 We parallelize several of those tasks such that we should observe a high
 memory usage if all of the outputs were to stack in RAM. We show that the
 memory is efficiently managed if we use the generator to perform a reduce
 step that progressively consumes the outputs and keeps the memory at an
 acceptable level. The RAM usage can depend on the behavior of the garbage
 collector, whose behavior can be hard to predict. Here we force it to ensure
 a low memory usage.

"""

##############################################################################
# Save memory by consuming the outputs of the tasks as fast as possible
##############################################################################
#
# We create a task whose output takes about 15MB of RAM

import time
import numpy as np


def return_big_object(i):
    time.sleep(1)
    return i * np.ones((10000, 200), dtype=np.float64)


##############################################################################
# We create a reduce step. The input will be a generator on big objects
# generated in parallel by several instances of :func:`return_big_object`.

import gc


def accumulator_sum(generator):
    result = 0
    for value in generator:
        result += value
        del value
        gc.collect()  # make sur to flush value from memory
    return result


###############################################################################
# We process many of the tasks in parallel. If `return_generator=False`
# (default), we should expect a usage of more than 3GB in RAM. Here we consume
# the outputs faster than the workers can compute new ones, and the overall
# memory footprint is less than 300MB.

from joblib import Parallel, delayed

with Parallel(n_jobs=2, return_generator=True) as parallel:
    out = parallel(delayed(return_big_object)(i) for i in range(200))
    reduced_output = accumulator_sum(out)
    print('All tasks completed and reduced successfully.')
