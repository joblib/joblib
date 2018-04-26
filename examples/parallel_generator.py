"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
 :class:`joblib.Parallel` to get a generator on the output of parallel jobs.
 We first create tasks that are very memory demanding. We parallelize
 several of those tasks such that we should observe a high memory usage if all
 of the outputs were to stack. We show that the memory is efficiently
 managed if we use the generator to perform a reduce step that progressively
 consumes the outputs and keeps the memory at an acceptable level.

"""

##############################################################################
# Save memory by consuming the outputs of the tasks as fast as possible
##############################################################################
#
# We create a task whose output takes about 15MB of RAM
import time
import numpy as np


def memory_demanding_task(i):
    time.sleep(1)
    return i * np.ones((10000, 200), dtype=np.float64)


###############################################################################
# We process many of those tasks in parallel. Normally, we should expect
# a usage of more than 3GB in RAM. Here we reduce the outputs faster than the
# workers can compute new ones. We see a clear benefit: the overall memory
# footprint is at least twice less.

from joblib import Parallel, delayed
out = Parallel(n_jobs=2, return_generator=True)(
    delayed(memory_demanding_task)(i) for i in range(200))
reduced_output = sum(out)
print('All tasks completed and reduced successfully.')
