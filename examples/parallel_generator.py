"""
========================================
Returning a generator in joblib.Parallel
========================================

This example illustrates memory optimization enabled by using
 :class:`joblib.Parallel` to get a generator on the output of parallel jobs.
 We first create a task that is very memory demanding. We parallelize
 several of those tasks such that we should expect a memory error if all
 of the outputs were to stack. We show that the memory is efficiently
 managed if we use the generator to perform a reduce step that progressively
 consumes the outputs and keep the memory at an acceptable level.

"""

##############################################################################
# Save memory by consuming the outputs of the tasks as soon as it's available
##############################################################################
#
# We create a task whose output takes about 0.6GB of available memory
import time
import numpy as np

def memory_consuming_task(i):
    time.sleep(2)
    return i*np.ones((100000,100),dtype=np.float64)


###############################################################################
# We process those many of those tasks in parallel. Normally we should expect
# a usage of more than 50GB in RAM (and the process would terminate with a
# MemoryError) before we get there on most computers). Here we reduce the
# outputs faster than the workers need to compute new ones, which controls
# the memory usage.

from joblib import Parallel, delayed
pool = Parallel(n_jobs=2, return_generator=True)
out = pool(delayed(memory_consuming_task)(i) for i in range(100))
reduced_output = sum(out)
print('All tasks completed and reduced successfully.')
