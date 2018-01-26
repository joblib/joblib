"""
Using distributed for single_machine parallel computing
========================================================

Realistic usage scenario: combining dask code with joblib code, for
instance using dask for preprocessing data, and scikit-learn for machine
learning.

"""

###############################################################################
# Setup the distributed client
###############################################################################
from distributed import Client
# Typically, to execute on a remote machine, the address of the scheduler
# would go there
client = Client()

# This import registers the dask backend for joblib
import distributed.joblib

###############################################################################
# Run parallel computation using dask.distributed
###############################################################################

import time
import joblib

def long_running_function(i):
    time.sleep(.1)
    return i

with joblib.parallel_backend('dask.distributed',
                             scheduler_host=client.scheduler.address):
    joblib.Parallel(n_jobs=2, verbose=100)(
        joblib.delayed(long_running_function)(i)
        for i in range(10))
    # We can check that joblib is indeed using the dask.distributed
    # backend
    print(joblib.Parallel(n_jobs=1)._backend)

###############################################################################
# Progress in computation can be followed on the distributed web
# interface, see http://distributed.readthedocs.io/en/latest/web.html
