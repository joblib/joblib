"""
Using dask distributed for single-machine parallel computing
=============================================================

This example shows the simplest usage of the dask `distributed
<https://distributed.readthedocs.io>`__ backend, on the local computer.

This is useful for prototyping a solution, to later be run on a truly
distributed cluster, as the only change to be made is the address of the
scheduler.

Another realistic usage scenario: combining dask code with joblib code,
for instance using dask for preprocessing data, and scikit-learn for
machine learning. In such a setting, it may be interesting to use
distributed as a backend scheduler for both dask and joblib, to
orchestrate well the computation.

"""

###############################################################################
# Setup the distributed client
###############################################################################
from dask.distributed import Client

# If you have a remote cluster running Dask
# client = Client('tcp://scheduler-address:8786')

# If you want Dask to set itself up on your personal computer
client = Client(processes=False)

###############################################################################
# Run parallel computation using dask.distributed
###############################################################################

import time
import joblib


def long_running_function(i):
    time.sleep(.1)
    return i


###############################################################################
# The verbose messages below show that the backend is indeed the
# dask.distributed one
with joblib.parallel_backend('dask'):
    joblib.Parallel(verbose=100)(
        joblib.delayed(long_running_function)(i)
        for i in range(10))

###############################################################################
# Progress in computation can be followed on the distributed web
# interface, see http://dask.pydata.org/en/latest/diagnostics-distributed.html
