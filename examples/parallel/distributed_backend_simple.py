"""
Using Dask for single-machine parallel computing
================================================

This example shows the simplest usage of the
`Dask <https://docs.dask.org/en/stable/>`_
backend on your local machine.

This is useful for prototyping a solution, to later be run on a truly
distributed Dask cluster (see the Dask_ documentation on deployment),
as the only change needed is the cluster class.

Another realistic usage scenario: combining dask code with joblib code,
for instance using dask for preprocessing data, and scikit-learn for
machine learning. In such a setting, it may be interesting to use
distributed as a backend scheduler for both dask and joblib, to
orchestrate the computation.

.. _Dask: https://docs.dask.org/en/stable/deploying.html#distributed-computing

"""

###############################################################################
# Setup the distributed client
###############################################################################
from dask.distributed import Client, LocalCluster

# replace with whichever cluster class you're using
cluster = LocalCluster()
# connect client to your cluster
client = Client(cluster)

# Monitor your computation with the Dask dashboard
print(client.dashboard_link)

###############################################################################
# Run parallel computation using dask.distributed
###############################################################################

import time
import joblib


def long_running_function(i):
    time.sleep(0.1)
    return i


###############################################################################
# The verbose messages below show that the backend is indeed the
# dask.distributed one
with joblib.parallel_backend("dask"):
    joblib.Parallel(verbose=100)(
        joblib.delayed(long_running_function)(i) for i in range(10)
    )

###############################################################################
