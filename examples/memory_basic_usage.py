"""
========================
How to use joblib.Memory
========================

This example illustrates the usage of :class:`joblib.Memory`. We illustrate
usages with a function and a method.

"""

###############################################################################
# Without :class:`joblib.Memory`
###############################################################################
# 
# To show the benefit of using :class:`joblib.Memory`, we will reduce the
# dimension of some data using a principal components analysis (i.e. ``pca``).
# Indeed, this example will greatly benefit from caching since the internal
# computation in ``pca`` is expensive. First, we will check the time required
# to perform the decomposition on a large array of data.

import time
import numpy as np


def pca(data, n_components=2):
    """Principal components analysis.

    Parameters
    ----------
    data : ndarray
        The data to decompose.

    n_components : int, default=2
        The number of components to keep.

    Return
    ------
    data_transformed : ndarray
        The original data projected on the `n_components` principal components.

    """
    data_centered = data - np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data_centered, full_matrices=False)
    U = U[:, :n_components]
    U *= np.sqrt(data_centered.shape[0] - 1)
    return U


rng = np.random.RandomState(42)
data = rng.randn(int(1e8), 10)
tic = time.time()
data_trans = pca(data)
toc = time.time()

print('\nPCA took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# Caching the result of a function avoiding recomputing
###############################################################################
# 
# In the case that we would need to call ``pca`` function several time with
# the same input data, it is beneficial to avoid recomputing the same results
# over and over since that this function is time consuming. We can use
# :class:`joblib.Memory` to avoid such recomputing. A :class:`joblib.Memory`
# instance can be created by passing a directory in which we want to store the
# results.

from joblib import Memory
cachedir = './pca_caching'
memory = Memory(cachedir=cachedir, verbose=0)


def pca_cached(data, n_components=2):
    data_centered = data - np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data_centered, full_matrices=False)
    U = U[:, :n_components]
    U *= np.sqrt(data_centered.shape[0] - 1)
    return U


pca_cached = memory.cache(pca_cached)
tic = time.time()
data_trans = pca_cached(data)
toc = time.time()

print('\nPCA took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the first call, the results will be cached. Therefore, the computation
# time correspond to the time to compute the results plus the time to dump the
# results into the disk.

tic = time.time()
data_trans = pca_cached(data)
toc = time.time()

print('\nPCA took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# At the second call, the computation time is largely reduced since that the
# results are obtained by loading the data previously dumped on the disk
# instead of recomputing the results.

###############################################################################
# Using :class:`joblib.Memory` with a method
###############################################################################
# 
# :class:`joblib.Memory` is designed to work with pure functions. When you want
# to cache a method within a class, you need to create and cache a pure
# function and use it inside the class.


def _pca_cached(data, n_components):
    data_centered = data - np.mean(data, axis=0)
    U, S, V = np.linalg.svd(data_centered, full_matrices=False)
    U = U[:, :n_components]
    U *= np.sqrt(data_centered.shape[0] - 1)
    return U


class PCA(object):
    """Principal components analysis.

    Parameters
    ----------
    n_components : int, default=2
        The number of components to keep.

    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def transform(self, data):
        """Transform the data.

        Parameters
        ----------
        data : ndarray
            The data to decompose.

        Return
        ------
        data_transformed : ndarray
            The original data projected on the `n_components` principal
            components.

        """
        pca_cached = memory.cache(_pca_cached)
        return pca_cached(data, self.n_components)


transformer = PCA()
tic = time.time()
data_trans = transformer.transform(data)
toc = time.time()

print('\nPCA took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################

tic = time.time()
data_trans = transformer.transform(data)
toc = time.time()

print('\nPCA took {:.2f} s to compute.'.format(toc - tic))
print('\nThe transformed data are:\n {}'.format(data_trans))

###############################################################################
# As expected, the second call to the ``transform`` method load the results
# which have been cached.

###############################################################################
# Clean up cache directory
###############################################################################

import shutil
try:
    shutil.rmtree(cachedir)
except:  # noqa
    print('We could not remove the folder containing the cached data.')
