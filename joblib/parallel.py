"""
Helpers for embarassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: Gael Varoquaux
# License: BSD 3 clause

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

import functools
import pickle


def delayed(function):
    """ Decorator used to capture the arguments of a function.
    """
    # Try to pickle the input function, to catch the problems early when
    # using with multiprocessing
    pickle.dumps(function)

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    return delayed_function
    

class Parallel(object):
    """ Helper class for readable parallel mapping.

        Parameters
        -----------
        n_jobs: int
            The number of jobs to use for the computation. If -1 all CPUs
            are used.

        Example
        --------

        >>> from math import sqrt
        >>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs
        # Not starting the pool in the __init__ is a design decision, to
        # be able to close it ASAP, and not burden the user with closing
        # it.

    def __call__(self, iterable):
        n_jobs = self.n_jobs
        if n_jobs is None or multiprocessing is None or n_jobs == 1:
            n_jobs = 1
            from __builtin__ import apply
        else:
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(n_jobs)
            apply = pool.apply_async

        output = list()
        try:
            for function, args, kwargs in iterable:
                output.append(apply(function, args, kwargs))

            if n_jobs > 1:
                output = [job.get() for job in output]
        finally:
            if n_jobs > 1:
                pool.close()
                pool.join()
        return output



