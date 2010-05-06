"""
Helpers for embarassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: Gael Varoquaux
# License: BSD 3 clause

import sys

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

import functools
import pickle

from .exception_fmt import print_exc, print_outer_frame

################################################################################

class JoblibException(Exception):
    """ A simple exception with an error message that you can get to.
    """

    def __init__(self, message):
        self.message = message

    def __reduce__(self):
        # For pickling
        return self.__class__, (self.message,), {}

    def __repr__(self):
        return '%s\n%s\n%s\n%s' % (
                    self.__class__.__name__,
                    75*'_',
                    self.message,
                    75*'_')

    __str__ = __repr__


class SafeFunction(object):
    """ Wraps a function to make it raise errors with full traceback. 
        Useful for parallel computing, for which errors cannot be
        captured.
    """

    def __init__(self, func):
        self.func = func


    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except:
            e_type, e_value, e_tb = sys.exc_info()
            text = print_exc(e_type, e_value, e_tb, context=10,
                             tb_offset=1)
            raise JoblibException(text)

 
################################################################################
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
                if n_jobs > 1:
                    function = SafeFunction(function)
                output.append(apply(function, args, kwargs))

            if n_jobs > 1:
                jobs = output
                output = list()
                for job in jobs:
                    try:
                        output.append(job.get())
                    except JoblibException, exception:
                        # Capture exception to add information on 
                        # the local stack in addition to the distant
                        # stack
                        this_text = print_outer_frame(
                                                context=10,
                                                stack_start=1,
                                                #stack_end=3,
                                                )
                        text = 'JoblibException: multiprocessing exception\n%s\n%s\nSub-process traceback:\n%s\n%s' % (
                                    this_text,
                                    75*'.',
                                    75*'.',
                                    exception.message,
                                )
                        raise JoblibException(text)
        finally:
            if n_jobs > 1:
                pool.close()
                pool.join()
        return output



