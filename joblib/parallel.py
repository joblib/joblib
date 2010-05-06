"""
Helpers for embarassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

import sys
import functools
try:
    import cPickle as pickle
except:
    import pickle

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

from .format_stack import format_exc, format_outer_frames

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
    """ Wraps a function to make it exception with full traceback in
        their representation.
        Useful for parallel computing with multiprocessing, for which 
        exceptions cannot be captured.
    """

    def __init__(self, func):
        self.func = func


    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except:
            e_type, e_value, e_tb = sys.exc_info()
            text = format_exc(e_type, e_value, e_tb, context=10,
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
                        this_report = format_outer_frames(
                                                context=10,
                                                stack_start=1,
                                                )
                        report = """Multiprocessing exception:
%s
---------------------------------------------------------------------------
Sub-process traceback: 
---------------------------------------------------------------------------
%s""" % (
                                    this_report,
                                    exception.message,
                                )
                        raise JoblibException(report)
        finally:
            if n_jobs > 1:
                pool.close()
                pool.join()
        return output



