"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


import os
import sys
import time
import pydoc
import functools
import inspect

# Local imports
from .func_inspect import get_func_name, filter_args
from .logger import Logger, format_time
from .disk import rm_subdirs
from .job_store import DirectoryJobStore, COMPUTED, MUST_COMPUTE, WAIT

# Backwards compatability imports -- they used to be found here:
from .job_store import JobLibCollisionWarning

FIRST_LINE_TEXT = "# first line:"

# TODO: The following object should have a data store object as a sub
# object, and the interface to persist and query should be separated in
# the data store.
#
# This would enable creating 'Memory' objects with a different logic for 
# pickling that would simply span a MemorizedFunc with the same
# store (or do we want to copy it to avoid cross-talks?), for instance to
# implement HDF5 pickling. 

# TODO: Same remark for the logger, and probably use the Python logging
# mechanism.

# TODO: Track history as objects are called, to be able to garbage
# collect them.


################################################################################
# class `MemorizedFunc`
################################################################################
class MemorizedFunc(Logger):
    """ Callable object decorating a function for caching its return value 
        each time it is called.
    
        All values are cached on the filesystem, in a deep directory
        structure. Methods are provided to inspect the cache or clean it.

        Attributes
        ----------
        func: callable
            The original, undecorated, function.
        cachedir: string
            Path to the base cache directory of the memory context.
        ignore: list or None
            List of variable names to ignore when choosing whether to
            recompute.
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
            The memmapping mode used when loading from cache
            numpy arrays. See numpy.load for the meaning of the
            arguments. Only used if save_npy was true when the
            cache was created.
        verbose: int, optional
            The verbosity flag, controls messages that are issued as 
            the function is revaluated.
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------
   
    def __init__(self, func, cachedir=None, ignore=None, save_npy=True, 
                             mmap_mode=None, verbose=1, timestamp=None,
                             store=None):
        """
            Parameters
            ----------
            func: callable
                The function to decorate
            cachedir: string
                The path of the base directory to use as a data store.
                Should be None if and only if ``store`` is provided.
            ignore: list or None
                List of variable names to ignore.
            save_npy: boolean, optional
                If True, numpy arrays are saved outside of the pickle
                files in the cache, as npy files.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. Only used if save_npy was true when the
                cache was created.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued 
                as functions are revaluated. The higher, the more verbose
            timestamp: float, optional
                The reference time from which times in tracing messages
                are reported.
            store: object, optional
                Object implementing the job store API, see the
                ``joblib.job_store`` module. If provided, then
                ``cachedir`` should not be provided, and ``save_npy``
                and ``mmap_mode`` will be ignored.
        """
        Logger.__init__(self)
        self._verbose = verbose
        self.func = func
        if store is not None and cachedir is not None:
            raise TypeError('Provide either store or cachedir')
        if store is None:
            store = DirectoryJobStore(cachedir, save_npy=save_npy, mmap_mode=mmap_mode,
                                      verbose=verbose)
        self.store = store
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        if ignore is None:
            ignore = []
        self.ignore = ignore
        try:
            functools.update_wrapper(self, func)
        except:
            " Objects like ufunc don't like that "
        if inspect.isfunction(func):
            doc = pydoc.TextDoc().document(func
                                    ).replace('\n', '\n\n', 1)
        else:
            # Pydoc does a poor job on other objects
            doc = func.__doc__
        self.__doc__ = 'Memoized version of %s' % doc

    def __call__(self, *args, **kwargs):
        return self._compute(args, kwargs, force=False)

    def __reduce__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
            In addition, when unpickling, we run the __init__
        """
        return (self.__class__, (self.func, None, None, None, None, self._verbose,
                                 None, self.store))

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------
    def get_job(self, *args, **kwargs):
        filtered_args_dict = filter_args(self.func, self.ignore,
                                         *args, **kwargs)
        job = self.store.get_job(self.func, filtered_args_dict)
        return job
    
    def _compute(self, args_tuple, kwargs_dict, force):
        filtered_args_dict = filter_args(self.func, self.ignore,
                                         *args_tuple, **kwargs_dict)
        job = self.store.get_job(self.func, filtered_args_dict)
        try:
            if force:
                job.clear()

            # TODO Delegate this logging to the store, and remove
            # the hooks
            t0 = time.time()            
            def pre_load():
                if self._verbose > 1:
                    t = time.time() - self.timestamp
                    print '[Memory]% 16s: Loading %s...' % (
                        format_time(t),
                        self.format_signature(self.func)[0]
                        )
            def post_load():
                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = '%s cache loaded - %s' % (name, format_time(t))
                    print max(0, (80 - len(msg)))*'_' + msg
                
            status, output = job.load_or_lock(True, pre_load, post_load)
            if status == COMPUTED:
                return output
            elif status == MUST_COMPUTE:
                start_time = time.time()
                if self._verbose:
                    print self.format_call(*args_tuple, **kwargs_dict)
                output = self.func(*args_tuple, **kwargs_dict)
                job.persist_output(output)
                job.persist_input(args_tuple, kwargs_dict, filtered_args_dict)
                duration = time.time() - start_time
                if self._verbose:
                    _, name = get_func_name(self.func)
                    msg = '%s - %s' % (name, format_time(duration))
                    print max(0, (80 - len(msg)))*'_' + msg
                job.commit()
            else:
                assert False
        finally:
            job.close()
        return output

    def clear(self, warn=True):
        """ Empty the function's cache. 
        """
        self.store.clear(self.func, warn=warn)

    def call(self, *args, **kwargs):
        """ Force the execution of the function with the given arguments and 
            persist the output values.
        """
        return self._compute(args, kwargs, force=True)
    
    def format_call(self, *args, **kwds):
        """ Returns a nicely formatted statement displaying the function 
            call with the given arguments.
        """
        path, signature = self.format_signature(self.func, *args,
                            **kwds)
        msg = '%s\n[Memory] Calling %s...\n%s' % (80*'_', path, signature)
        return msg
        # XXX: Not using logging framework
        #self.debug(msg)

    def format_signature(self, func, *args, **kwds):
        # XXX: This should be moved out to a function
        # XXX: Should this use inspect.formatargvalues/formatargspec?
        module, name = get_func_name(func)
        module = [m for m in module if m]
        if module:
            module.append(name)
            module_path = '.'.join(module)
        else:
            module_path = name
        arg_str = list()
        previous_length = 0
        for arg in args:
            arg = self.format(arg, indent=2)
            if len(arg) > 1500:
                arg = '%s...' % arg[:700]
            if previous_length > 80:
                arg = '\n%s' % arg
            previous_length = len(arg)
            arg_str.append(arg)
        arg_str.extend(['%s=%s' % (v, self.format(i)) for v, i in
                                    kwds.iteritems()])
        arg_str = ', '.join(arg_str)

        signature = '%s(%s)' % (name, arg_str)
        return module_path, signature

    # XXX: Need a method to check if results are available.

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(func=%s, store=%s)' % (
                    self.__class__.__name__,
                    self.func,
                    self.store.repr_for_func(self.func), # ugh
                    )



################################################################################
# class `Memory`
################################################################################
class Memory(Logger):
    """ A context object for caching a function's return value each time it
        is called with the same input arguments.
    
        All values are cached on the filesystem, in a deep directory
        structure.

        see :ref:`memory`
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------
   
    def __init__(self, cachedir=None, save_npy=True, mmap_mode=None,
                       verbose=1, store=None):
        """
            Parameters
            ----------
            cachedir: string or None
                The path of the base directory to use as a data store
                or None. If neither ``cachedir`` nor ``store`` is given,
                no caching is done and the Memory object is completely transparent.
            save_npy: boolean, optional
                If True, numpy arrays are saved outside of the pickle
                files in the cache, as npy files.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. Only used if save_npy was true when the
                cache was created.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued 
                as functions are revaluated.
            store: object, optional
                Object implementing the job store API, see the
                ``joblib.job_store`` module. If provided, then
                ``cachedir`` should not be provided, and ``save_npy``
                and ``mmap_mode`` will be ignored.                
        """
        # XXX: Bad explaination of the None value of cachedir
        Logger.__init__(self)
        self._verbose = verbose
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self.timestamp = time.time()
        self.store = store
        if cachedir is None:
            self.cachedir = None
        else:
            self.cachedir = os.path.join(cachedir, 'joblib')
            if not os.path.exists(self.cachedir):
                os.makedirs(self.cachedir)


    def cache(self, func=None, ignore=None, verbose=None,
                        mmap_mode=False):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is 
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword 
            # arguments in decorators
            return functools.partial(self.cache, ignore=ignore)
        if self.cachedir is self.store is None:
            return func
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return MemorizedFunc(func, cachedir=self.cachedir,
                                   save_npy=self.save_npy,
                                   mmap_mode=mmap_mode,
                                   ignore=ignore,
                                   verbose=verbose,
                                   timestamp=self.timestamp,
                                   store=self.store)


    def clear(self, warn=True):
        """ Erase the complete cache directory.
        """
        if warn:
            self.warn('Flushing completely the cache')
        rm_subdirs(self.cachedir)


    def eval(self, func, *args, **kwargs):
        """ Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        """
        if self.cachedir is None:
            return func(*args, **kwargs)
        return self.cache(func)(*args, **kwargs)

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    repr(self.cachedir),
                    )


