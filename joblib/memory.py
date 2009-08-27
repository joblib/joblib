"""
A context object for caching a function's return value each time it
are called.

If called later with the same arguments, the cached value is returned, and
not re-evaluated. Slow for mutable types.

Taken from U{http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/466320}.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


import os
import shutil
from cPickle import dumps, PicklingError
import inspect
import itertools
import functools
import traceback
import logging

# Local imports
from .memoize import _function_code_hash

# XXX: Need to enable pickling, to use with multiprocessing.

################################################################################
class Memory(object):
    """ A context object for caching a function's return value each time 
    it are called.
    
    Cache is a file, hence memory is not lost even computer is shut down! :-)
    The file is syncronized anytime a value is added to it.
    
    """
    # A cache to store the previous function code, for faster disk
    # access
    _previous_func_code = dict()
   
    def __init__(self, cachedir, debug=False):
        self._debug = debug
        self._cachedir = cachedir
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)


    def eval(self, func, *args, **kwargs):
        # Compare the function code with the previous to see if the
        # function code has changed
        if not self._check_previous_func_code(func):
            self._cache_clear(func)


    def cache(self, func):
        """ A decorator.
        """
        # XXX: Should not be using closures: this is not pickleable
        @functools.wraps(func)
        def my_func(*args, **kwargs):
            return self.eval(func, *args, **kwargs)
        return my_func


    def _get_func_dir(self, func):
        """ Get the directory corresponding to the cache for the
            function.
        """
        module = func.__module__
        module = module.split('.')
        module.append(func.func_name)
        func_dir = os.path.join(self._cachedir, *module)
        if not os.path.exists(func_dir):
            os.makedirs(func_dir)
        return func_dir


    def _check_previous_func_code(self, func):
        func_code = _function_code_hash(func)
        func_dir = self._get_func_dir(func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        # I cannot use inspect.getsource because it is not
        # reliable when using IPython's magic "%run".

        if not os.path.exists(func_code_file): 
            file(func_code_file, 'w').write(func_code)
            return False
        elif not file(func_code_file).read() == func_code:
            # If the function has changed wipe the cache directory.
            shutil.rmtree(func_dir)
            os.makedirs(func_dir)
            file(func_code_file, 'w').write(func_code)
            return False
        else:
            return True

    def _cache_clear(self, func):
        func_code = _function_code_hash(func)
        func_dir = self._get_func_dir(func)
        func_code_file = os.path.join(func_dir, 'func_content.py')
        if self._debug:
            self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir)
        os.makedirs(func_dir)
        func_code_file = os.path.join(func_dir, 'func_content.py')
        file(func_code_file, 'w').write(func_code)


    def warn(self, msg):
        logging.warn("[memory]%s (%s line %i): %s" %
            ( self.func.func_name, self.func.__module__,
              self.func.func_code.co_firstlineno, msg)
                )


    def __call__(self, *args, **kwds):
        key = args
        if kwds:
            items = kwds.items()
            key = key + tuple(items)
        try:
            if key in self._cache:
                return self._cache[key]
            if self._debug:
                self.warn("Arguments not in cache.")
                self.print_call(*args, **kwds)
            self._cache[key] = result = self.func(*args, **kwds)
            if self._persist:
                # cache the result to file
                self._cache.sync()
            return result
        except TypeError:
            try:
                dump = dumps(key)
            except PicklingError:
                if self._debug:
                    self.warn("Cannot hash arguments.")
                    self.print_call(*args, **kwds)
                return self.func(*args, **kwds)
            else:
                try:
                    if dump in self._cache:
                        return self._cache[dump]
                    if self._debug:
                        self.warn("Arguments hash not in cache.")
                        self.print_call(*args, **kwds)
                except:
                    if self._debug:
                        self.warn("Error while unpickling for %s." % \
                                        self.func.func_name)
                        traceback.print_exc()
                    self._cache_clear()
                result = self.func(*args, **kwds)
                try:
                    self._cache[dump] = result
                except Exception, e:
                    if isinstance(e, _bsddb.DBRunRecoveryError):
                        self.warn('Unrecoverable DB error, clearing DB')
                        self._cache = None
                        os.unlink(self._cache_filename)
                        self._cache = sopen(self._cache_filename, 'c')
                        self._cache[dump] = result
                if self._persist:
                    # cache the result to file
                    self._cache.sync()
                return result


    def print_call(self, *args, **kwds):
        """ Print a debug statement displaying the function call with the 
            arguments.
        """
        self.warn('Calling %s(%s, %s)' % (self.func.func_name,
                                    repr(args)[1:-1], 
                                    ', '.join('%s=%s' % (v, i) for v, i
                                    in kwds.iteritems())))


    def clear(self):
        """ Erase the complete cache directory and log.
        """
        # TODO



