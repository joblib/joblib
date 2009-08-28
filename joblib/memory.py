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
try:
    import cPickle as pickle
except ImportError:
    import pickle
import inspect
import hashlib
import functools
import traceback
import logging
import types

# Local imports
from .hashing import get_func_code, get_func_name, hash, NON_MUTABLE_TYPES


# XXX: Need to enable pickling, to use with multiprocessing.

################################################################################
# class `Memory`
################################################################################
class Memory(object):
    """ A context object for caching a function's return value each time 
        it are called.
    
        All values are cached on the filesystem, in a deep directory
        structure.
    """
    # A cache to store the previous function code, for faster disk
    # access
    _previous_func_code = dict()

    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------
   
    def __init__(self, cachedir, debug=False):
        self._debug = debug
        self._cachedir = cachedir
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)


    def cache(self, func):
        """ A decorator.
        """
        # XXX: Should not be using closures: this is not pickleable
        @functools.wraps(func)
        def my_func(*args, **kwargs):
            return self.eval(func, *args, **kwargs)
        return my_func


    def warn(self, msg):
        logging.warn("[%s]: %s" % (self, msg))



    def clear(self):
        """ Erase the complete cache directory.
        """
        self.warn('Flushing completely the cache')
        shutil.rmtree(self._cachedir)
        os.makedirs(self._cachedir)


    def eval(self, func, *args, **kwargs):
        # Compare the function code with the previous to see if the
        # function code has changed
        output_dir = self._get_output_dir(func, args, kwargs)
        if not (self._check_previous_func_code(func) and 
                                 os.path.exists(output_dir)):
            return self._call(func, args, kwargs)
        else:
            try:
                return self._read_output(func, args, kwargs)
            except Exception, e:
                # XXX: Should use an exception logger
                self.warn('Exception while loading results for '
                '%s(args=%s, kwargs=%s)\n %s' %
                    (func, args, kwargs, traceback.format_exc())
                    )
                      
                shutil.rmtree(output_dir)
                return self._call(func, args, kwargs)



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
                dump = pickle.dumps(key)
            except pickle.PicklingError:
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


    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------
   
    def _get_func_dir(self, func, mkdir=True):
        """ Get the directory corresponding to the cache for the
            function.
        """
        module, name = get_func_name(func)
        module.append(name)
        func_dir = os.path.join(self._cachedir, *module)
        if mkdir and not os.path.exists(func_dir):
            os.makedirs(func_dir)
        return func_dir

    def _get_output_dir(self, func, args, kwargs):
        output_dir = os.path.join(self._get_func_dir(func), 
                            hash((args, kwargs)))
        return output_dir
        

    def _check_previous_func_code(self, func):
        func_code = get_func_code(func)
        func_dir = self._get_func_dir(func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        # I cannot use inspect.getsource because it is not
        # reliable when using IPython's magic "%run".

        if not os.path.exists(func_code_file): 
            file(func_code_file, 'w').write(func_code)
            return False
        elif not file(func_code_file).read() == func_code:
            # If the function has changed, wipe the cache directory.
            self._func_cache_clear(func)
            return False
        else:
            return True


    def _func_cache_clear(self, func):
        """ Empty a function's cache. 
        """
        func_dir = self._get_func_dir(func, mkdir=False)
        if self._debug:
            self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir)
        os.makedirs(func_dir)
        func_code = get_func_code(func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        file(func_code_file, 'w').write(func_code)


    def _call(self, func, args, kwargs):
        """ Execute the function and persist the output arguments.
        """
        output = func(*args, **kwargs)
        output_dir = self._get_output_dir(func, args, kwargs)
        self._persist_output(output, output_dir)
        return output


    def print_call(self, func, *args, **kwds):
        """ Print a debug statement displaying the function call with the 
            arguments.
        """
        self.warn('Calling %s(%s, %s)' % (func.func_name,
                                    repr(args)[1:-1], 
                                    ', '.join('%s=%s' % (v, i) for v, i
                                    in kwds.iteritems())))


    def _persist_output(self, output, dir):
        """ Persist the given output tuple in the directory.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        output_file = file(os.path.join(dir, 'output.pkl'), 'w')
        pickle.dump(output, output_file, protocol=2)


    def _read_output(self, func, args, kwargs):
        """ Read the results of a previous calculation from a file.
        """
        output_dir = self._get_output_dir(func, args, kwargs)
        output_file = file(os.path.join(output_dir, 'output.pkl'), 'r')
        return pickle.load(output_file)

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    self._cachedir,
                    )


