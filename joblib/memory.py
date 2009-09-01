"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


import os
import shutil
import sys
try:
    import cPickle as pickle
except ImportError:
    import pickle
import functools
import traceback

# Local imports
from .hashing import get_func_code, get_func_name, hash
from .logger import Logger
from . import numpy_pickle

# TODO: The following object should have a data store object as a sub
# object, and the interface to persist and query should be separated in
# the data store.

################################################################################
# class `Memory`
################################################################################
class MemorizedFunc(Logger):
    """ A functor (callable object) for caching a function's return value 
        each time it are called.
    
        All values are cached on the filesystem, in a deep directory
        structure.
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------
   
    def __init__(self, func, cachedir, save_npy=True, 
                             mmap_mode=None, debug=False):
        """
            Parameters
            ----------
            func: callable
                The function to decorate
            cachedir: string
                The path of the base directory to use as a data store
            save_npy: boolean, optional
                If True, numpy arrays are saved outside of the pickle
                files in the cache, as npy files.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. Only used if save_npy was true when the
                cache was created.
            debug: boolean, optional
                If True, debug messages will be issued as functions 
                are revaluated.
        """
        Logger.__init__(self)
        self._debug = debug
        self._cachedir = cachedir
        self.func = func
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)
        try:
            functools.update_wrapper(self, func)
        except:
            " Objects like ufunc don't like that "


    def __call__(self, *args, **kwargs):
        # Compare the function code with the previous to see if the
        # function code has changed
        output_dir = self._get_output_dir(args, kwargs)
        if not (self._check_previous_func_code() and 
                                 os.path.exists(output_dir)):
            return self._call(args, kwargs)
        else:
            try:
                return self._read_output(args, kwargs)
            except Exception, e:
                # XXX: Should use an exception logger
                self.warn('Exception while loading results for '
                '(args=%s, kwargs=%s)\n %s' %
                    (args, kwargs, traceback.format_exc())
                    )
                      
                shutil.rmtree(output_dir)
                return self._call(args, kwargs)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------
   
    def _get_func_dir(self, mkdir=True):
        """ Get the directory corresponding to the cache for the
            function.
        """
        module, name = get_func_name(self.func)
        module.append(name)
        func_dir = os.path.join(self._cachedir, *module)
        if mkdir and not os.path.exists(func_dir):
            os.makedirs(func_dir)
        return func_dir


    def _get_output_dir(self, args, kwargs):
        output_dir = os.path.join(self._get_func_dir(self.func),
                                  hash((args, kwargs)))
        return output_dir
        

    def _check_previous_func_code(self):
        func_code = get_func_code(self.func)
        func_dir = self._get_func_dir()
        func_code_file = os.path.join(func_dir, 'func_code.py')
        # I cannot use inspect.getsource because it is not
        # reliable when using IPython's magic "%run".

        if not os.path.exists(func_code_file): 
            file(func_code_file, 'w').write(func_code)
            return False
        elif not file(func_code_file).read() == func_code:
            # If the function has changed, wipe the cache directory.
            self.clear()
            return False
        else:
            return True


    def clear(self):
        """ Empty the function's cache. 
        """
        func_dir = self._get_func_dir(mkdir=False)
        if self._debug:
            self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir)
        os.makedirs(func_dir)
        func_code = get_func_code(self.func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        file(func_code_file, 'w').write(func_code)


    def _call(self, args, kwargs):
        """ Execute the function and persist the output arguments.
        """
        if self._debug:
            print self.print_call(*args, **kwargs)
        output = self.func(*args, **kwargs)
        output_dir = self._get_output_dir(args, kwargs)
        self._persist_output(output, output_dir)
        return output


    def print_call(self, *args, **kwds):
        """ Print a debug statement displaying the function call with the 
            arguments.
        """
        module, name = get_func_name(self.func)
        module = [m for m in module if m]
        if module:
            module.append(name)
            module = '.'.join(module)
        else:
            module = name
        indent = len(name)
        if args:
            args = self.format(args, indent=indent)[1:-2]
        else:
            args = ''
        kwds = ', '.join(('%s=%s' % (v, self.format(i)) for v, i in
                                kwds.iteritems()))
        if kwds and args:
            args += ', '

        msg = 'DBG:Calling %s\n%s(%s%s)' % (module, name, args, kwds)
        return msg
        # XXX: Not using logging framework
        #self.debug(msg)


    def _persist_output(self, output, dir):
        """ Persist the given output tuple in the directory.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        filename = os.path.join(dir, 'output.pkl')

        if 'numpy' in sys.modules and self.save_npy:
            numpy_pickle.dump(output, filename) 
        else:
            output_file = file(filename, 'w')
            pickle.dump(output, output_file, protocol=2)


    def _read_output(self, args, kwargs):
        """ Read the results of a previous calculation from a file.
        """
        output_dir = self._get_output_dir(args, kwargs)
        filename = os.path.join(output_dir, 'output.pkl')
        output_file = file(filename, 'r')
        if self.save_npy:
            return numpy_pickle.load(filename, 
                                     mmap_mode=self.mmap_mode)
        else:
            return pickle.load(output_file)

    # XXX: Need a method to check if results are available.

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(func=%s, cachedir=%s)' % (
                    self.__class__.__name__,
                    self.func,
                    self._cachedir,
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
   
    def __init__(self, cachedir, save_npy=True, mmap_mode=None,
                       debug=False):
        """
            Parameters
            ----------
            cachedir: string or None
                The path of the base directory to use as a data store
                or None. If None is given, no caching is done and
                the Memory object is completely transparent.
            save_npy: boolean, optional
                If True, numpy arrays are saved outside of the pickle
                files in the cache, as npy files.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. Only used if save_npy was true when the
                cache was created.
            debug: boolean, optional
                If True, debug messages will be issued as functions 
                are revaluated.
        """
        # XXX: Bad explaination of the None value of cachedir
        self._debug = debug
        self._cachedir = cachedir
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)


    def cache(self, func):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.
        """
        if self._cachedir is None:
            return func
        return MemorizedFunc(func, cachedir=self._cachedir,
                                   save_npy=self.save_npy,
                                   mmap_mode=self.mmap_mode,
                                   debug=self._debug)


    def clear(self):
        """ Erase the complete cache directory.
        """
        self.warn('Flushing completely the cache')
        shutil.rmtree(self._cachedir)
        os.makedirs(self._cachedir)


    def eval(self, func, *args, **kwargs):
        """ Eval function func with arguments `*args` and `**kwargs`,
            in the context of the memory.

            This method works similarly to the builtin `apply`, except
            that the function is called only if the cache is not
            up to date.

        """
        if self._cachedir is None:
            return func(*args, **kwargs)
        return self.cache(func)(*args, **kwargs)

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    self._cachedir,
                    )


