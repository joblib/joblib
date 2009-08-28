"""
A context object for caching a function's return value each time it
are called.

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
import functools
import traceback

# Local imports
from .hashing import get_func_code, get_func_name, hash
from .logger import Logger

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
   
    def __init__(self, func, cachedir, debug=False):
        self._debug = debug
        self._cachedir = cachedir
        self.func = func
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)
        functools.update_wrapper(self, func)


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
        output = self.func(*args, **kwargs)
        output_dir = self._get_output_dir(args, kwargs)
        self._persist_output(output, output_dir)
        return output


    def print_call(self, *args, **kwds):
        """ Print a debug statement displaying the function call with the 
            arguments.
        """
        self.warn('Calling %s(%s, %s)' % (self.func.func_name,
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


    def _read_output(self, args, kwargs):
        """ Read the results of a previous calculation from a file.
        """
        output_dir = self._get_output_dir(args, kwargs)
        output_file = file(os.path.join(output_dir, 'output.pkl'), 'r')
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
    """ A context object for caching a function's return value each time 
        it are called.
    
        All values are cached on the filesystem, in a deep directory
        structure.
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------
   
    def __init__(self, cachedir, debug=False):
        self._debug = debug
        self._cachedir = cachedir
        if not os.path.exists(self._cachedir):
            os.makedirs(self._cachedir)


    def cache(self, func):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.
        """
        return MemorizedFunc(func, cachedir=self._cachedir,
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
        return self.cache(func)(*args, **kwargs)

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    self._cachedir,
                    )


