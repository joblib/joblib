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
import time
import pydoc
try:
    import cPickle as pickle
except ImportError:
    import pickle
import functools
import traceback
import warnings
import inspect
from sqlite3 import OperationalError
try:
    # json is in the standard library for Python >= 2.6
    import json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        # Not the end of the world: we'll do without this functionality
        json = None

# Local imports
from .hashing import hash
from .func_inspect import get_func_code, get_func_name, filter_args
from .logger import Logger, format_time
from . import numpy_pickle
from .cache_db import CacheDB
from .disk import disk_used, memstr_to_kbytes, rm_subdirs, safe_listdir

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


################################################################################
# Helper functions

def extract_first_line(func_code):
    """ Extract the first line information from the function code
        text if available.
    """
    if func_code.startswith(FIRST_LINE_TEXT):
        func_code = func_code.split('\n')
        first_line = int(func_code[0][len(FIRST_LINE_TEXT):])
        func_code = '\n'.join(func_code[1:])
    else:
        first_line = -1
    return func_code, first_line


def cost(db_entry, current_time):
    """ The cost per cache entry for the cache replacement policy.
    """
    last_cost        = db_entry['last_cost']
    size             = db_entry['size']
    last_access      = db_entry['access_time']
    computation_time = db_entry['computation_time']
    delta_t = max(1e-6, current_time - last_access)
    alpha = 1 - computation_time/delta_t
    new_cost = alpha*last_cost + size
    return new_cost


def sort_entries(db):
    current_time = time.time()
    return sorted(db, key=lambda x: -cost(x, current_time))


def compress_cache(db, cachedir, fraction=.1):
    """ Cache replacement: remove 'fraction' of the size of the stored 
        cache.
    """
    index = db.get('__INDEX__')
    cache_size = -index['size']
    target_size = fraction * cache_size
    try:
        for db_entry in sort_entries(db):
            if db_entry['key'] == '__INDEX__':
                continue
            module = db_entry['module'].split('.')
            name = db_entry['func_name']
            argument_hash = db_entry['argument_hash']
            module.append(name)
            func_dir = os.path.join(cachedir, *module)
            argument_dir = os.path.join(func_dir, argument_hash)
            if os.path.exists(argument_dir):
                try:    
                    shutil.rmtree(argument_dir)
                except:
                    # XXX: Where is our logging framework?
                    print ('[joblib] Warning could not empty cache directory %s'
                            % argument_dir)
            try:
                db.remove(db_entry['key'])
                cache_size -= db_entry['size']
            except KeyError, OperationalError:
                # A KeyError can be created by a race-condition between
                # different processes trying to remove the same entry
                # An operational error means that the db is locked. Not a
                # big deal: we erased the directory, so JobLib will
                # figure out that the key is obsolete
                pass
            if safe_listdir(func_dir) == ['func_code.py']:
                try:    
                    shutil.rmtree(func_dir)
                except:
                    # XXX: Where is our logging framework?
                    print ('[joblib] Warning could not empty cache directory %s'
                            % func_dir)
            if cache_size <= target_size:
                break
    finally:
        db.update_entry('__INDEX__', size=-cache_size)


class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


################################################################################
# class `Memory`
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
   
    def __init__(self, func, cachedir, ignore=None, save_npy=True, 
                             mmap_mode=None, verbose=1, db=None,
                             limit=None, timestamp=None):
        """
            Parameters
            ----------
            func: callable
                The function to decorate
            cachedir: string
                The path of the base directory to use as a data store
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
                as functions are revaluated.
            db: CacheDB object or None
                The database to keep track of the access.
            limit: string of the form '1M' or None, optional
                The maximum size of the cache stored on disk
            timestamp: float, optional
                The reference time from which times in tracing messages
                are reported.
        """
        Logger.__init__(self)
        self._verbose = verbose
        self.cachedir = cachedir
        # Check that the given argument is OK
        assert limit is None or memstr_to_kbytes(limit) >= 40, ValueError(
            'The cache size limit should be greater than 40K, %s was passed'
            % limit)
        self.limit = limit
        self.func = func
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        self.db = db
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        if ignore is None:
            ignore = []
        self.ignore = ignore
        if not os.path.exists(self.cachedir):
            os.makedirs(self.cachedir)
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
        # Compare the function code with the previous to see if the
        # function code has changed
        db_entry = self.get_db_entry(*args, **kwargs)
        output_dir = db_entry['output_dir']
        # FIXME: The statements below should be try/excepted
        if (not self._check_previous_func_code(stacklevel=3) 
                or not 'size' in db_entry):
            return self.call(*args, **kwargs)
        else:
            try:
                # Update the stored cost
                current_time = time.time()
                new_cost = cost(db_entry, current_time)
                # XXX: We should probably add the option to commit only
                # every once in a while for speed reasons. For joblib, 
                # commiting is important only in multiprocess situations,
                # or in case of crashes. The commit policy could be
                # based on computation/loading time
                self.db.update_entry(db_entry['key'],
                            last_cost=new_cost,
                            access_time=current_time,
                    )

                # Return the stored value
                return self.load_output(output_dir)
            except Exception:
                # XXX: Should use an exception logger
                self.warn(
                'Exception while loading results for '
                '(args=%s, kwargs=%s)\n %s' %
                    (args, kwargs, traceback.format_exc())
                    )

                shutil.rmtree(output_dir)
                return self.call(*args, **kwargs)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------
   
    def get_db_entry(self, *args, **kwargs):
        output_dir, argument_hash = self.get_output_dir(*args, **kwargs)
        module, func_name  = get_func_name(self.func)
        module = '.'.join(module)
        key = ':'.join((module, func_name, argument_hash))
        if self.db is None or not os.path.exists(output_dir):
            # FIXME: Really ugly way of dealing we no db
            db_entry = dict()
        else:
            try:
                db_entry = self.db.get(key)
            except KeyError:
                # The key is not in the database, but the cache directory
                # may exist, we can try to rebuild the key
                input_repr = self._persist_input(None, *args, **kwargs)
                size = disk_used(output_dir)
                db_entry = self.db.get('__INDEX__')
                db_entry.update(key=key,
                                func_name=func_name, 
                                module=module, 
                                args=repr(input_repr), 
                                argument_hash=argument_hash,
                                # We are using as a creation time, the
                                # creation_time of the repo, as an
                                # access_time, we are using a date half time 
                                # between the current time and the
                                # creation_time
                                access_time=.5*(db_entry['creation_time'] +
                                                time.time()),
                                # A computation time of 100ms, as a guess
                                computation_time=100,
                                size=size,
                                last_cost=size + 1,
                            )

        db_entry['output_dir'] = output_dir
        return db_entry

    def get_output_dir(self, *args, **kwargs):
        """ Returns the directory in which are persisted the results
            of the function corresponding to the given arguments.

            The results can be loaded using the .load_output method.
        """
        coerce_mmap = (self.mmap_mode is not None)
        argument_hash = hash(filter_args(self.func, self.ignore,
                             *args, **kwargs), 
                             coerce_mmap=coerce_mmap)
        output_dir = os.path.join(self._get_func_dir(),
                                    argument_hash)
        return output_dir, argument_hash
        

    def _get_func_dir(self, mkdir=True):
        """ Get the directory corresponding to the cache for the
            function.
        """
        module, name = get_func_name(self.func)
        module.append(name)
        func_dir = os.path.join(self.cachedir, *module)
        if mkdir and not os.path.exists(func_dir):
            try:
                os.makedirs(func_dir)
            except OSError:
                """ Dir exists: we have a race condition here, when using 
                    multiprocessing.
                """
                # XXX: Ugly
        return func_dir


    def _write_func_code(self, filename, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        file(filename, 'w').write(func_code)


    def _check_previous_func_code(self, stacklevel=2):
        """ 
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        # Here, we go through some effort to be robust to dynamically
        # changing code and collision. We cannot inspect.getsource
        # because it is not reliable when using IPython's magic "%run".
        func_code, source_file, first_line = get_func_code(self.func)
        func_dir = self._get_func_dir()
        func_code_file = os.path.join(func_dir, 'func_code.py')

        try:
            old_func_code, old_first_line = \
                            extract_first_line(file(func_code_file).read())
        except IOError:
            # Either the file did not exist, or it has been erased while we 
            # weren't looking
            self._write_func_code(func_code_file, func_code, first_line)
            return False
        if old_func_code == func_code:
            return True

        # We have differing code, is this because we are refering to
        # differing functions, or because the function we are refering as 
        # changed?

        if old_first_line == first_line == -1:
            _, func_name = get_func_name(self.func, resolv_alias=False,
                                         win_characters=False)
            if not first_line == -1:
                func_description = '%s (%s:%i)' % (func_name, 
                                                source_file, first_line)
            else:
                func_description = func_name
            warnings.warn(JobLibCollisionWarning(
                "Cannot detect name collisions for function '%s'"
                        % func_description), stacklevel=stacklevel)

        # Fetch the code at the old location and compare it. If it is the
        # same than the code store, we have a collision: the code in the
        # file has not changed, but the name we have is pointing to a new
        # code block.
        if (not old_first_line == first_line 
                                    and source_file is not None
                                    and os.path.exists(source_file)):
            _, func_name = get_func_name(self.func, resolv_alias=False)
            num_lines = len(func_code.split('\n'))
            on_disk_func_code = file(source_file).readlines()[
                        old_first_line-1:old_first_line-1+num_lines-1]
            on_disk_func_code = ''.join(on_disk_func_code)
            if on_disk_func_code.rstrip() == old_func_code.rstrip():
                warnings.warn(JobLibCollisionWarning(
                'Possible name collisions between functions '
                "'%s' (%s:%i) and '%s' (%s:%i)" %
                (func_name, source_file, old_first_line, 
                 func_name, source_file, first_line)),
                 stacklevel=stacklevel)

        # The function has changed, wipe the cache directory.
        # XXX: Should be using warnings, and giving stacklevel
        self.clear(warn=True)
        return False


    def clear(self, warn=True):
        """ Empty the function's cache. 
        """
        # XXX: Need to flush the db also
        func_dir = self._get_func_dir(mkdir=False)
        if self._verbose and warn:
            self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir)
        os.makedirs(func_dir)
        func_code, _, first_line = get_func_code(self.func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        self._write_func_code(func_code_file, func_code, first_line)


    def call(self, *args, **kwargs):
        """ Force the execution of the function with the given arguments and 
            persist the output values.
        """
        start_time = time.time()
        if self._verbose:
            print self.format_call(*args, **kwargs)
        output_dir, argument_hash = self.get_output_dir(*args, **kwargs)
        output = self.func(*args, **kwargs)
        self._persist_output(output, output_dir)
        input_repr = self._persist_input(output_dir, *args, **kwargs)
        duration = time.time() - start_time
        if self.db is not None:
            # Add one to size to avoid it being 0
            size = disk_used(output_dir) + 1
            module, func_name  = get_func_name(self.func)
            module = '.'.join(module)
            key = ':'.join((module, func_name, argument_hash))
            self.db.new_entry(dict(
                        key=key,
                        func_name=func_name,
                        module=module,
                        args=repr(input_repr),
                        argument_hash=argument_hash,
                        creation_time=start_time,
                        access_time=start_time,
                        computation_time=duration,
                        size=size,
                        last_cost=float(size),
                    ))
            total_size = self.db.get('__INDEX__')['size'] - size
            self.db.update_entry('__INDEX__', 
                        size=total_size,
                        access_time=time.time()
                    )
            if ( self.limit is not None 
                    and -total_size > memstr_to_kbytes(self.limit)):
                # XXX: We should really have a store object
                compress_cache(self.db, self.cachedir)
        if self._verbose:
            _, name = get_func_name(self.func)
            msg = '%s - %s' % (name, format_time(duration))
            print max(0, (80 - len(msg)))*'_' + msg
        return output


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

    # Make make public

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
            output_file.close()


    def _persist_input(self, output_dir, *args, **kwargs):
        """ Save a small summary of the call using json format in the
            output directory.
        """
        argument_dict = filter_args(self.func, self.ignore,
                                    *args, **kwargs)

        input_repr = dict((k, repr(v)) for k, v in argument_dict.iteritems())
        if json is not None and output_dir is not None:
            # Make sure that our output_dir has not been deleted
            # in the mean time
            # XXX: We should have a function to create the output_dir,
            # that would populate it correctly (func_code.py)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            json.dump(
                input_repr,
                file(os.path.join(output_dir, 'input_args.json'), 'w'),
                )
        return input_repr

    def load_output(self, output_dir):
        """ Read the results of a previous calculation from the directory
            it was cached in.
        """
        if self._verbose > 1:
            t = time.time() - self.timestamp
            print '[Memory]% 16s: Loading %s...' % (
                                    format_time(t),
                                    self.format_signature(self.func)[0]
                                    )
        filename = os.path.join(output_dir, 'output.pkl')
        if self.save_npy:
            return numpy_pickle.load(filename, 
                                     mmap_mode=self.mmap_mode)
        else:
            output_file = file(filename, 'r')
            return pickle.load(output_file)

    # XXX: Need a method to check if results are available.

    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(func=%s, cachedir=%s)' % (
                    self.__class__.__name__,
                    self.func,
                    repr(self.cachedir),
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
                       verbose=1, limit=None):
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
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued 
                as functions are revaluated.
            limit: string of the form '1M' or None, optional
                The maximum size of the cache stored on disk
        """
        # XXX: Bad explaination of the None value of cachedir
        Logger.__init__(self)
        self._verbose = verbose
        self.save_npy = save_npy
        self.mmap_mode = mmap_mode
        # Check that the given argument is OK
        assert limit is None or memstr_to_kbytes(limit) >= 40, ValueError(
            'The cache size limit should be greater than 40K, %s was passed'
            % limit)
        self.limit = limit
        self.timestamp = time.time()
        if cachedir is None:
            self.cachedir = None
            self.db = None
        else:
            self.cachedir = os.path.join(cachedir, 'joblib')
            if not os.path.exists(self.cachedir):
                os.makedirs(self.cachedir)
            self.db = CacheDB(filename=os.path.join(self.cachedir,
                                                    'db.sqlite'))


    def cache(self, func=None, ignore=None):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

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
        if self.cachedir is None:
            return func
        return MemorizedFunc(func, cachedir=self.cachedir,
                                   save_npy=self.save_npy,
                                   mmap_mode=self.mmap_mode,
                                   ignore=ignore,
                                   verbose=self._verbose, 
                                   db=self.db, 
                                   limit=self.limit,
                                   timestamp=self.timestamp)


    def clear(self, warn=True):
        """ Erase the complete cache directory.
        """
        if warn:
            self.warn('Flushing completely the cache')
        rm_subdirs(self.cachedir)
        if self.db is not None:
            self.db.clear()


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


    # Context manager methods for the Memory object.
    # Closes the database when leaving the context.
    # Using the context manager avoids having to write a long try ... except
    # statement around every test.

    def __enter__(self):
        return self


    def __exit__(self, type, value, traceback):
        if self.db is not None:
            self.db.close()


    #-------------------------------------------------------------------------
    # Private `object` interface
    #-------------------------------------------------------------------------
   
    def __repr__(self):
        return '%s(cachedir=%s)' % (
                    self.__class__.__name__,
                    repr(self.cachedir),
                    )
