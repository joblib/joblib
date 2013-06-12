"""
A context object for caching a function's return value each time it
is called with the same input arguments.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.


from __future__ import with_statement
import os
import shutil
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
import json

# Local imports
from .hashing import hash
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_signature, format_call
from .logger import Logger, format_time, pformat
from . import numpy_pickle
from .disk import mkdirp, rm_subdirs

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


class JobLibCollisionWarning(UserWarning):
    """ Warn that there might be a collision between names of functions.
    """


###############################################################################
# class `MemorizedResult`
###############################################################################
def _get_func_fullname(func):
    """Compute the part of part associated with a function.

    See code of_cache_key_to_dir() for details
    """
    modules, funcname = get_func_name(func)
    modules.append(funcname)
    return os.path.join(*modules)


def _cache_key_to_dir(cachedir, func, argument_hash):
    """Compute directory associated with a given cache key.

    func can be a function or a string as returned by _get_func_fullname().
    """
    parts = [cachedir]
    if isinstance(func, basestring):
        parts.append(func)
    else:
        parts.append(_get_func_fullname(func))

    if argument_hash is not None:
        parts.append(argument_hash)
    return os.path.join(*parts)


class MemorizedResult(Logger):
    """Object representing a cached value.
    """
    def __init__(self, cachedir, func, argument_hash,
                 mmap_mode=None, signature='', verbose=0, timestamp=None):
        Logger.__init__(self)
        if isinstance(func, basestring):
            self.func = func
        else:
            self.func = _get_func_fullname(func)
        self.argument_hash = argument_hash
        self.cachedir = cachedir
        self.mmap_mode = mmap_mode

        self._output_dir = _cache_key_to_dir(cachedir, self.func,
                                             argument_hash)

        self.signature = signature
        self.verbose = verbose
        self.timestamp = timestamp

    def get(self):
        """Read value from cache and return it."""
        # See also MemorizedFunc.load_output()
        # Read signature from cache
        if self.verbose > 1:
            if self.timestamp is not None:
                t = "% 16s" % format_time(time.time() - self.timestamp)
            else:
                t = ""
            if self.verbose < 10:
                print('[Memory]%s: Loading %s...' % (t, str(self.signature)))
            else:
                print('[Memory]s: Loading %s from %s' % (
                                    t, str(self.signature), self._output_dir))
        filename = os.path.join(self._output_dir, 'output.pkl')
        if not os.path.isfile(filename):
            raise KeyError(
                "Non-existing cache value (may have been cleared).\n"
                "File %s does not exist" % filename)
        return numpy_pickle.load(filename, mmap_mode=self.mmap_mode)

    def clear(self):
        """Clear value from cache"""
        shutil.rmtree(self._output_dir, ignore_errors=True)

    def __repr__(self):
        return (self.__class__.__name__
                + '(cachedir="' + self.cachedir + '", '
                + 'func="' + self.func + '", argument_hash="'
                + self.argument_hash + '")')

    def __reduce__(self):
        return (self.__class__, (self.cachedir, self.func, self.argument_hash),
                {'mmap_mode': self.mmap_mode})


class NotMemorizedResult(object):
    """Class representing an arbitrary value.

    This class is a replacement for MemorizedResult when there is no cache.
    """
    __slots__ = ('value', 'valid')

    def __init__(self, value):
        self.value = value
        self.valid = True

    def get(self):
        if self.valid:
            return self.value
        else:
            raise KeyError("No value stored.")

    def clear(self):
        self.valid = False
        self.value = None

    def __repr__(self):
        if self.valid:
            return (self.__class__.__name__
                    + '(' + pformat(self.value) + ')')
        else:
            return self.__class__.__name__ + ' with no value'

    # __getstate__ and __setstate__ are required because of __slots__
    def __getstate__(self):
        return {"valid": self.valid, "value": self.value}

    def __setstate__(self, state):
        self.valid = state["valid"]
        self.value = state["value"]


###############################################################################
# class `NotMemorizedFunc`
###############################################################################
class NotMemorizedFunc(object):
    """No-op object decorating a function.

    This class replaces MemorizedFunc when there is no cache. It provides an
    identical API but does not write anything on disk.

    Attributes
    ----------
    func: callable
        Original undecorated function.
    """
    # Should be a light as possible (for speed)
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def call_and_shelve(self, *args, **kwargs):
        return NotMemorizedResult(self.func(*args, **kwargs))

    def __reduce__(self):
        return (self.__class__, (self.func,))

    def __repr__(self):
        return '%s(func=%s)' % (
                    self.__class__.__name__,
                    self.func
            )

    def clear(self, warn=True):
        # Argument "warn" is for compatibility with MemorizedFunc.clear
        pass


###############################################################################
# class `MemorizedFunc`
###############################################################################
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
            arguments.

        compress: boolean
            Whether to zip the stored data on disk. Note that compressed
            arrays cannot be read by memmapping.

        verbose: int, optional
            The verbosity flag, controls messages that are issued as
            the function is evaluated.
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------

    def __init__(self, func, cachedir, ignore=None, mmap_mode=None,
                 compress=False, verbose=1, timestamp=None):
        """
            Parameters
            ----------
            func: callable
                The function to decorate
            cachedir: string
                The path of the base directory to use as a data store
            ignore: list or None
                List of variable names to ignore.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued
                as functions are evaluated. The higher, the more verbose
            timestamp: float, optional
                The reference time from which times in tracing messages
                are reported.
        """
        Logger.__init__(self)
        self.mmap_mode = mmap_mode
        self.func = func
        if ignore is None:
            ignore = []
        self.ignore = ignore

        self._verbose = verbose
        self.cachedir = cachedir
        self.compress = compress
        if compress and self.mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped',
                          stacklevel=2)
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        mkdirp(self.cachedir)
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

    def call_and_shelve(self, *args, **kwargs):
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. cachedir=None in Memory).
        """
        # TODO: add signature (format_signature)
        self.__call__(*args, **kwargs)

        # FIXME: argument_hash is already computed in self.__call__
        argument_hash = self._get_argument_hash(*args, **kwargs)
        return MemorizedResult(self.cachedir, self.func, argument_hash)

    def __call__(self, *args, **kwargs):
        # Compare the function code with the previous to see if the
        # function code has changed
        output_dir, argument_hash = self._get_output_dir(*args, **kwargs)
        # FIXME: The statements below should be try/excepted
        if not (self._check_previous_func_code(stacklevel=3) and
                                 os.path.exists(output_dir)):
            if self._verbose > 10:
                _, name = get_func_name(self.func)
                self.warn('Computing func %s, argument hash %s in '
                          'directory %s'
                        % (name, argument_hash, output_dir))
            return self.call(*args, **kwargs)
        else:
            try:
                t0 = time.time()
                out = self.load_output(output_dir)
                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = '%s cache loaded - %s' % (name, format_time(t))
                    print(max(0, (80 - len(msg))) * '_' + msg)
                return out
            except Exception:
                # XXX: Should use an exception logger
                self.warn('Exception while loading results for '
                          '(args=%s, kwargs=%s)\n %s' %
                          (args, kwargs, traceback.format_exc()))

                shutil.rmtree(output_dir, ignore_errors=True)
                return self.call(*args, **kwargs)

    def __reduce__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
            In addition, when unpickling, we run the __init__
        """
        return (self.__class__, (self.func, self.cachedir, self.ignore,
                self.mmap_mode, self.compress, self._verbose))

    # Compatibility methods: deprecate.
    def format_signature(self, *args, **kwargs):
        return format_signature(self.func, *args, **kwargs)

    def format_call(self, *args, **kwargs):
        return format_call(self.func, args, kwargs)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _get_argument_hash(self, *args, **kwargs):
        return hash(filter_args(self.func, self.ignore,
                                         args, kwargs),
                             coerce_mmap=(self.mmap_mode is not None))

    def _get_output_dir(self, *args, **kwargs):
        """ Returns the directory in which are persisted the result
            of the function corresponding to the given arguments.

            The result can be loaded using the .load_output method.
        """
        argument_hash = self._get_argument_hash(*args, **kwargs)
        output_dir = os.path.join(self._get_func_dir(self.func),
                                  argument_hash)
        return output_dir, argument_hash

    get_output_dir = _get_output_dir  # backward compatibility

    def _get_func_dir(self, mkdir=True):
        """ Get the directory corresponding to the cache for the
            function.
        """
        func_dir = _cache_key_to_dir(self.cachedir, self.func, None)
        if mkdir:
            mkdirp(func_dir)
        return func_dir

    def _write_func_code(self, filename, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        func_code = '%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        with open(filename, 'w') as out:
            out.write(func_code)

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
            with open(func_code_file) as infile:
                old_func_code, old_first_line = \
                            extract_first_line(infile.read())
        except IOError:
                self._write_func_code(func_code_file, func_code, first_line)
                return False
        if old_func_code == func_code:
            return True

        # We have differing code, is this because we are referring to
        # differing functions, or because the function we are referring as
        # changed?

        _, func_name = get_func_name(self.func, resolv_alias=False,
                                     win_characters=False)
        if old_first_line == first_line == -1 or func_name == '<lambda>':
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
        if not old_first_line == first_line and source_file is not None:
            possible_collision = False
            if os.path.exists(source_file):
                _, func_name = get_func_name(self.func, resolv_alias=False)
                num_lines = len(func_code.split('\n'))
                with open(source_file) as f:
                    on_disk_func_code = f.readlines()[
                            old_first_line - 1
                            :old_first_line - 1 + num_lines - 1]
                on_disk_func_code = ''.join(on_disk_func_code)
                possible_collision = (on_disk_func_code.rstrip()
                                      == old_func_code.rstrip())
            else:
                possible_collision = source_file.startswith('<doctest ')
            if possible_collision:
                warnings.warn(JobLibCollisionWarning(
                        'Possible name collisions between functions '
                        "'%s' (%s:%i) and '%s' (%s:%i)" %
                        (func_name, source_file, old_first_line,
                        func_name, source_file, first_line)),
                    stacklevel=stacklevel)

        # The function has changed, wipe the cache directory.
        # XXX: Should be using warnings, and giving stacklevel
        if self._verbose > 10:
            _, func_name = get_func_name(self.func, resolv_alias=False)
            self.warn("Function %s (stored in %s) has changed." %
                        (func_name, func_dir))
        self.clear(warn=True)
        return False

    def clear(self, warn=True):
        """ Empty the function's cache.
        """
        func_dir = self._get_func_dir(mkdir=False)
        if self._verbose and warn:
            self.warn("Clearing cache %s" % func_dir)
        if os.path.exists(func_dir):
            shutil.rmtree(func_dir, ignore_errors=True)
        mkdirp(func_dir)
        func_code, _, first_line = get_func_code(self.func)
        func_code_file = os.path.join(func_dir, 'func_code.py')
        self._write_func_code(func_code_file, func_code, first_line)

    def call(self, *args, **kwargs):
        """ Force the execution of the function with the given arguments and
            persist the output values.
        """
        start_time = time.time()
        output_dir, _ = self._get_output_dir(*args, **kwargs)
        if self._verbose:
            print(format_call(self.func, args, kwargs))
        output = self.func(*args, **kwargs)
        self._persist_output(output, output_dir)
        self._persist_input(output_dir, *args, **kwargs)
        duration = time.time() - start_time
        if self._verbose:
            _, name = get_func_name(self.func)
            msg = '%s - %s' % (name, format_time(duration))
            print(max(0, (80 - len(msg))) * '_' + msg)
        return output

    # Make public
    def _persist_output(self, output, dir):
        """ Persist the given output tuple in the directory.
        """
        try:
            mkdirp(dir)
            filename = os.path.join(dir, 'output.pkl')
            numpy_pickle.dump(output, filename, compress=self.compress)
            if self._verbose > 10:
                print('Persisting in %s' % dir)
        except OSError:
            " Race condition in the creation of the directory "

    def _persist_input(self, output_dir, *args, **kwargs):
        """ Save a small summary of the call using json format in the
            output directory.
        """
        argument_dict = filter_args(self.func, self.ignore,
                                    args, kwargs)

        input_repr = dict((k, repr(v)) for k, v in argument_dict.items())
        # This can fail do to race-conditions with multiple
        # concurrent joblibs removing the file or the directory
        try:
            mkdirp(output_dir)
            json.dump(
                input_repr,
                file(os.path.join(output_dir, 'input_args.json'), 'w'),
                )
        except:
            pass
        return input_repr

    def load_output(self, output_dir):
        """ Read the results of a previous calculation from the directory
            it was cached in.
        """
        if self._verbose > 1:
            t = time.time() - self.timestamp
            if self._verbose < 10:
                print('[Memory]% 16s: Loading %s...' % (
                                    format_time(t),
                                    format_signature(self.func)[0]
                                    ))
            else:
                print('[Memory]% 16s: Loading %s from %s' % (
                                    format_time(t),
                                    format_signature(self.func)[0],
                                    output_dir
                                    ))
        filename = os.path.join(output_dir, 'output.pkl')
        return numpy_pickle.load(filename,
                                 mmap_mode=self.mmap_mode)

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


###############################################################################
# class `Memory`
###############################################################################
class Memory(Logger):
    """ A context object for caching a function's return value each time it
        is called with the same input arguments.

        All values are cached on the filesystem, in a deep directory
        structure.

        see :ref:`memory_reference`
    """
    #-------------------------------------------------------------------------
    # Public interface
    #-------------------------------------------------------------------------

    def __init__(self, cachedir, mmap_mode=None, compress=False, verbose=1):
        """
            Parameters
            ----------
            cachedir: string or None
                The path of the base directory to use as a data store
                or None. If None is given, no caching is done and
                the Memory object is completely transparent.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments.
            compress: boolean
                Whether to zip the stored data on disk. Note that
                compressed arrays cannot be read by memmapping.
            verbose: int, optional
                Verbosity flag, controls the debug messages that are issued
                as functions are evaluated.
        """
        # XXX: Bad explanation of the None value of cachedir
        Logger.__init__(self)
        self._verbose = verbose
        self.mmap_mode = mmap_mode
        self.timestamp = time.time()
        self.compress = compress
        if compress and mmap_mode is not None:
            warnings.warn('Compressed results cannot be memmapped',
                          stacklevel=2)
        if cachedir is None:
            self.cachedir = None
        else:
            self.cachedir = os.path.join(cachedir, 'joblib')
            mkdirp(self.cachedir)

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
        if self.cachedir is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, MemorizedFunc):
            func = func.func
        return MemorizedFunc(func, cachedir=self.cachedir,
                                   mmap_mode=mmap_mode,
                                   ignore=ignore,
                                   compress=self.compress,
                                   verbose=verbose,
                                   timestamp=self.timestamp)

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

    def __reduce__(self):
        """ We don't store the timestamp when pickling, to avoid the hash
            depending from it.
            In addition, when unpickling, we run the __init__
        """
        # We need to remove 'joblib' from the end of cachedir
        cachedir = self.cachedir[:-7] if self.cachedir is not None else None
        return (self.__class__, (cachedir,
                self.mmap_mode, self.compress, self._verbose))
