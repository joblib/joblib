"""Collection of decorators that caches a function's return value each time they
are called.

If called later with the same arguments, the cached value is returned, and
not re-evaluated. Slow for mutable types.

Taken from U{http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/466320}.

"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


import os
import warnings
from shelve import open as sopen
from cPickle import dumps, PicklingError
import inspect
import itertools
import traceback
import logging
from bsddb import _bsddb

def _function_code_hash(func):
    """ Attempts to retrieve a reliable function code hash.
    
        The reason we don't use inspect.getsource is that it caches the
        source, whereas we want this to be modified on the fly when the
        function is modified.
    """
    try:
        # Try to retrieve the source code.
        source_file = file(func.func_code.co_filename)
        first_line = func.func_code.co_firstlineno
        # All the lines after the function definition:
        source_lines = list(itertools.islice(source_file, first_line-1, None))
        return ''.join(inspect.getblock(source_lines))
    except:
        # If the source code fails, we use the hash. This is fragile and
        # might change from one session to another.
        return func.func_code.__hash__()


################################################################################
class MemoizeFunctor(object):
    """ Functor to decorate a function, caching its return value each time
    it is called. If called later with the same arguments, the cached
    value is returned, and not re-evaluated. Slow for mutable types.

    Cache is a file, hence memory is not lost even computer is shut down! :-)
    The file is syncronized anytime a value is added to it.
    
    Note 
    ----

    Taken from U{http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/466320}.

    """

    __slots__ = ['func', '_cache_filename', '_cache', '_cachedir', '_persist',
                    '_debug']
    
    def __init__(self, func, cachedir='cache', persist=True, debug=False):
        self.func = func
        self._persist = persist
        self._debug = debug
        if persist is False:
            if not cachedir=='cache':
                warnings.warn('cachedir specified, but persist is False;'
                             'not persisting', stacklevel=2)
            self._cache = dict()
        else:
            self._cachedir = cachedir
            if not os.path.exists(self._cachedir):
                os.makedirs(self._cachedir)
            cache_filename = os.path.join(self._cachedir, 
                    '%s.%s.shelve' % (func.__module__, func.func_name))
            cache = self._cache = sopen(cache_filename, 'c', writeback=True)
            self._cache_filename = cache_filename
            # Compare the function code with the previous to see if the
            # function code has changed

            # I cannot use inspect.getsource because it is not
            # reliable when using IPython's magic "%run".
            func_code = _function_code_hash(func)
            try:
                if not ('__func_code__' in cache  and
                            cache['__func_code__'] == func_code):
                    self._cache_clear()
            except _bsddb.DBRunRecoveryError:
                self.warn('Unrecoverable DB error, clearing DB')
                self._cache = None
                os.unlink(self._cache_filename)
                self._cache = sopen(self._cache_filename, 'c')


    def _cache_clear(self):
        if self._debug:
            self.warn("Clearing cache")
        try:
            self._cache.clear()
        except KeyError:
            "DB not found: the db has probably not yet been created."
        self._cache['__func_code__'] = _function_code_hash(self.func)

    def warn(self, msg):
        logging.warn("[memoize]%s (%s line %i): %s" %
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
        self._cache.clear()
        if self._persist:
            self._cache.sync()


def memoize(persist=True, cachedir='cache', debug=False):
    """ Decorator to cache the return values of a function each time
    it is called. If called later with the same arguments, the cached
    value is returned, and not re-evaluated. If the function code is modified, 
    the cache is flushed. 

    If persist is True, the cache is a file, hence memory is not lost even
    computer is shut down! :-) The file is syncronized anytime a value is 
    added to it. This is slower, especially if mutable types are used.

    If debug is true, the different calls to the function are displayed
    when the function is actually called.
    """
    return lambda func: MemoizeFunctor(func, 
                                    persist=persist, cachedir=cachedir,
                                    debug=debug)

