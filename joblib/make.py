"""
Make-like decorator for long running functions.

Provides dependency tracking, persistence of output of functions, and
lazy-reevaluation for Python functions.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


try:
    import cPickle as pickle 
except ImportError:
    import pickle
import os
import sys
import types as t
import logging
import time
import traceback
from weakref import ref

# Local imports 
from memoize import _function_code_hash

################################################################################
# Central registry to hold time stamps for objects.

class Reference(object):
    """ All the information we want to track about objects.
    """
    def __init__(self, obj, time_stamp, desc=''):
        self.type = type(obj)
        if hasattr(obj, '__class__'):
            self.obj_class = obj.__class__
        else:
            self.obj_class = None
        self.repr = repr(obj)
        self.time_stamp = time_stamp
        self.id = id(obj)
        self.desc = desc

    def __repr__(self):
        return 'Ref to %s %s, id:%i, time_stamp %s' % (
            self.repr, self.desc, self.id, 
            time.strftime("%a, %d %b %Y %H:%M:%S", 
                                    time.localtime(self.time_stamp))
            )


class ReferenceRegistry(object):
    """ A registry to keep weak references to objects, and trac
        information about them.
    """

    def __init__(self, default_time_stamp=None):
        # A table storing all the information for the objects that we
        # track, by id.
        default_ref = Reference(None, time_stamp=0)
        self.id_table = {id(None):default_ref}

        if default_time_stamp is None:
            default_time_stamp = time.time()
        self.default_time_stamp = default_time_stamp


    def register(self, obj, time_stamp=None, desc=''):
        """ Add a reference to the object in the registry.
        """
        if time_stamp is None:
            time_stamp = self.default_time_stamp
        reference =  Reference(obj, time_stamp=time_stamp, desc=desc)

        obj_id = id(obj)
        def erase_ref(weak_ref):
            """ A callback for removing the object from our tables when it
                is garbage-collected.
            """
            self.id_table.pop(obj_id)
        reference._weakref = ref(obj, erase_ref)
        self.id_table[id(obj)] = reference

    def latest_reference(self):
        """ Returns the reference with the latest time stamp.
        """
        key = lambda ref: ref.time_stamp
        return max(self.id_table.values(), key=key)

    def update(self, registry):
        """ Merge the content of another registry in this one.
        """
        self.id_table.update(registry.id_table)

    def __contains__(self, obj):
        return (id(obj) in self.id_table)

    def __getitem__(self, obj):
        return self.id_table[id(obj)]

    def __iter__(self):
        return self.id_table.__iter__()


_time_stamp_registry = ReferenceRegistry()


################################################################################
class TimeStamp(object):
    """ Placeholder for an object that we don't want to serialize. 
        
        This is only to serialize the time at which the object was
        created.
    """

    time_stamp = 0

    def __init__(self, time_stamp=None):
        if time_stamp is None:
            time_stamp = time.time()
        self.time_stamp = time_stamp

    def __eq__(self, other):
        """ Anything that we don't serialize is equal to a time_stamp.
        """
        if type(other) in (t.BooleanType, t.NoneType, t.StringType, 
                        t.UnicodeType, t.FloatType, t.IntType,
                        t.LongType, t.ComplexType, t.ListType, 
                        t.TupleType, t.DictType):
            return False
        else:
            return True



non_mutable_types = (t.BooleanType, t.NoneType, t.StringType, 
                        t.UnicodeType, t.FloatType, t.IntType,
                        t.LongType, t.ComplexType)


################################################################################
class Serializer(object):

    def __init__(self, default_time_stamp=None):
        if default_time_stamp is None:
            default_time_stamp = time.time()
        self.default_time_stamp = default_time_stamp
        self.reference_registry = ReferenceRegistry()

    def _hash_iterable(self, iterable, desc='', stored_time_stamp=None):
        output = list()
        for index, item in enumerate(iterable):
            output.append(self.hash(item, desc='%s[%i]' % (desc, index)))
        return output
    
    def hash(self, item, desc=''):
        """ Return the hash of an item.
            
            The 'desc' keyword argument refers to the human-readable
            description of the item, stored for debug information.
        """
        if type(item) in non_mutable_types:
            return item 
        elif type(item) == t.ListType:
            return self._hash_iterable(item, desc=desc)
        elif type(item) == t.TupleType:
            return tuple(self._hash_iterable(item, desc=desc))
        elif type(item) == t.DictType:
            keys = self._hash_iterable(item.keys(), desc='%s.keys()' % desc)
            values = self._hash_iterable(item.values(), 
                                    desc='%s.values()'% desc)
            return dict(zip(keys, values))
            return d
        # XXX: This is to avoid duplications with objects coming from
        # different functions having the same id. Maybe the solution is
        # to trac the history of the object.

        # Now some special cases for other non-mutable non-hashable types
        elif 'numpy' in sys.modules:
            # Trick to keep loose-coupling with numpy
            from numpy.core.numerictypes import issctype
            # numpy numeric types are also non-mutable and with an unique
            # id.
            if issctype(type(item)):
                return item
            import numpy as np
            if type(item) == np.ndarray and item.ndim ==0:
                return item
        
        if item in _time_stamp_registry:
            time_stamp = _time_stamp_registry[item].time_stamp
        else:
            time_stamp = self.default_time_stamp 
        output = TimeStamp(time_stamp)
        self.reference_registry.register(item, time_stamp=time_stamp,
                                                desc=desc)
        return output


################################################################################
# Persisters
################################################################################

class Persister(TimeStamp):
    """ Abstract class for persisting data.
    """
    # This inherits from TimeStamp because the equality rules are the
    # same as for the time stamps.

    def __init__(self, filename):
        self._filename = filename


class PickleFile(Persister):
    """ Persist the data to a file using the pickle protocole.
    """

    def save(self, data):
        pickle.dump(data, file(self._filename, 'w'))

    def load(self):
        return pickle.load(file(self._filename))


class NumpyFile(Persister):
    """ Persist the data to a file using a '.npy' or '.npz' file.
    """

    def __init__(self, filename, mmap_mode=None):
        """ mmap_mode is the mmap_mode argument to numpy.save. When
            given, memmapping is used to read the results. This can be
            much faster.
        """
        self._filename = filename
        self._mmap_mode = mmap_mode
    
    def save(self, ary):
        import numpy as np
        np.save(self._filename, ary)

    def load(self):
        import numpy as np
        filename = self._filename
        if not os.path.exists(filename):
            filename += '.npy'
        return np.load(filename, mmap_mode=self._mmap_mode)


class NiftiFile(Persister):
    """ Persists the data using a nifti file.
        
        Requires PyNifti to be installed.
    """

    def __init__(self, filename, header=None, dtype=None):
        """ header is the optional nifti header.

            dtype is a numpy dtype and is used to force the loading of
            the results with a certain dtype. Useful when the types
            understood by nifti are not complete enough for your purpose.
        """
        self._filename = filename
        self._header = header
        self._dtype = dtype

    def save(self, ary):
        import nifti
        import numpy as np
        if np.dtype(self._dtype).kind == 'b':
            ary = ary.astype(np.int8)
        nifti.NiftiImage(ary.T, self._header).save(self._filename)

    def load(self):
        import nifti
        ary = nifti.NiftiImage(self._filename).asarray().T
        if self._dtype is not None:
            ary = ary.astype(self._dtype)
        return ary


class MemMappedNiftiFile(NiftiFile):
    """ Persists the data using a memmapped nifti file.
    """

    def save(self, ary):
        import numpy as np
        if np.dtype(self._dtype).kind == 'b':
            ary = ary.astype(np.int8)
        if hasattr(self, 'nifti_image'):
            self.nifti_image.save(self._filename)
        else:
            from nifti import NiftiImage
            NiftiImage(ary.T, self._header).save(self._filename)

    def load(self):
        from nifti.niftiimage import MemMappedNiftiImage
        self.nifti_image = MemMappedNiftiImage(self._filename)
        ary = self.nifti_image.asarray().T
        if self._dtype is not None:
            ary = ary.astype(self._dtype)
        return ary


################################################################################
# The make functionnality itself
################################################################################

class MakeFunctor(object):
    """ Functor to decorate a function for lazy-reevaluation.

        Provides dependency tracking, persistence of output of functions, 
        and bookeeping of arguments and function code for 
        lazy-reevaluation for Python functions.
    """

    def __init__(self, func, output=None, cachedir='cache', debug=False,
                    raise_errors=False, name=None, force=False):
        self._output = output
        self._debug = debug
        self._func = func
        self._raise_errors = raise_errors
        self._force = force
        if name is None:
            name = func.func_name
        self._name = name
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        self._cachefile = os.path.join(cachedir,
                        '%s.%s.cache' % (func.__module__, name))

    def warn(self, msg):
        """ For warning and debug messages.
        """
        logging.warn("[make]%s (%s line %i): %s" %
            ( self._name, self._func.__module__,
              self._func.func_code.co_firstlineno, msg)
                )

    def get_call_signature(self, args, kwargs):
        """ Returned a call signature that can be serialized safely.
        """
        self._serializer = Serializer()
        out = dict()
        out['func_code'] = _function_code_hash(self._func)
        out['args'] = self._serializer.hash(args, desc='args')
        out['kwargs'] = self._serializer.hash(kwargs, desc='kwargs') 
        return self._serializer.reference_registry.latest_reference(), out
       
    def is_cache_ok(self, args, kwargs):
        """ Check to see if the cache files tells us we can use previous
            retults.
        """
        if self._force:
            self.warn('Forcing the reload of the results')
            # FIXME: I need to check if I should return True or false
            # here.
            return True
        run_func = False
        obj, call_signature = self.get_call_signature(args, kwargs)
        if os.path.exists(self._cachefile):
            cache = pickle.load(file(self._cachefile))
            if not cache['call_signature'] == call_signature:
                if self._debug:
                    self.warn("Different hashable arguments")
                run_func = True
            elif obj.time_stamp > cache['time_stamp']:
                if self._debug:
                    self.warn('obj.time_stamp %s, cache[time_stamp] %s' 
                                % (obj.time_stamp, cache['time_stamp'])) # DBG
                    self.warn("Newer time stamp: %s (function last ran %s), for %s %s"
                            % (time.strftime("%a, %d %b %Y %H:%M:%S",
                                   time.localtime(cache['time_stamp'])), 
                               time.strftime("%a, %d %b %Y %H:%M:%S",
                                   time.localtime(obj.time_stamp)), 
                               obj.desc, obj.repr)
                              )
                    if not obj.id in _time_stamp_registry.id_table:
                        logging.warn(
                            "Unknown or deleted object. "
                            "Are you sure all the inputs "
                            "of the function are either base types, or "
                            "come from the output of a `make`"
                            )
                run_func = True
        else:
            if self._debug:
                self.warn("No cache file")
            run_func = True
        return not run_func

    def _persist_output(self, new_output, persister):
        """ Given a persister, recursively explore it to persist the
            given output.
        """
        if isinstance(persister, Persister):
            persister.save(new_output)
        elif type(persister) in (t.ListType, t.TupleType):
            try:
                for out, sub_persister in zip(new_output, persister):
                    self._persist_output(out, sub_persister)
            except TypeError:
                self.warn("Persistent output not properly specified")
                if self._raise_errors:
                    raise
                elif self._debug:
                    traceback.print_exc()
        elif not type(persister) in non_mutable_types:
            self.warn("Can't persist the output")


    def persist_output(self, new_output):
        """ Save the output values using the provided persisters.
        """
        self._persist_output(new_output, self._output)

    def _load_output(self, persister):
        """ Load the output from a previous run, using the provided
            persisters.
        """
        if isinstance(persister, Persister):
            new_output = persister.load()
        elif type(persister) in (t.ListType, t.TupleType):
            new_output = tuple(self._load_output(sub_persister)
                                            for sub_persister in persister)
        else:
            new_output = persister
        return new_output

    def load_output(self):
        """ Load the output from a previous run, using the provided
            persisters.
        """
        new_output =  self._load_output(self._output)
        cache = pickle.load(file(self._cachefile))
        self.store_output_time_stamps(new_output, cache['time_stamp'])
        return new_output

    def store_output_time_stamps(self, output, time_stamp):
        """ Store the output timestamps in the central registry
        """
        serializer = Serializer(default_time_stamp=time_stamp)
        serializer.hash(output)
        _time_stamp_registry.update(serializer.reference_registry)

    def __call__(self, *args, **kwargs):
        """ Call to the function that we intercept to see if we can use
            the caching mechanism.
        """
        try:
            if self.is_cache_ok(args, kwargs):
                return self.load_output()
        except:
            self.warn('exception while loading previous results')
            if self._raise_errors:
                raise
            elif self._debug:
                traceback.print_exc()
        # We need to rerun the function:
        new_output = self._func(*args, **kwargs)
        time_stamp = time.time()
        self.persist_output(new_output)
        # Update our cache header
        _, call_signature = self.get_call_signature(args, kwargs) 
        # And store the time stamps of our outputs in the time-stamp 
        # Registry
        cache = dict(call_signature=call_signature,
                        time_stamp=time_stamp)
        pickle.dump(cache, file(self._cachefile, 'w'))
        self.store_output_time_stamps(new_output, time_stamp)
        return new_output


def make(func=None, output=None, cachedir='cache', debug=False,
         name=None, raise_errors=False, force=False):
    """ Decorate a function for lazy re-evaluation.

        Parameters
        -----------
        func : a callable, optional
            If func is given, the function is returned decorated.
            Elsewhere, the call to 'make' returns a decorator object that
            can be applied to a function.
        output : persisters, optional
            output can be a persistence objects, or a list of persistence
            objects. This argument describes the mapping to the disk used
            to save the output of the decorated function.
        cachedir : string, optional
            Name of the directory used to store the function calls cache 
            information.
        debug : boolean, optional
            If debug is true, joblib produces a verbose output that can
            be useful to understand why memoized functions are being 
            re-evaluated.
        name : string, optional
            Identifier for the function used in the cache. If none is
            given, the function name is used. Changing the default
            value of this identifier is usefull when you want to call
            the function several times with different arguments and 
            store in different caches.
        force : boolean, optional
            If force is true, make tries to reload the results, even if
            the input arguments have changed. This is useful to avoid 
            long recalculation when minor details up the pipeline
            changed.

        Returns
        ---------
        The decorated function, if func is given. A decorator object
        elsewhere.
    """
    # Some magic to use this both as a decorator and as a nicer function
    # call (is this magic evil?)
    if func is None:
        return lambda f: MakeFunctor(f, output=output, cachedir=cachedir,
                                        debug=debug, name=name, 
                                        force=force,
                                        raise_errors=raise_errors)
    else:
        return MakeFunctor(func, output=output, cachedir=cachedir,
                                        debug=debug, name=name,
                                        force=force,
                                        raise_errors=raise_errors)


