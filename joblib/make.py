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
        return 'Ref to %s %s, id:%i, time_stamp %f' % (
            self.repr, self.desc, self.id, self.time_stamp)


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
            d = dict()
            for key, value in zip(keys, values):
                d[key] = value
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
    
    def save(self, ary):
        import numpy as np
        np.save(self._filename, ary)

    def load(self):
        import numpy as np
        filename = self._filename
        if not os.path.exists(filename):
            filename += '.npy'
        return np.load(filename)


class NiftiFile(Persister):

    def __init__(self, filename, header=None, dtype=None):
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
                    name=None):
        self._output = output
        self._debug = debug
        self._func = func
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
        # XXX This should probably be tested, using tempfiles
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
                    self.warn("Newer time stamp: %s, for %s %s"
                                    % (obj.time_stamp, obj.desc, obj.repr)
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

    def persist_output(self, new_output):
        """ Save the output values using the provided persisters.
        """
        if isinstance(self._output, Persister):
            self._output.save(new_output)
        elif type(self._output) in (t.ListType, t.TupleType):
            try:
                for out, persistence in zip(new_output, self._output):
                    persistence.save(out)
            except TypeError:
                self.warn("Persistent output not properly specified")
                if self._debug:
                    traceback.print_exc()
        elif not type(self._output) in non_mutable_types:
            self.warn("Can't persist the output of %s")

    def load_output(self):
        """ Load the output from a previous run, using the provided
            persisters.
        """
        if issubclass(self._output.__class__, Persister):
            new_output = self._output.load()
        elif type(self._output) in (t.ListType, t.TupleType):
            new_output = tuple(persistence.load() 
                        for persistence in self._output)
        else:
            new_output = self._output
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
            if self._debug:
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


def make(func=None, output=None, cachedir='cache', debug=False, name=None):
    # Some magic to use this both as a decorator and as a nicer function
    # call (is this magic evil?)
    if func is None:
        return lambda f: MakeFunctor(f, output=output, cachedir=cachedir,
                                        debug=debug, name=name)
    else:
        return MakeFunctor(func, output=output, cachedir=cachedir,
                                        debug=debug, name=name)


