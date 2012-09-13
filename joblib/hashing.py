"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import hashlib
import sys
import types
import struct

if sys.version_info[0] == 3:
    # in python3, StringIO does not accept binary data
    # see http://packages.python.org/six/
    import io
    StringIO = io.BytesIO
else:
    import cStringIO
    StringIO = cStringIO.StringIO


class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """
    def __init__(self, set_sequence):
        self._sequence = sorted(set_sequence)


class Hasher(pickle.Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='md5'):
        self.stream = StringIO()
        pickle.Pickler.__init__(self, self.stream, protocol=2)
        # Initialise the hash obj
        self._hash = hashlib.new(hash_name)

    def hash(self, obj, return_digest=True):
        self.dump(obj)
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()

    def save(self, obj):
        if isinstance(obj, types.MethodType):
            # the Pickler cannot pickle instance methods; here we decompose
            # them into components that make them uniquely identifiable
            func_name = obj.im_func.__name__
            inst = obj.im_self
            cls = obj.im_class
            obj = (func_name, inst, cls)
        pickle.Pickler.save(self, obj)

    if sys.version_info[0] < 3:
        # The dispatch table of the pickler is not accessible in Python
        # 3, as these lines are only bugware for IPython, we skip them.
        def save_global(self, obj, name=None, pack=struct.pack):
            # We have to override this method in order to deal with objects
            # defined interactively in IPython that are not injected in
            # __main__
            module = getattr(obj, "__module__", None)
            if module == '__main__':
                my_name = name
                if my_name is None:
                    my_name = obj.__name__
                mod = sys.modules[module]
                if not hasattr(mod, my_name):
                    # IPython doesn't inject the variables define
                    # interactively in __main__
                    setattr(mod, my_name, obj)
            pickle.Pickler.save_global(self, obj, name=name, pack=struct.pack)

        dispatch = pickle.Pickler.dispatch.copy()
        # builtin
        dispatch[type(len)] = save_global
        # type
        dispatch[type(object)] = save_global
        # classobj
        dispatch[type(pickle.Pickler)] = save_global
        # function
        dispatch[type(pickle.dump)] = save_global

    def _batch_setitems(self, items):
        # forces order of keys in dict to ensure consistent hash
        pickle.Pickler._batch_setitems(self, iter(sorted(items)))

    def save_set(self, set_items):
        # forces order of items in Set to ensure consistent hash
        pickle.Pickler.save_inst(self, _ConsistentSet(set_items))

    # set
    dispatch[type(set())] = save_set


class NumpyHasher(Hasher):
    """ Special case the hasher for when numpy is loaded.
    """

    def __init__(self, hash_name='md5', coerce_mmap=False):
        """
            Parameters
            ----------
            hash_name: string
                The hash algorithm to be used
            coerce_mmap: boolean
                Make no difference between np.memmap and np.ndarray
                objects.
        """
        self.coerce_mmap = coerce_mmap
        Hasher.__init__(self, hash_name=hash_name)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np
        self.np = np

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
            # Compute a hash of the object:
            try:
                self._hash.update(self.np.getbuffer(obj))
            except TypeError:
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                self._hash.update(self.np.getbuffer(obj.flatten()))

            # We store the class, to be able to distinguish between
            # Objects with the same binary content, but different
            # classes.
            if self.coerce_mmap and isinstance(obj, self.np.memmap):
                # We don't make the difference between memmap and
                # normal ndarrays, to be able to reload previously
                # computed results with memmap.
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            # We also return the dtype and the shape, to distinguish
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = (klass, ('HASHED', obj.dtype, obj.shape, obj.strides))
        Hasher.save(self, obj)


def hash(obj, hash_name='md5', coerce_mmap=False):
    """ Quick calculation of a hash to identify uniquely Python objects
        containing numpy arrays.


        Parameters
        -----------
        hash_name: 'md5' or 'sha1'
            Hashing algorithm used. sha1 is supposedly safer, but md5 is
            faster.
        coerce_mmap: boolean
            Make no difference between np.memmap and np.ndarray
    """
    if 'numpy' in sys.modules:
        hasher = NumpyHasher(hash_name=hash_name, coerce_mmap=coerce_mmap)
    else:
        hasher = Hasher(hash_name=hash_name)
    return hasher.hash(obj)
