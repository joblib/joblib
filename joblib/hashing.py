"""
Fast cryptographic hash of Python objects, with a special case for fast
hashing of numpy arrays.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import hashlib
import io
import sys
import types
import warnings

from ._compat import _bytes_or_unicode
from .externals import dill

PY3 = sys.version[0] == '3'
Pickler = dill.Pickler


class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """
    def __init__(self, set_sequence):
        self._sequence = sorted(set_sequence)


class _MyHash(object):
    """ Class used to hash objects that won't normally pickle """

    def __init__(self, *args):
        self.args = args


class Hasher(Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='md5'):
        self.stream = io.BytesIO()
        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        protocol = (dill.DEFAULT_PROTOCOL if PY3
                    else dill.HIGHEST_PROTOCOL)
        Pickler.__init__(self, self.stream, protocol=protocol)
        # Initialise the hash obj
        self._hash = hashlib.new(hash_name)

    def hash(self, obj, return_digest=True):
        try:
            self.dump(obj)
        except dill.PicklingError as e:
            e.args += ('PicklingError while hashing %r: %r' % (obj, e),)
            raise
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()

    def save(self, obj):
        if isinstance(obj, (types.MethodType, type({}.pop))):
            # the Pickler cannot pickle instance methods; here we decompose
            # them into components that make them uniquely identifiable
            if hasattr(obj, '__func__'):
                func_name = obj.__func__.__name__
            else:
                func_name = obj.__name__
            inst = obj.__self__
            if type(inst) == type(dill):
                obj = _MyHash(func_name, inst.__name__)
            elif inst is None:
                # type(None) or type(module) do not pickle
                obj = _MyHash(func_name, inst)
            else:
                cls = obj.__self__.__class__
                obj = _MyHash(func_name, inst, cls)
        Pickler.save(self, obj)

    def memoize(self, obj):
        # We want hashing to be sensitive to value instead of reference.
        # For example we want ['aa', 'aa'] and ['aa', 'aaZ'[:2]]
        # to hash to the same value and that's why we disable memoization
        # for strings
        if isinstance(obj, _bytes_or_unicode):
            return
        Pickler.memoize(self, obj)

    dispatch = Pickler.dispatch.copy()

    def _batch_setitems(self, items):
        # forces order of keys in dict to ensure consistent hash
        Pickler._batch_setitems(self, iter(sorted(items)))

    def save_set(self, set_items):
        # forces order of items in Set to ensure consistent hash
        Pickler.save(self, _ConsistentSet(set_items))

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
        if hasattr(np, 'getbuffer'):
            self._getbuffer = np.getbuffer
        else:
            self._getbuffer = memoryview

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, self.np.ndarray) and not obj.dtype.hasobject:
            # Compute a hash of the object:
            try:
                # memoryview is not supported for some dtypes,
                # e.g. datetime64, see
                # https://github.com/numpy/numpy/issues/4983.  The
                # workaround is to view the array as bytes before
                # taking the memoryview
                obj_bytes_view = obj.view(self.np.uint8)
                self._hash.update(self._getbuffer(obj_bytes_view))
            # ValueError is raised by .view when the array is not contiguous
            # BufferError is raised by Python 3 in the hash update if
            # the array is Fortran rather than C contiguous
            except (ValueError, BufferError):
                # Cater for non-single-segment arrays: this creates a
                # copy, and thus aleviates this issue.
                # XXX: There might be a more efficient way of doing this
                obj_bytes_view = obj.flatten().view(self.np.uint8)
                self._hash.update(self._getbuffer(obj_bytes_view))

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
        elif isinstance(obj, self.np.dtype):
            # Atomic dtype objects are interned by their default constructor:
            # np.dtype('f8') is np.dtype('f8')
            # This interning is not maintained by a
            # pickle.loads + pickle.dumps cycle, because __reduce__
            # uses copy=True in the dtype constructor. This
            # non-deterministic behavior causes the internal memoizer
            # of the hasher to generate different hash values
            # depending on the history of the dtype object.
            # To prevent the hash from being sensitive to this, we use
            # .descr which is a full (and never interned) description of
            # the array dtype according to the numpy doc.
            klass = obj.__class__
            obj = (klass, ('HASHED', obj.descr))
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
