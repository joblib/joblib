"""
Utilities for hashing input arguments of functions.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import itertools
import inspect
import pickle
import types
import hashlib
import sys
import cStringIO

NON_MUTABLE_TYPES = (types.BooleanType, types.NoneType, types.StringType,
                     types.UnicodeType, types.FloatType, types.IntType,
                     types.LongType, types.ComplexType)


################################################################################
# Functions for non-human readable hashing

class Hasher(pickle.Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='sh1'):
        self.stream = cStringIO.StringIO()
        pickle.Pickler(self, self.stream, protocol=None)
        # Initialise the hash obj
        self.hash = hashlib.new(hash_name)

    def hash(self, obj, return_digest=True):
        self.dump(obj)
        self.hash.update(self.stream.getvalue())
        if return_digest:
            return self.hash.hexdigest()


class NumpyHasher(Hasher):
    """ Special case the haser for when numpy is loaded.
    """

    def __init__(self, hash_name='sh1'):
        Hasher.__init__(self, hash_name='sh1')
        # delayed import of numpy, to avoid tight coupling
        import numpy as np
        self.np = np

    def save(self, obj):
        """ Subclass the save method, to hash ndarray subclass, rather
            than pickling them. Off course, this is a total abuse of
            the Pickler class.
        """
        if isinstance(obj, self.np.ndarray):
            # Compute a hash of the object:
            self.hash.update(obj)

            # We store the class, to be able to distinguish between 
            # Objects with the same binary content, but different
            # classes.
            if isinstance(obj, self.np.memmap):
                # XXX: I should add a keyword argument to make this 
                # optional.
                # We don't make the difference between memmap and
                # normal ndarrays, to be able to reload previously
                # computed results with memmap.
                klass = self.np.ndarray
            else:
                klass = obj.__class__
            # We also return the dtype and the shape, to distinguish 
            # different views on the same data with different dtypes.

            # The object will be pickled by the pickler hashed at the end.
            obj = (klass, ('HASHED', obj.dtype, obj.shape))


def hash(obj, hash_name='sha1'):
    if 'numpy' in sys.modules:
        hasher = NumpyHasher(hash_name='sh1')
    else:
        hasher = Hasher(hash_name='sh1')
    return hasher.hash(obj)


################################################################################
# Function-specific hashing
def function_code_hash(func):
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


def get_arg_hash(self, args, kwargs):
    """ Return a readable unique argument hash for the given function.
    """
    # First replace types that we want to hash by their hash
    if len(args) > 4:
        # TODO
        raise NotImplementedError
    output = '_'.join(args[:4])
        

