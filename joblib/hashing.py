"""
Utilities for hashing input arguments of functions.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import itertools
import inspect
import pickle
import hashlib
import sys
import cStringIO


################################################################################
# Functions for non-human readable hashing

class Hasher(pickle.Pickler):
    """ A subclass of pickler, to do cryptographic hashing, rather than
        pickling.
    """

    def __init__(self, hash_name='md5'):
        self.stream = cStringIO.StringIO()
        pickle.Pickler.__init__(self, self.stream, protocol=2)
        # Initialise the hash obj
        self._hash = hashlib.new(hash_name)

    def hash(self, obj, return_digest=True):
        self.dump(obj)
        dumps = self.stream.getvalue()
        self._hash.update(dumps)
        if return_digest:
            return self._hash.hexdigest()


class NumpyHasher(Hasher):
    """ Special case the haser for when numpy is loaded.
    """

    def __init__(self, hash_name='md5'):
        Hasher.__init__(self, hash_name=hash_name)
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
            self._hash.update(obj)

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
        Hasher.save(self, obj)


def hash(obj, hash_name='md5'):
    """
        Parameters
        -----------
        hash_name: 'md5' or 'sha1'
            sha1 is supposedly safer, but md5 is faster.
    """
    if 'numpy' in sys.modules:
        hasher = NumpyHasher(hash_name=hash_name)
    else:
        hasher = Hasher(hash_name=hash_name)
    return hasher.hash(obj)


################################################################################
# Function-specific hashing
def get_func_code(func):
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
        return str(func.func_code.__hash__())


def get_func_name(func):       
    """ Return the function import path (as a list of module names), and
        a name for the function.
    """
    module = func.__module__
    module = module.split('.')
    name = func.func_name
    # Hack to detect functions not defined at the module-level
    if name in func.func_globals:
        if not func.func_globals[name] is func:
            name = '%s-local' % name
    return module, name

