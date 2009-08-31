"""
A pickler to save numpy arrays in .npy files.
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

class NumpyHasher(pickle.Pickler):
    """ Special case the haser for when numpy is loaded.
    """

    def __init__(self, filename):
        self._filename = filename
        self.file = open(filename, 'w')
        # Count the number of npy files that we have created:
        self._npy_counter = 0
        pickle.Pickler.__init__(self, self.file, protocol=2)
        # delayed import of numpy, to avoid tight coupling
        import numpy as np
        self.np = np

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
            files, rather than pickling them. Off course, this is a 
            total abuse of the Pickler class.
        """
        if isinstance(obj, self.np.ndarray):
            self._npy_counter += 1
            try:
                filename = '%s_%02i.npy' % (self._filename,
                                            self._npy_counter )
                obj = (klass, ('HASHED', obj.dtype, obj.shape))
            except:
                self._npy_counter -= 1
        Hasher.save(self, obj)


################################################################################
# Utility functions

def dump(value, filename):
    """Pickles the object (`value`) into the passed filename.

    XXX: Add info on the npy files created.
    """
    try:
        pickler = NumpyPickler(filename)
        pickler.dump(value)
    finally:
        if hasattr(pickler, 'file'):
            pickler.file.flush()
            pickler.file.close()



