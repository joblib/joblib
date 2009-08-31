"""
A pickler to save numpy arrays in separate .npy files.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import traceback
import os

################################################################################
# Utility objects for persistence.

class NDArrayWrapper:
    """ An object to be persisted instead of numpy arrays.

        The only thing this object does, is store the filename in wich
        the array has been persisted.

        We use an old-style class, because it is easy to override the
        unpickler with old-style classes. This is a hack, of course.
    """
    def __init__(self, filename):
        self.filename = filename


################################################################################
# Pickler classes

class NumpyPickler(pickle.Pickler):
    # XXX: Docstring!

    def __init__(self, filename):
        self._filename = filename
        self._filenames = [filename, ]
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
            # XXX: The way to do this would be to replace with an object
            # that has a __reduce_ex__ that does the work.
            self._npy_counter += 1
            try:
                filename = '%s_%02i.npy' % (self._filename,
                                            self._npy_counter )
                self._filenames.append(filename)
                self.np.save(filename, obj)
                obj = NDArrayWrapper(os.path.basename(filename))
            except:
                self._npy_counter -= 1
                # XXX: We should have a logging mechanism
                traceback.print_exc()
        pickle.Pickler.save(self, obj)



class NumpyUnpickler(pickle.Unpickler):
    # XXX: Docstring!
    dispatch = pickle.Unpickler.dispatch.copy()

    def __init__(self, filename):
        self._filename = filename
        self._dirname  = os.path.dirname(filename)
        self.file = open(filename, 'rb')
        pickle.Unpickler.__init__(self, self.file)
        import numpy as np
        self.np = np

    def load_build(self):
        pickle.Unpickler.load_build(self)
        if isinstance(self.stack[-1], NDArrayWrapper):
            nd_array_wrapper = self.stack.pop()
            array = self.np.load(os.path.join(self._dirname,
                                                nd_array_wrapper.filename))
            self.stack.append(array)


    # Be careful to register our new method.
    dispatch[pickle.BUILD] = load_build


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
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            pickler.file.flush()
            pickler.file.close()
    return pickler._filenames


def load(filename):
    """ Load the pickled objects from the given file.
    """
    try:
        unpickler = NumpyUnpickler(filename)
        obj = unpickler.load()
    finally:
        if 'unpickler' in locals() and hasattr(unpickler, 'file'):
            unpickler.file.close()
    return obj

