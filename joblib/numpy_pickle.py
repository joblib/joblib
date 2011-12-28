"""
A pickler to save numpy arrays in separate .npy files.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import traceback
import sys
import os
import gzip
import warnings

if sys.version_info[0] == 3:
    from pickle import _Unpickler as Unpickler
    from cStringIO import StringIO as BytesIO
else:
    from io import BytesIO
    from pickle import Unpickler


###############################################################################
# Utility objects for persistence.


class NDArrayWrapper(object):
    """ An object to be persisted instead of numpy arrays.

        The only thing this object does, is store the filename in wich
        the array has been persisted.
    """
    def __init__(self, filename, subclass=None):
        self.filename = filename
        self.subclass = subclass


###############################################################################
# Pickler classes

class NumpyPickler(pickle.Pickler):
    """ A pickler subclass that extracts ndarrays and saves them in .npy
        files outside of the pickle.
    """

    def __init__(self, filename, compress=0):
        self._filename = filename
        self._filenames = [filename, ]
        self.compress = compress
        self.file = self._open(filename)
        # Count the number of npy files that we have created:
        self._npy_counter = 0
        pickle.Pickler.__init__(self, self.file,
                                protocol=pickle.HIGHEST_PROTOCOL)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _open(self, filename):
        if not self.compress:
            return open(filename, 'wb')
        else:
            return gzip.open(filename, 'wb',
                             compresslevel=self.compress)

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
            files, rather than pickling them. Of course, this is a
            total abuse of the Pickler class.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                                 self.np.matrix, self.np.memmap):
            self._npy_counter += 1
            try:
                filename = '%s_%02i.npy' % (self._filename,
                                            self._npy_counter)
                if self.compress:
                    filename += '.gz'
                file_handle = self._open(filename)
                self.np.save(file_handle, obj)
                file_handle.close()
                self._filenames.append(filename)
                obj = NDArrayWrapper(os.path.basename(filename),
                                     type(obj))
            except:
                self._npy_counter -= 1
                # XXX: We should have a logging mechanism
                print 'Failed to save %s to .npy file:\n%s' % (
                        type(obj),
                        traceback.format_exc())
        pickle.Pickler.save(self, obj)


class ZipNumpyPickler(NumpyPickler):
    """ A subclass of the Pickler, to zip as we go.
    """

class NumpyUnpickler(Unpickler):
    """ A subclass of the Unpickler to unpickle our numpy pickles.
    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle=None, mmap_mode=None):
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        if file_handle is None:
            file_handle = self._open_pickle()
        self.file_handle = file_handle
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _open_pickle(self):
        return open(os.path.join(self._dirname, self._filename), 'rb')

    def _open_npy(self, name):
        return os.path.join(self._dirname, name)

    def load_build(self):
        """ This method is called to set the state of a newly created
            object.

            We capture it to replace our place-holder objects,
            NDArrayWrapper, by the array we are interested in. We
            replace directly in the stack of pickler.
        """
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], NDArrayWrapper):
            if self.np is None:
                raise ImportError('Trying to unpickle an ndarray, '
                        "but numpy didn't import correctly")
            nd_array_wrapper = self.stack.pop()
            obj = self._open_npy(nd_array_wrapper.filename)
            if not isinstance(obj, basestring):
                # Must be a file-like object
                # read_array can handle them
                array = self.np.lib.format.read_array(obj)
            elif self.np.__version__ >= '1.3':
                array = self.np.load(obj,
                                mmap_mode=self.mmap_mode)
            else:
                # Numpy does not have mmap_mode before 1.3
                array = self.np.load(obj)
            if (not nd_array_wrapper.subclass is self.np.ndarray
                    and not nd_array_wrapper.subclass is self.np.memmap):
                # We need to reconstruct another subclass
                new_array = self.np.core.multiarray._reconstruct(
                        nd_array_wrapper.subclass, (0,), 'b')
                new_array.__array_prepare__(array)
                array = new_array
            self.stack.append(array)

    # Be careful to register our new method.
    dispatch[pickle.BUILD] = load_build


class ZipNumpyUnpickler(NumpyUnpickler):
    """ A subclass of our Unpickler to unpickle on the fly from zips.
    """

    def __init__(self, filename, file_handle=None):
        NumpyUnpickler.__init__(self, filename,
                                mmap_mode=None)

    def _open_pickle(self):
        return gzip.open(os.path.join(self._dirname, self._filename), 'rb')

    def _open_npy(self, name):
        return gzip.open(os.path.join(self._dirname, name), 'rb')


###############################################################################
# Utility functions

def dump(value, filename, compress=0):
    """ Persist an arbitrary Python object into a filename, with numpy arrays
        saved as separate .npy files.

        Parameters
        -----------
        value: any Python object
            The object to store to disk
        filename: string
            The name of the file in which it is to be stored
        compress: boolean, optional
            Whether to compress the data on the disk or not

        Returns
        -------
        filenames: list of strings
            The list of file names in which the data is stored. If
            compress is false, each array is stored in a different file.

        See Also
        --------
        joblib.load : corresponding loader

        Notes
        -----
        compressed files take extra disk space during the dump, and extra
        memory during the loading.
    """
    try:
        pickler = NumpyPickler(filename, compress=compress)
        pickler.dump(value)
    finally:
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            pickler.file.flush()
            pickler.file.close()
    return pickler._filenames


def load(filename, mmap_mode=None):
    """ Reconstruct a Python object and the numpy arrays it contains from
        a persisted file.

        Parameters
        -----------
        filename: string
            The name of the file from which to load the object
        mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
            If not None, the arrays are memory-mapped from the disk. This
            mode has not effect for compressed files. Note that in this
            case the reconstructed object might not longer match exactly
            the originally pickled object.

        Returns
        -------
        result: any Python object
            The object stored in the file.

        See Also
        --------
        joblib.dump : function to save an object

        Notes
        -----

        This function loads the numpy array files saved separately. If
        the mmap_mode argument is given, it is passed to np.save and
        arrays are loaded as memmaps. As a consequence, the reconstructed
        object might not match the original pickled object.

    """
    # Code to detect zip files
    _GZIP_PREFIX = '\x1f\x8b'
    try:
        # Py3k compatibility
        from numpy.compat import asbytes
        _GZIP_PREFIX = asbytes(_GZIP_PREFIX)
    except ImportError:
        pass

    file_handle = open(filename, 'rb')
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    if file_handle.read(len(_GZIP_PREFIX)) == _GZIP_PREFIX:
        if mmap_mode is not None:
            warnings.warn('file "%(filename)s" appears to be a zip, '
                    'ignoring mmap_mode "%(mmap_mode)s" flag passed'
                    % locals(),
                    Warning, stacklevel=2)
        unpickler = ZipNumpyUnpickler(filename)
    else:
        unpickler = NumpyUnpickler(filename,
                                   file_handle=file_handle,
                                   mmap_mode=mmap_mode)

    try:
        obj = unpickler.load()
    finally:
        if 'unpickler' in locals():
            if hasattr(unpickler, 'file'):
                unpickler.file.close()
            if hasattr(unpickler, '_zip_file'):
                unpickler._zip_file.close()
    return obj


