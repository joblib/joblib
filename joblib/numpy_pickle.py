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
import zlib
import warnings

if sys.version_info[0] >= 3:
    from io import BytesIO
    from pickle import _Unpickler as Unpickler
    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')
else:
    try:
        from io import BytesIO
    except ImportError:
        # BytesIO has been added in Python 2.5
        from cStringIO import StringIO as BytesIO
    # XXX: create a py3k_compat module and subclass all this in it
    from pickle import Unpickler
    asbytes = str

_MEGA = 2**20
_MAX_LEN = len(hex(2**64))

# To detect file types
_ZFILE_PREFIX = asbytes('ZF')

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
# Compressed file with Zlib

def _read_magic(file_handle):
    magic = file_handle.read(len(_ZFILE_PREFIX))
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    return magic


class ZFile(BytesIO):
    """ A file-like object doing compression with Zlib, but only on
        close.

        This is faster than GZipFile, as GZipFile
        compresses/decompresses on the fly, but it may use more memory.

        Notes
        =====

        Do not forget to explicitely call close() on this object when
        open in write mode.
    """

    def __init__(self, filename, mode='r', compress=1):
        if 'r' in mode and 'w' in mode:
            raise ValueError('Cannot open ZFile in read and write '
                             'mode simultaneously')
        self.mode = mode
        self.compress = compress
        if isinstance(filename, basestring):
            file_handle = file(filename, mode)
        else:
            file_handle = filename
        self.final_file = file_handle
        if 'r' in mode:
            # Uncompress the file in a buffer
            file_handle.seek(0)
            assert _read_magic(file_handle) == _ZFILE_PREFIX, \
                "File does not have the right magic"
            length = file_handle.read(len(_ZFILE_PREFIX) + _MAX_LEN)
            length = length[len(_ZFILE_PREFIX):]
            length = int(length, 16)
            data = zlib.decompress(file_handle.read(), 15, length)
            assert len(data) == length, (
                "Incorrect data length while decompressing %s."
                "The file could be corrupted." % filename)
            BytesIO.__init__(self, data)
        elif 'w' in mode:
            # Write the header, the rest is delayed till the file is
            # closed
            file_handle.write(_ZFILE_PREFIX)
            BytesIO.__init__(self)
        else:
            raise ValueError('Invalide mode "%s"' % mode)

    def close(self):
        if 'w' in self.mode:
            # Compress and write the data
            data = self.getvalue()
            length = len(data)
            if sys.version_info[0] < 3 and type(length) is long:
                # We need to remove the trailing 'L'
                length = hex(length)[:-1]
            else:
                length = hex(length)
            # Store the length of the data
            self.final_file.write(length.ljust(_MAX_LEN))
            self.final_file.write(
                zlib.compress(data, self.compress))
        BytesIO.close(self)
        self.final_file.close()


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
            return ZFile(filename, 'w',
                         compress=self.compress)

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
            files, rather than pickling them. Of course, this is a
            total abuse of the Pickler class.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                            self.np.matrix, self.np.memmap):
            size = obj.size * obj.itemsize
            if self.compress and size < 100 * _MEGA:
                # When compressing, as we are not writing directly to the
                # disk, it is more efficient to use standard pickling
                if type(obj) is self.np.memmap:
                    # Pickling doesn't work with memmaped arrays
                    obj = self.np.asarray(obj)
                return pickle.Pickler.save(self, obj)
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
        return pickle.Pickler.save(self, obj)


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
            if not nd_array_wrapper.subclass in (self.np.ndarray,
                                                 self.np.memmap):
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

    def __init__(self, filename):
        NumpyUnpickler.__init__(self, filename,
                                mmap_mode=None)

    def _open_pickle(self):
        return self._open_npy(self._filename)

    def _open_npy(self, name):
        filename = os.path.join(self._dirname, name)
        return ZFile(filename, 'r')


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
    file_handle = open(filename, 'rb')
    if _read_magic(file_handle) == _ZFILE_PREFIX:
        if mmap_mode is not None:
            warnings.warn('file "%(filename)s" appears to be a zip, '
                    'ignoring mmap_mode "%(mmap_mode)s" flag passed'
                    % locals(), Warning, stacklevel=2)
        unpickler = ZipNumpyUnpickler(filename)
    else:
        unpickler = NumpyUnpickler(filename,
                                   file_handle=file_handle,
                                   mmap_mode=mmap_mode)

    try:
        obj = unpickler.load()
    finally:
        if hasattr(unpickler, 'file_handle'):
            unpickler.file_handle.close()
    return obj


