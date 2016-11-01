"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.
import contextlib
import pickle
import os
import sys
import threading
import warnings
from functools import partial
from os.path import split, splitext, dirname, abspath, join

import IPython

try:
    from pathlib import Path
except ImportError:
    Path = None

from .numpy_pickle_utils import _COMPRESSORS
from .numpy_pickle_utils import BinaryZlibFile
from .numpy_pickle_utils import Unpickler, Pickler
from .numpy_pickle_utils import _read_fileobject, _write_fileobject
from .numpy_pickle_utils import _read_bytes, BUFFER_SIZE
from .numpy_pickle_compat import load_compatibility
from .numpy_pickle_compat import NDArrayWrapper
# For compatibility with old versions of joblib, we need ZNDArrayWrapper
# to be visible in the current namespace.
# Explicitly skipping next line from flake8 as it triggers an F401 warning
# which we don't care.
from .numpy_pickle_compat import ZNDArrayWrapper  # noqa
from ._compat import _basestring, PY3_OR_LATER

if PY3_OR_LATER:
    import pickle as fast_pickle
else:
    try:
        import cPickle as fast_pickle
    except ImportError:
        warnings.warn("Can't use cpickle using default pickle. "
                      "This could lead to performance problems.")
        import pickle as fast_pickle

storage = threading.local()

###############################################################################
# Utility objects for persistence.


def read_np_dump(new_args, new_kwargs):
    directory = dirname(abspath(storage.filename))
    mmap_mode = storage.mmap_mode
    file_opener = storage.file_opener
    return NumpyArrayWrapper(*new_args, **new_kwargs).read(directory, mmap_mode, file_opener=file_opener)


class NumpyArrayWrapper(object):
    """An object to be persisted instead of numpy arrays.

    This object is used to hack into the pickle machinery and read numpy
    array data from our custom persistence format.
    More precisely, this object is used for:
    * carrying the information of the persisted array: subclass, shape, order,
    dtype. Those ndarray metadata are used to correctly reconstruct the array
    with low level numpy functions.
    * determining if memmap is allowed on the array.
    * reading the array bytes from a file.
    * reading the array using memorymap from a file.
    * writing the array bytes to a file.

    Attributes
    ----------
    subclass: numpy.ndarray subclass
        Determine the subclass of the wrapped array.
    shape: numpy.ndarray shape
        Determine the shape of the wrapped array.
    order: {'C', 'F'}
        Determine the order of wrapped array data. 'C' is for C order, 'F' is
        for fortran order.
    dtype: numpy.ndarray dtype
        Determine the data type of the wrapped array.
    allow_mmap: bool
        Determine if memory mapping is allowed on the wrapped array.
        Default: False.
    """

    def __init__(self, subclass, shape, order, dtype, filename, allow_mmap=False):
        """Constructor. Store the useful information for later."""
        self.subclass = subclass
        self.shape = shape
        self.order = order
        self.dtype = dtype
        self.allow_mmap = allow_mmap
        _, self.filename = split(filename)

        try:
            import numpy
            self.np = numpy
        except ImportError:
            self.np = None

    def __reduce__(self):
        return (read_np_dump, (
            (
                self.subclass,
                self.shape,
                self.order,
                self.dtype,
            ),
            {
                'allow_mmap': self.allow_mmap,
                'filename': self.filename
            }
        ),)


    # def __getnewargs__(self):
    #     return (
    #         (
    #             self.subclass,
    #             self.shape,
    #             self.order,
    #             self.dtype,
    #         ),
    #         {
    #             'allow_mmap': self.allow_mmap,
    #             'filename': self.filename
    #         }
    #     )


    def write_array(self, array, directory, file_opener=partial(open, mode='bw')):
        """Write array bytes to pickler file handle.

        This function is an adaptation of the numpy write_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        # Set buffer size to 16 MiB to hide the Python loop overhead.

        with file_opener(os.path.join(directory, self.filename)) as file_handle:
            buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)
            if array.dtype.hasobject:
                # We contain Python objects so we cannot write out the data
                # directly. Instead, we will pickle it out with version 2 of the
                # pickle protocol.
                fast_pickle.dump(array, file_handle, protocol=2)
            else:
                for chunk in self.np.nditer(array,
                                               flags=['external_loop',
                                                      'buffered',
                                                      'zerosize_ok'],
                                               buffersize=buffersize,
                                               order=self.order):
                    file_handle.write(chunk.tostring('C'))

    def read_array(self, directory, file_opener=partial(open, mode='br')):
        """Read array from unpickler file handle.

        This function is an adaptation of the numpy read_array function
        available in version 1.10.1 in numpy/lib/format.py.
        """
        if len(self.shape) == 0:
            count = 1
        else:
            count = self.np.multiply.reduce(self.shape)
        # Now read the actual data.

        with file_opener(join(directory, self.filename)) as file_handle:
            if self.dtype.hasobject:
                # The array contained Python objects. We need to unpickle the data.
                array = fast_pickle.load(file_handle)
            else:
                if (not PY3_OR_LATER and
                        self.np.compat.isfileobj(file_handle)):
                    # In python 2, gzip.GzipFile is considered as a file so one
                    # can use numpy.fromfile().
                    # For file objects, use np.fromfile function.
                    # This function is faster than the memory-intensive
                    # method below.
                    array = self.np.fromfile(file_handle,
                                                  dtype=self.dtype, count=count)
                else:
                    # This is not a real file. We have to read it the
                    # memory-intensive way.
                    # crc32 module fails on reads greater than 2 ** 32 bytes,
                    # breaking large reads from gzip streams. Chunk reads to
                    # BUFFER_SIZE bytes to avoid issue and reduce memory overhead
                    # of the read. In non-chunked case count < max_read_count, so
                    # only one read is performed.
                    max_read_count = BUFFER_SIZE // min(BUFFER_SIZE,
                                                        self.dtype.itemsize)

                    array = self.np.empty(count, dtype=self.dtype)
                    for i in range(0, count, max_read_count):
                        read_count = min(max_read_count, count - i)
                        read_size = int(read_count * self.dtype.itemsize)
                        data = _read_bytes(file_handle,
                                           read_size, "array data")
                        array[i:i + read_count] = \
                            self.np.frombuffer(data, dtype=self.dtype,
                                                    count=read_count)
                        del data
                try:
                    if self.order == 'F':
                        array.shape = self.shape[::-1]
                        array = array.transpose()
                    else:
                        array.shape = self.shape
                except Exception:
                    IPython.embed()

        return array

    def read_mmap(self, directory, mmap_mode):
        """Read an array using numpy memmap."""
        full_filename = join(directory, self.filename)
        if mmap_mode == 'w+':
            mmap_mode = 'r+'

        marray = self.np.memmap(full_filename,
                                 dtype=self.dtype,
                                 shape=self.shape,
                                 order=self.order,
                                 mode=mmap_mode,)

        return marray

    def read(self, directory, mmap_mode, file_opener=partial(open, mode='rb')):
        """Read the array corresponding to this wrapper.

        Use the unpickler to get all information to correctly read the array.

        Parameters
        ----------
        unpickler: NumpyUnpickler

        Returns
        -------
        array: numpy.ndarray

        """
        # When requested, only use memmap mode if allowed.
        if mmap_mode is not None and self.allow_mmap:
            array = self.read_mmap(directory, mmap_mode)
        else:
            array = self.read_array(directory, file_opener)

        # Manage array subclass case
        if (hasattr(array, '__array_prepare__') and
            self.subclass not in (self.np.ndarray,
                                  self.np.memmap)):
            # We need to reconstruct another subclass
            new_array = self.np.core.multiarray._reconstruct(
                self.subclass, (0,), 'b')
            return new_array.__array_prepare__(array)
        else:
            return array

###############################################################################
# Pickler classes


class NumpyPicklerCompatible(Pickler):
    """A pickler to persist big data efficiently.

    The main features of this object are:
    * persistence of numpy arrays in a single file.
    * optional compression with a special care on avoiding memory copies.

    Attributes
    ----------
    fp: file
        File object handle used for serializing the input object.
    protocol: int
        Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL under
        python 3, pickle.HIGHEST_PROTOCOL otherwise.
    """

    dispatch = Pickler.dispatch.copy()

    def __init__(self, filename, file_handle, file_opener, protocol=None):
        self.filename = filename
        self.file_opener = file_opener

        self.buffered = isinstance(file_handle, BinaryZlibFile)
        self.array_count = 0

        # By default we want a pickle protocol that only changes with
        # the major python version and not the minor one
        if protocol is None:
            protocol = (pickle.DEFAULT_PROTOCOL if PY3_OR_LATER
                        else pickle.HIGHEST_PROTOCOL)

        Pickler.__init__(self, file_handle, protocol=protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _create_array_wrapper(self, array):
        """Create and returns a numpy array wrapper from a numpy array."""
        order = 'F' if (array.flags.f_contiguous and
                        not array.flags.c_contiguous) else 'C'
        allow_mmap = not self.buffered and not array.dtype.hasobject
        self.array_count += 1
        filename = "{}.array{:0>3}.part".format(self.filename, self.array_count)
        wrapper = NumpyArrayWrapper(type(array),
                                    array.shape, order, array.dtype,
                                    allow_mmap=allow_mmap,
                                    filename=filename)

        return wrapper

    def save(self, obj):
        """Subclass the Pickler `save` method.

        This is a total abuse of the Pickler class in order to use the numpy
        persistence function `save` instead of the default pickle
        implementation. The numpy array is replaced by a custom wrapper in the
        pickle persistence stack and the serialized array is written right
        after in the file. Warning: the file produced does not follow the
        pickle format. As such it can not be read with `pickle.load`.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                                 self.np.matrix,
                                                 self.np.memmap):
            if type(obj) is self.np.memmap:
                # Pickling doesn't work with memmapped arrays
                obj = self.np.asanyarray(obj)

            # The array wrapper is pickled instead of the real array.
            wrapper = self._create_array_wrapper(obj)

            # Wrapper is always pure-old python object
            Pickler.save(self, wrapper)

            # A framer was introduced with pickle protocol 4 and we want to
            # ensure the wrapper object is written before the numpy array
            # buffer in the pickle file.
            # See https://www.python.org/dev/peps/pep-3154/#framing to get
            # more information on the framer behavior.
            if self.proto >= 4:
                self.framer.commit_frame(force=True)

            # And then array bytes are written right after the wrapper.
            wrapper.write_array(obj, dirname(abspath(self.filename)), file_opener=self.file_opener)
            return

        return Pickler.save(self, obj)


if PY3_OR_LATER and sys.version_info[1] >= 3:
    class NumpyPicklerFast(fast_pickle.Pickler):
        """A pickler to persist big data efficiently.

        The main features of this object are:
        * persistence of numpy arrays in a single file.
        * optional compression with a special care on avoiding memory copies.

        Attributes
        ----------
        fp: file
            File object handle used for serializing the input object.
        protocol: int
            Pickle protocol used. Default is pickle.DEFAULT_PROTOCOL under
            python 3, pickle.HIGHEST_PROTOCOL otherwise.
        """

        def __init__(self, filename, file_handle, file_opener, protocol=None):
            super(NumpyPicklerFast, self).__init__(file_handle, protocol=protocol)

            import copyreg
            self.dispatch_table = copyreg.dispatch_table.copy()
            self.dispatch_table[0] = 0

            self.filename = filename
            self.file_opener = file_opener

            self.buffered = isinstance(file_handle, BinaryZlibFile)
            self.array_count = 0

            # By default we want a pickle protocol that only changes with
            # the major python version and not the minor one
            if protocol is None:
                protocol = (pickle.DEFAULT_PROTOCOL if PY3_OR_LATER
                            else pickle.HIGHEST_PROTOCOL)

            # delayed import of numpy, to avoid tight coupling
            try:
                import numpy as np
            except ImportError:
                np = None
            self.np = np

            def reduce_array(array):
                wrapper = self._create_array_wrapper(array)
                wrapper.write_array(array, dirname(abspath(self.filename)), file_opener=self.file_opener)

                return wrapper.__reduce__()

            if self.np:
                self.dispatch_table[self.np.ndarray] = reduce_array
                self.dispatch_table[self.np.matrix] = reduce_array
                self.dispatch_table[self.np.memmap] = reduce_array

        def _create_array_wrapper(self, array):
            """Create and returns a numpy array wrapper from a numpy array."""
            order = 'F' if (array.flags.f_contiguous and
                            not array.flags.c_contiguous) else 'C'
            allow_mmap = not self.buffered and not array.dtype.hasobject
            self.array_count += 1
            filename = "{}.array{:0>3}.part".format(self.filename, self.array_count)
            wrapper = NumpyArrayWrapper(type(array),
                                        array.shape, order, array.dtype,
                                        allow_mmap=allow_mmap,
                                        filename=filename)

            return wrapper


    NumpyPickler = NumpyPicklerFast
else:
    NumpyPickler = NumpyPicklerCompatible


###############################################################################
# Utility functions

def dump(value, filename, compress=0, protocol=None, cache_size=None):
    """Persist an arbitrary Python object into one file.

    Parameters
    -----------
    value: any Python object
        The object to store to disk.
    filename: str or pathlib.Path
        The path of the file in which it is to be stored. The compression
        method corresponding to one of the supported filename extensions ('.z',
        '.gz', '.bz2', '.xz' or '.lzma') will be used automatically.
    compress: int from 0 to 9 or bool or 2-tuple, optional
        Optional compression level for the data. 0 or False is no compression.
        Higher value means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
        If compress is True, the compression level used is 3.
        If compress is a 2-tuple, the first element must correspond to a string
        between supported compressors (e.g 'zlib', 'gzip', 'bz2', 'lzma'
        'xz'), the second element must be an integer from 0 to 9, corresponding
        to the compression level.
    protocol: positive int
        Pickle protocol, see pickle.dump documentation for more details.
    cache_size: positive int, optional
        This option is deprecated in 0.10 and has no effect.

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
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take extra extra memory during
    dump and load.

    """

    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    is_filename = isinstance(filename, _basestring)
    is_fileobj = hasattr(filename, "write")

    compress_method = 'zlib'  # zlib is the default compression method.
    if compress is True:
        # By default, if compress is enabled, we want to be using 3 by default
        compress_level = 3
    elif isinstance(compress, tuple):
        # a 2-tuple was set in compress
        if len(compress) != 2:
            raise ValueError(
                'Compress argument tuple should contain exactly 2 elements: '
                '(compress method, compress level), you passed {0}'
                .format(compress))
        compress_method, compress_level = compress
    else:
        compress_level = compress

    if compress_level is not False and compress_level not in range(10):
        # Raising an error if a non valid compress level is given.
        raise ValueError(
            'Non valid compress level given: "{0}". Possible values are '
            '{1}.'.format(compress_level, list(range(10))))

    if compress_method not in _COMPRESSORS:
        # Raising an error if an unsupported compression method is given.
        raise ValueError(
            'Non valid compression method given: "{0}". Possible values are '
            '{1}.'.format(compress_method, _COMPRESSORS))

    if not is_filename and not is_fileobj:
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
            'Second argument should be a filename or a file-like object, '
            '%s (type %s) was given.'
            % (filename, type(filename))
        )

    if is_filename and not isinstance(compress, tuple):
        # In case no explicit compression was requested using both compression
        # method and level in a tuple and the filename has an explicit
        # extension, we select the corresponding compressor.
        if filename.endswith('.z'):
            compress_method = 'zlib'
        elif filename.endswith('.gz'):
            compress_method = 'gzip'
        elif filename.endswith('.bz2'):
            compress_method = 'bz2'
        elif filename.endswith('.lzma'):
            compress_method = 'lzma'
        elif filename.endswith('.xz'):
            compress_method = 'xz'
        else:
            # no matching compression method found, we unset the variable to
            # be sure no compression level is set afterwards.
            compress_method = None

        if compress_method in _COMPRESSORS and compress_level == 0:
            # we choose a default compress_level of 3 in case it was not given
            # as an argument (using compress).
            compress_level = 3

    if not PY3_OR_LATER and compress_method in ('lzma', 'xz'):
        raise NotImplementedError("{0} compression is only available for "
                                  "python version >= 3.3. You are using "
                                  "{1}.{2}".format(compress_method,
                                                   sys.version_info[0],
                                                   sys.version_info[1]))

    if cache_size is not None:
        # Cache size is deprecated starting from version 0.10
        warnings.warn("Please do not set 'cache_size' in joblib.dump, "
                      "this parameter has no effect and will be removed. "
                      "You used 'cache_size={0}'".format(cache_size),
                      DeprecationWarning, stacklevel=2)
    file_opener = partial(open, mode='wb')

    if compress_level != 0:
        file_opener = partial(_write_fileobject, compress=(compress_method, compress_level))

    with file_opener(filename) as file_handle:
        NumpyPickler(filename, file_handle, file_opener, protocol=protocol).dump(value)

    # For compatibility, the list of created filenames (e.g with one element
    # after 0.10.0) is returned by default.
    return [filename]


def _unpickle(fobj, filename="", mmap_mode=None):
    """Internal unpickling function."""
    # We are careful to open the file handle early and keep it open to
    # avoid race-conditions on renames.
    # That said, if data is stored in companion files, which can be
    # the case with the old persistence format, moving the directory
    # will create a race when joblib tries to access the companion
    # files.
    obj = None
    try:
        obj = fast_pickle.load(fobj)
    except UnicodeDecodeError as exc:
        # More user-friendly error message
        if PY3_OR_LATER:
            new_exc = ValueError(
                'You may be trying to read with '
                'python 3 a joblib pickle generated with python 2. '
                'This feature is not supported by joblib.')
            new_exc.__cause__ = exc
            raise new_exc
        # Reraise exception with Python 2
        raise

    return obj


def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Parameters
    -----------
    filename: str or pathlib.Path
        The path of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
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

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmaped.
    """
    if Path is not None and isinstance(filename, Path):
        filename = str(filename)

    try:
        storage.filename = filename
        storage.mmap_mode = mmap_mode

        if hasattr(filename, "read"):
            fobj = filename
            filename = getattr(fobj, 'name', '')
            with _read_fileobject(fobj, filename, mmap_mode) as fobj:
                obj = _unpickle(fobj)
        else:
            @contextlib.contextmanager
            def file_opener(filename):
                file_ = None
                try:
                    file_ = open(filename, 'rb')
                    with _read_fileobject(file_, filename, mmap_mode) as fobj:
                        yield fobj
                finally:
                    if file_ is not None:
                        file_.close()

            storage.file_opener = file_opener

            with file_opener(filename) as fobj:
                if isinstance(fobj, _basestring):
                    # if the returned file object is a string, this means we
                    # try to load a pickle file generated with an version of
                    # Joblib so we load it with joblib compatibility function.
                    return load_compatibility(fobj)
                obj = _unpickle(fobj, filename, mmap_mode)
    finally:
        del storage.filename
        del storage.mmap_mode
        try:
            del storage.file_opener
        except Exception:
            pass

    return obj
