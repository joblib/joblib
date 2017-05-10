"""Utilities for fast persistence of big data, with optional compression."""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import sys
import io
import warnings
import contextlib
from contextlib import closing

from ._compat import PY3_OR_LATER, PY27
from .compressor import BinaryGzipFile, BinaryZlibFile
from .compressor import _EXTRA_COMPRESSORS, _PREFIXES_MAX_LEN
from .compressor import (_ZFILE_PREFIX, _ZLIB_PREFIX, _GZIP_PREFIX,
                         _BZ2_PREFIX, _XZ_PREFIX, _LZMA_PREFIX, _LZ4_PREFIX)
from .compressor import register_compressor

if PY3_OR_LATER:
    Unpickler = pickle._Unpickler
    Pickler = pickle._Pickler
    xrange = range
else:
    Unpickler = pickle.Unpickler
    Pickler = pickle.Pickler

try:
    import numpy as np
except ImportError:
    np = None

try:
    import lzma
except ImportError:
    lzma = None

try:
    # The python standard library can be built without bz2 so we make bz2
    # usage optional.
    # see https://github.com/scikit-learn/scikit-learn/issues/7526 for more
    # details.
    import bz2
except ImportError:
    bz2 = None

try:
    import lz4
except ImportError:
    lz4 = None

if PY3_OR_LATER and lz4 is not None:
    if not tuple(map(int, lz4.__version__.split('.')[:-1])) >= (0, 19):
        # lz4 can only be used starting from version 0.19.
        lz4 = None
    else:
        import lz4.frame
        register_compressor('lz4', _LZ4_PREFIX, lz4.frame.LZ4FrameFile)
else:
    lz4 = None

# Buffer size used in io.BufferedReader and io.BufferedWriter
_IO_BUFFER_SIZE = 1024 ** 2


def _is_raw_file(fileobj):
    """Check if fileobj is a raw file object, e.g created with open."""
    if PY3_OR_LATER:
        fileobj = getattr(fileobj, 'raw', fileobj)
        return isinstance(fileobj, io.FileIO)
    else:
        return isinstance(fileobj, file)  # noqa


def _get_prefixes_max_len():
    # Compute the max prefix len of extra registered compression formats.
    extra_max_prefix_len = 0
    if _EXTRA_COMPRESSORS:
        extra_max_prefix_len = max(len(obj.prefix)
                                   for obj in _EXTRA_COMPRESSORS.values())

    return max(_PREFIXES_MAX_LEN, extra_max_prefix_len)


###############################################################################
# Cache file utilities
def _detect_compressor(fileobj):
    """Return the compressor matching fileobj.

    Parameters
    ----------
    fileobj: file object

    Returns
    -------
    str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat', 'not-compressed'}
    """
    # Read the magic number in the first bytes of the file.
    max_prefix_len = _get_prefixes_max_len()
    if hasattr(fileobj, 'peek'):
        # Peek allows to read those bytes without moving the cursor in the
        # file whic.
        first_bytes = fileobj.peek(max_prefix_len)
    else:
        # Fallback to seek if the fileobject is not peekable.
        first_bytes = fileobj.read(max_prefix_len)
        fileobj.seek(0)

    if first_bytes.startswith(_ZLIB_PREFIX):
        return "zlib"
    elif first_bytes.startswith(_GZIP_PREFIX):
        return "gzip"
    elif first_bytes.startswith(_BZ2_PREFIX):
        return "bz2"
    elif first_bytes.startswith(_LZMA_PREFIX):
        return "lzma"
    elif first_bytes.startswith(_XZ_PREFIX):
        return "xz"
    elif first_bytes.startswith(_ZFILE_PREFIX):
        return "compat"
    for name, compressor in _EXTRA_COMPRESSORS.items():
        if first_bytes.startswith(compressor.prefix):
            return name

    return "not-compressed"


def _buffered_read_file(fobj):
    """Return a buffered version of a read file object."""
    if PY27 and bz2 is not None and isinstance(fobj, bz2.BZ2File):
        # Python 2.7 doesn't work with BZ2File through a buffer: "no
        # attribute 'readable'" error.
        return fobj
    else:
        return io.BufferedReader(fobj, buffer_size=_IO_BUFFER_SIZE)


def _buffered_write_file(fobj):
    """Return a buffered version of a write file object."""
    if PY27 and bz2 is not None and isinstance(fobj, bz2.BZ2File):
        # Python 2.7 doesn't work with BZ2File through a buffer: no attribute
        # 'writable'.
        # BZ2File doesn't implement the file object context manager in python 2
        # so we wrap the fileobj using `closing`.
        return closing(fobj)
    else:
        return io.BufferedWriter(fobj, buffer_size=_IO_BUFFER_SIZE)


@contextlib.contextmanager
def _read_fileobject(fileobj, filename, mmap_mode=None):
    """Utility function opening the right fileobject from a filename.

    The magic number is used to choose between the type of file object to open:
    * regular file object (default)
    * zlib file object
    * gzip file object
    * bz2 file object
    * lzma file object (for xz and lzma compressor)

    Parameters
    ----------
    fileobj: file object
    compressor: str in {'zlib', 'gzip', 'bz2', 'lzma', 'xz', 'compat',
                        'not-compressed'}
    filename: str
        filename path corresponding to the fileobj parameter.
    mmap_mode: str
        memory map mode that should be used to open the pickle file. This
        parameter is useful to verify that the user is not trying to one with
        compression. Default: None.

    Returns
    -------
        a file like object

    """
    # Detect if the fileobj contains compressed data.
    compressor = _detect_compressor(fileobj)

    if compressor == 'compat':
        # Compatibility with old pickle mode: simply return the input
        # filename "as-is" and let the compatibility function be called by the
        # caller.
        warnings.warn("The file '%s' has been generated with a joblib "
                      "version less than 0.10. "
                      "Please regenerate this pickle file." % filename,
                      DeprecationWarning, stacklevel=2)
        yield filename
    else:
        # based on the compressor detected in the file, we open the
        # correct decompressor file object, wrapped in a buffer.
        if compressor == 'zlib':
            fileobj = _buffered_read_file(BinaryZlibFile(fileobj, 'rb'))
        elif compressor == 'gzip':
            fileobj = _buffered_read_file(BinaryGzipFile(fileobj, 'rb'))
        elif compressor == 'bz2' and bz2 is not None:
            if PY3_OR_LATER:
                fileobj = _buffered_read_file(bz2.BZ2File(fileobj, 'rb'))
            else:
                # In python 2, BZ2File doesn't support a fileobj opened in
                # binary mode. In this case, we pass the filename.
                fileobj = _buffered_read_file(bz2.BZ2File(fileobj.name, 'rb'))
        elif (compressor == 'lzma' or compressor == 'xz'):
            if PY3_OR_LATER and lzma is not None:
                # We support lzma only in python 3 because in python 2 users
                # may have installed the pyliblzma package, which also provides
                # the lzma module, but that unfortunately doesn't fully support
                # the buffer interface required by joblib.
                # See https://github.com/joblib/joblib/issues/403 for details.
                fileobj = _buffered_read_file(lzma.LZMAFile(fileobj, 'rb'))
            else:
                raise NotImplementedError("Lzma decompression is not "
                                          "supported for this version of "
                                          "python ({}.{})"
                                          .format(sys.version_info[0],
                                                  sys.version_info[1]))
        else:
            for name, compressor_type in _EXTRA_COMPRESSORS.items():
                if compressor == name:
                    obj = compressor_type.object
                    fileobj = _buffered_read_file(obj(filename, 'rb'))
                    break

        # Checking if incompatible load parameters with the type of file:
        # mmap_mode cannot be used with compressed file or in memory buffers
        # such as io.BytesIO.
        if mmap_mode is not None:
            if isinstance(fileobj, io.BytesIO):
                warnings.warn('In memory persistence is not compatible with '
                              'mmap_mode "%(mmap_mode)s" flag passed. '
                              'mmap_mode option will be ignored.'
                              % locals(), stacklevel=2)
            elif compressor != 'not-compressed':
                warnings.warn('mmap_mode "%(mmap_mode)s" is not compatible '
                              'with compressed file %(filename)s. '
                              '"%(mmap_mode)s" flag will be ignored.'
                              % locals(), stacklevel=2)
            elif not _is_raw_file(fileobj):
                warnings.warn('"%(fileobj)r" is not a raw file, mmap_mode '
                              '"%(mmap_mode)s" flag will be ignored.'
                              % locals(), stacklevel=2)

        yield fileobj


def _write_fileobject(filename, compress=("zlib", 3)):
    """Return the right compressor file object in write mode."""
    compressmethod = compress[0]
    compresslevel = compress[1]
    if compressmethod == "gzip":
        return _buffered_write_file(BinaryGzipFile(filename, 'wb',
                                    compresslevel=compresslevel))
    elif compressmethod == "bz2" and bz2 is not None:
        return _buffered_write_file(bz2.BZ2File(filename, 'wb',
                                                compresslevel=compresslevel))
    elif lzma is not None and compressmethod == "xz":
        return _buffered_write_file(lzma.LZMAFile(filename, 'wb',
                                                  check=lzma.CHECK_NONE,
                                                  preset=compresslevel))
    elif lzma is not None and compressmethod == "lzma":
        return _buffered_write_file(lzma.LZMAFile(filename, 'wb',
                                                  preset=compresslevel,
                                                  format=lzma.FORMAT_ALONE))
    elif compressmethod in _EXTRA_COMPRESSORS.keys():
        file_objet = _EXTRA_COMPRESSORS[compressmethod].object
        return _buffered_write_file(file_objet(filename, 'wb'))
    else:
        return _buffered_write_file(BinaryZlibFile(filename, 'wb',
                                    compresslevel=compresslevel))



# Utility functions/variables from numpy required for writing arrays.
# We need at least the functions introduced in version 1.9 of numpy. Here,
# we use the ones from numpy 1.10.2.
BUFFER_SIZE = 2 ** 18  # size of buffer for reading npz files in bytes


def _read_bytes(fp, size, error_template="ran out of data"):
    """Read from file-like object until size bytes are read.

    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.

    This function was taken from numpy/lib/format.py in version 1.10.2.

    Parameters
    ----------
    fp: file-like object
    size: int
    error_template: str

    Returns
    -------
    a bytes object
        The data read in bytes.

    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except io.BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data
