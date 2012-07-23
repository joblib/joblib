import sys
import numpy as np
import mmap
import ctypes
import os.path

from _multiprocessing import address_of_buffer


# Constants taken from numpy.core.memmap module that is unfortunately
# shadowed by the memmap class itself, hence the local copy here

valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]
mode_equivalents = {
    "readonly": "r",
    "copyonwrite": "c",
    "readwrite": "r+",
    "write": "w+",
}


class SharedArray(np.ndarray):
    """Array sharable by multiple processes using the mmap kernel features.

    This class aims to blend the good features from numpy.memmap (behave like
    a regular numpy array with n-dimensional slicing and views), while adding
    the good feature from multiprocessing.Array like picklability, anonymous
    shared memory and shared locks for safe write concurrent to the array by
    several processes).

    All of this is implemented in a cross-platform manner without
    any dependencies beyond numpy and the Python standard library
    (using the mmap and the multiprocessing modules).

    Parameters
    ----------
    TODO

    Attributes
    ----------
    TODO

    Examples
    --------
    TODO
    """

    __array_priority__ = -100.0  # TODO: why? taken from np.memmap

    def __new__(subtype, filename=None, address=None, dtype=np.uint8,
                mode=None, offset=0, shape=None, order='C'):
        if mode is None:
            if filename is None:
                mode = 'w+'
            elif getattr(filename, 'mode', '').startswith('r'):
                mode = 'r'
            else:
                mode = 'r+'

        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" %
                                 (valid_filemodes
                                  + mode_equivalents.keys()))

        if hasattr(filename, 'read'):
            fileobj = filename
            fileno = fileobj.fileno()
            own_file = False
        elif filename is None:
            fileno = -1
            fileobj = None
            own_file = False
        else:
            fileobj = open(filename, (mode == 'c' and 'r' or mode) + 'b')
            fileno = fileobj.fileno()
            own_file = True

        if (mode == 'w+' or fileobj is None) and shape is None:
            raise ValueError("shape must be given")

        if fileobj is not None:
            fileobj.seek(0, 2)  # move 0 bytes relative to end of the file
            file_length = fileobj.tell()
        else:
            file_length = None

        dtype = np.dtype(dtype)
        _dbytes = dtype.itemsize

        if shape is None:
            bytes = file_length - offset
            if (bytes % _dbytes):
                if own_file:
                    fileobj.close()
                raise ValueError("Size of available data is not a "
                        "multiple of the data-type size.")
            size = bytes // _dbytes
            shape = (size,)
        else:
            if not isinstance(shape, tuple):
                shape = (shape,)
            size = 1
            for k in shape:
                size *= k

        bytes = long(offset + size * _dbytes)

        if (fileobj is not None
            and (mode == 'w+' or (mode == 'r+' and file_length < bytes))):
            fileobj.seek(bytes - 1, 0)
            fileobj.write(np.compat.asbytes('\0'))
            fileobj.flush()

        if mode == 'c':
            acc = mmap.ACCESS_COPY
        elif mode == 'r':
            acc = mmap.ACCESS_READ
        else:
            acc = mmap.ACCESS_WRITE

        if address is None:
            if sys.version_info[:2] >= (2, 6):
                # The offset keyword in mmap.mmap needs Python >= 2.6
                start = offset - offset % mmap.ALLOCATIONGRANULARITY
                bytes -= start
                offset -= start
                mm = mmap.mmap(fileno, bytes, access=acc, offset=start)
            else:
                mm = mmap.mmap(fileno, bytes, access=acc)
            buffer = mm
        else:
            # Reuse an existing memory address from an anonymous mmap
            buffer = (ctypes.c_byte * bytes).from_address(address)
            mm = None

        self = np.ndarray.__new__(subtype, shape, dtype=dtype, buffer=buffer,
                                  offset=offset, order=order)
        self._mmap = mm
        self.offset = offset
        self.mode = mode

        if isinstance(filename, basestring):
            self.filename = os.path.abspath(filename)
        elif hasattr(filename, "name"):
            self.filename = os.path.abspath(filename.name)
        else:
            self.filename = None  # anonymouse mmap

        if own_file:
            fileobj.close()

        return self

    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None

    def flush(self):
        """Write any changes in the array to the file on disk."""
        self._mmap.flush()

    def __reduce__(self):
        """Support for pickling while still sharing the original buffer"""
        order = 'F' if self.flags['F_CONTIGUOUS'] else 'C'
        address = None
        if self.filename is None:
            address, _ = address_of_buffer(self._mmap)
        return SharedArray, (self.filename, address, self.dtype, self.mode,
                             self.offset, self.shape, order)


def assharedarray(a, dtype=None, shape=None, order=None):
    """Make an anonymous SharedArray instance out of a

    If a is already a SharedArray instance, return it-self.
    Otherwise a new shared buffer is allocated and the content of a is copied
    into it.
    """
    if isinstance(a, SharedArray):
        return a
    elif isinstance(a, np.memmap):
        order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
        return SharedArray(filename=a.filename, dtype=a.dtype, shape=a.shape,
                           offset=a.offset, mode=a.mode, order=order)
    else:
        a = np.asanyarray(a, dtype=dtype, order=order)
        if shape is not None and shape != a.shape:
            a = a.reshape(shape)
        order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
        sa = SharedArray(dtype=a.dtype, shape=a.shape, order=order)
        sa[:] = a[:]
        return sa
