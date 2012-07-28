import sys
import numpy as np
import mmap
import ctypes
import os.path

from _multiprocessing import address_of_buffer


valid_filemodes = ["c", "r+"]
mode_equivalents = {
    "copyonwrite": "c",
    "readwrite": "r+",
}


class SharedArray(np.ndarray):
    """Array sharable by multiple processes using the mmap kernel features.

    This class is a subclass of numpy.ndarray that uses a shared
    memory buffer allocated by the kernel using the mmap system API
    so as to be shared by multiple workers in a multiprocessing
    context without incuring memory copies of the array content.

    The default pickling behavior of numpy arrays is thus overridden
    (see the __reduce__ method) under the assumption that a SharedArray
    instance will always be unpickled on the same machine as the
    original instance and that the memory is still allocated (i.e.
    in the same Python process or a subprocess).

    TODO: implement shared multiprocessing locks.

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

    def __new__(subtype, shape, dtype=np.uint8, mode='r+', order='C',
                address=None):

        try:
            mode = mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" %
                                 (valid_filemodes
                                  + mode_equivalents.keys()))

        dtype = np.dtype(dtype)
        _dbytes = dtype.itemsize

        if not isinstance(shape, tuple):
            shape = (shape,)
        size = 1
        for k in shape:
            size *= k
        bytes = long(size * _dbytes)

        if mode == 'c':
            acc = mmap.ACCESS_COPY
        else:
            acc = mmap.ACCESS_WRITE

        if address is None:
            buffer = mmap.mmap(-1, bytes, access=acc)
            address = address_of_buffer(buffer)[0]
        else:
            # Reuse an existing memory address from an anonymous mmap
            buffer = (ctypes.c_byte * bytes).from_address(address)

        self = np.ndarray.__new__(subtype, shape, dtype=dtype, buffer=buffer,
                                  order=order)
        self._address = address
        self.mode = mode
        return self

    def __array_finalize__(self, obj):
        # XXX: should we really do this? Check the numpy subclassing reference
        # to guess what is the best behavior to follow here
        if hasattr(obj, '_address') and np.may_share_memory(self, obj):
            self._address = obj._address
            self.mode = obj.mode
        else:
            self._address = None
            self.mode = None

    def __reduce__(self):
        """Support for pickling while still sharing the original buffer"""
        order = 'F' if self.flags['F_CONTIGUOUS'] else 'C'
        return SharedArray, (self.shape, self.dtype, self.mode, order,
                             self._address)


def as_shared_array(a, dtype=None, shape=None, order=None):
    """Make an anonymous SharedArray instance out of a array-like.

    If a is already a SharedArray instance, return it-self.

    Otherwise a new shared buffer is allocated and the content of
    a is copied into it.

    """
    if isinstance(a, SharedArray):
        return a
    else:
        a = np.asanyarray(a, dtype=dtype, order=order)
        if shape is not None and shape != a.shape:
            a = a.reshape(shape)
        order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
        sa = SharedArray(a.shape, dtype=a.dtype, order=order)
        sa[:] = a[:]
        return sa


class SharedMemmap(np.memmap):
    """A np.memmap derivative with pickling support for multiprocessing

    The default pickling behavior of numpy arrays is overriden as
    we make the assumption that a pickled SharedMemmap instance
    will be unpickled on the same machine in a multiprocessing or
    joblib.Parallel context.

    In this case for the original and the unpickled copied SharedMemmap
    instances share a reference to the same memory mapped files
    hence implementing filesystem-backed shared memory.

    This is extremly useful to avoid useless memory copy of a
    readonly datasets provided as input to several workers in a
    multiprocessing Pool for instance.

    TODO: add the multiprocessing shared lock feature

    Parameters and attributes are inherited from the nump.memmap
    class.  The only change is a dedicated __reduce__ implementation
    used for pickling SharedMemmap without performing any kind of
    additional in-memory allocation for the content of the array.

    """

    def __reduce__(self):
        """Support for pickling while still sharing the original buffer"""
        order = 'F' if self.flags['F_CONTIGUOUS'] else 'C'
        return SharedMemmap, (self.filename, self.dtype, self.mode,
                              self.offset, self.shape, order)


def as_shared_memmap(a):
    """Create a SharedMemmap instance pointing to the same file.

    The original memmap array and its shared variant will have the
    same behavior but the pickling of the later is overriden to
    avoid inmemory copies so as to be used efficiently in a single mode,
    multiple pooled workers multiprocessing context.

    Parameters
    ----------
    a : numpy.memmap
        The memmap array to share.
    """
    if isinstance(a, SharedMemmap):
        return a
    order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
    return SharedMemmap(a.filename, dtype=a.dtype, shape=a.shape,
                        offset=a.offset, mode=a.mode, order=order)
