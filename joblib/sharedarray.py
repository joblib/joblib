import numpy as np
import mmap
import os.path



class SharedArray(ndarray):
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

    def __new__(subtype, filename=None, address=None, dtype=uint8, mode=None,
                offset=0, shape=None, order='C'):
        if mode is None:
            mode = 'r+' if filename is not None else 'w+'

        try:
            mode = np.memmap.mode_equivalents[mode]
        except KeyError:
            if mode not in valid_filemodes:
                raise ValueError("mode must be one of %s" %
                                 (valid_filemodes
                                  + np.memmap.mode_equivalents.keys()))

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

        if sys.version_info[:2] >= (2, 6):
            # The offset keyword in mmap.mmap needs Python >= 2.6
            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            offset -= start
            mm = mmap.mmap(fileno, bytes, access=acc, offset=start)
        else:
            mm = mmap.mmap(fileno, bytes, access=acc)

        self = ndarray.__new__(subtype, shape, dtype=dtype, buffer=mm,
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
