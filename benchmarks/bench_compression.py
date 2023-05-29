"""Script comparing different pickling strategies."""

from joblib.numpy_pickle import NumpyPickler, NumpyUnpickler
from joblib.numpy_pickle_utils import BinaryZlibFile, BinaryGzipFile
from pickle import _Pickler, _Unpickler, Pickler, Unpickler
import numpy as np
import bz2
import lzma
import time
import io
import sys
import os
from collections import OrderedDict


def fileobj(obj, fname, mode, kwargs):
    """Create a file object."""
    return obj(fname, mode, **kwargs)


def bufferize(f, buf):
    """Bufferize a fileobject using buf."""
    if buf is None:
        return f
    else:
        if (buf.__name__ == io.BufferedWriter.__name__ or
                buf.__name__ == io.BufferedReader.__name__):
            return buf(f, buffer_size=10 * 1024 ** 2)
        return buf(f)


def _load(unpickler, fname, f):
    if unpickler.__name__ == NumpyUnpickler.__name__:
        p = unpickler(fname, f)
    else:
        p = unpickler(f)

    return p.load()


def print_line(obj, strategy, buffer, pickler, dump, load, disk_used):
    """Nice printing function."""
    print('% 20s | %6s | % 14s | % 7s | % 5.1f | % 5.1f | % 5s' % (
          obj, strategy, buffer, pickler, dump, load, disk_used))


class PickleBufferedWriter():
    """Protect the underlying fileobj against numerous calls to write
    This is achieved by internally keeping a list of small chunks and
    only flushing to the backing fileobj if passed a large chunk or
    after a threshold on the number of small chunks.
    """

    def __init__(self, fileobj,
                 max_buffer_size=10 * 1024 ** 2):
        self._fileobj = fileobj
        self._chunks = chunks = []

        # As the `write` method is called many times by the pickler,
        # attribute look ups on the self's __dict__ are too expensive
        # hence we define a closure here with all the regularly
        # accessed parameters
        def _write(data):
            chunks.append(data)
            if len(chunks) > max_buffer_size:
                self.flush()
        self.write = _write

    def flush(self):
        self._fileobj.write(b''.join(self._chunks[:]))
        del self._chunks[:]

    def close(self):
        self.flush()
        self._fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class PickleBufferedReader():
    """Protect the underlying fileobj against numerous calls to write
    This is achieved by internally keeping a list of small chunks and
    only flushing to the backing fileobj if passed a large chunk or
    after a threshold on the number of small chunks.
    """

    def __init__(self, fileobj,
                 max_buffer_size=10 * 1024 ** 2):
        self._fileobj = fileobj
        self._buffer = bytearray(max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self._position = 0

    def read(self, n=None):
        data = b''
        if n is None:
            data = self._fileobj.read()
        else:
            while len(data) < n:
                if self._position == 0:
                    self._buffer = self._fileobj.read(self.max_buffer_size)
                elif self._position == self.max_buffer_size:
                    self._position = 0
                    continue
                next_position = min(self.max_buffer_size,
                                    self._position + n - len(data))
                data += self._buffer[self._position:next_position]
                self._position = next_position
        return data

    def readline(self):
        line = []
        while True:
            c = self.read(1)
            line.append(c)
            if c == b'\n':
                break
        return b''.join(line)

    def close(self):
        self._fileobj.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def run_bench():
    print('% 20s | %10s | % 12s | % 8s | % 9s | % 9s | % 5s' % (
          'Object', 'Compression', 'Buffer', 'Pickler/Unpickler',
          'dump time (s)', 'load time (s)', 'Disk used (MB)'))
    print("--- | --- | --- | --- | --- | --- | ---")

    for oname, obj in objects.items():
        # Looping over the objects (array, dict, etc)
        if isinstance(obj, np.ndarray):
            osize = obj.nbytes / 1e6
        else:
            osize = sys.getsizeof(obj) / 1e6

        for cname, f in compressors.items():
            fobj = f[0]
            fname = f[1]
            fmode = f[2]
            fopts = f[3]
            # Looping other defined compressors
            for bname, buf in bufs.items():
                writebuf = buf[0]
                readbuf = buf[1]
                # Looping other picklers
                for pname, p in picklers.items():
                    pickler = p[0]
                    unpickler = p[1]
                    t0 = time.time()
                    # Now pickling the object in the file
                    if (writebuf is not None and
                            writebuf.__name__ == io.BytesIO.__name__):
                        b = writebuf()
                        p = pickler(b)
                        p.dump(obj)
                        with fileobj(fobj, fname, fmode, fopts) as f:
                            f.write(b.getvalue())
                    else:
                        with bufferize(fileobj(fobj, fname, fmode, fopts),
                                       writebuf) as f:
                            p = pickler(f)
                            p.dump(obj)
                    dtime = time.time() - t0
                    t0 = time.time()
                    # Now loading the object from the file
                    obj_r = None
                    if (readbuf is not None and
                            readbuf.__name__ == io.BytesIO.__name__):
                        b = readbuf()
                        with fileobj(fobj, fname, 'rb', {}) as f:
                            b.write(f.read())
                        b.seek(0)
                        obj_r = _load(unpickler, fname, b)
                    else:
                        with bufferize(fileobj(fobj, fname, 'rb', {}),
                                       readbuf) as f:
                            obj_r = _load(unpickler, fname, f)
                    ltime = time.time() - t0
                    if isinstance(obj, np.ndarray):
                        assert (obj == obj_r).all()
                    else:
                        assert obj == obj_r
                    print_line("{} ({:.1f}MB)".format(oname, osize),
                               cname,
                               bname,
                               pname,
                               dtime,
                               ltime,
                               "{:.2f}".format(os.path.getsize(fname) / 1e6))


# Defining objects used in this bench
DICT_SIZE = int(1e6)
ARRAY_SIZE = int(1e7)

arr = np.random.normal(size=(ARRAY_SIZE))
arr[::2] = 1

# Objects used for testing
objects = OrderedDict([
    ("dict", dict((i, str(i)) for i in range(DICT_SIZE))),
    ("list", [i for i in range(DICT_SIZE)]),
    ("array semi-random", arr),
    ("array random", np.random.normal(size=(ARRAY_SIZE))),
    ("array ones", np.ones((ARRAY_SIZE))), ])

#  We test 3 different picklers
picklers = OrderedDict([
    # Python implementation of Pickler/Unpickler
    ("Pickle", (_Pickler, _Unpickler)),
    # C implementation of Pickler/Unpickler
    ("cPickle", (Pickler, Unpickler)),
    # Joblib Pickler/Unpickler designed for numpy arrays.
    ("Joblib", (NumpyPickler, NumpyUnpickler)), ])

# The list of supported compressors used for testing
compressors = OrderedDict([
    ("No", (open, '/tmp/test_raw', 'wb', {})),
    ("Zlib", (BinaryZlibFile, '/tmp/test_zlib', 'wb', {'compresslevel': 3})),
    ("Gzip", (BinaryGzipFile, '/tmp/test_gzip', 'wb', {'compresslevel': 3})),
    ("Bz2", (bz2.BZ2File, '/tmp/test_bz2', 'wb', {'compresslevel': 3})),
    ("Xz", (lzma.LZMAFile, '/tmp/test_xz', 'wb',
                           {'preset': 3, 'check': lzma.CHECK_NONE})),
    ("Lzma", (lzma.LZMAFile, '/tmp/test_lzma', 'wb',
                             {'preset': 3, 'format': lzma.FORMAT_ALONE})), ])

# Test 3 buffering strategies
bufs = OrderedDict([
    ("None", (None, None)),
    ("io.BytesIO", (io.BytesIO, io.BytesIO)),
    ("io.Buffered", (io.BufferedWriter, io.BufferedReader)),
    ("PickleBuffered", (PickleBufferedWriter, PickleBufferedReader)), ])

if __name__ == "__main__":
    run_bench()
