"""Custom implementation of multiprocessing.Pool with custom pickler

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.

This module should not be imported if multiprocessing is not
available as it implements subclasses of multiprocessing Pool
that uses a custom alternative to SimpleQueue.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Copyright: 2012, Olivier Grisel
# License: BSD 3 clause

from mmap import mmap
import os
import sys
import threading
import atexit
import tempfile
import shutil
import warnings
try:
    from cPickle import loads
    from cPickle import dumps
    from pickle import Pickler as _Pickler  # pure Python pickler in 2
except ImportError:
    from pickle import loads
    from pickle import dumps
    from pickle import _Pickler  # pure Python pickler in 3

from pickle import HIGHEST_PROTOCOL
from io import BytesIO

from multiprocessing.pool import Pool
from multiprocessing import Pipe
from multiprocessing.synchronize import Lock
from multiprocessing.forking import assert_spawning

try:
    import numpy as np
    from numpy.lib.stride_tricks import as_strided
except ImportError:
    np = None

from .numpy_pickle import load
from .numpy_pickle import dump
from .hashing import hash

# Some system have a ramdisk mounted by default, we can use it instead of /tmp
# as the default folder to dump big arrays to share with subprocesses
SYSTEM_SHARED_MEM_FS = '/dev/shm'


###############################################################################
# Support for efficient transient pickling of numpy data structures


def _get_backing_memmap(a):
    """Recursively look up the original np.memmap instance base if any"""
    b = getattr(a, 'base', None)
    if b is None:
        # TODO: check scipy sparse datastructure if scipy is installed
        # a nor its descendants do not have a memmap base
        return None

    elif isinstance(b, mmap):
        # a is already a real memmap instance.
        return a

    else:
        # Recursive exploration of the base ancestry
        return _get_backing_memmap(b)


def has_shareable_memory(a):
    """Return True if a is backed by some mmap buffer directly or not"""
    return _get_backing_memmap(a) is not None


def strided_from_memmap(filename, dtype, mode, offset, order, shape, strides):
    """Reconstruct an array view on a memmory mapped file"""
    if mode == 'w+':
        # Do not zero the original data when unpickling
        mode = 'r+'
    m = np.memmap(filename, dtype=dtype, shape=shape, mode=mode, offset=offset,
                  order=order)
    if strides is None:
        return m
    return as_strided(m, strides=strides)


def _reduce_memmap_backed(a, m):
    """Pickling reduction for memmap backed arrays

    a is expected to be an instance of np.ndarray (or np.memmap)
    m is expected to be an instance of np.memmap on the top of the ``base``
    attribute ancestry of a. ``m.base`` should be the real python mmap object.
    """
    # offset that comes from the striding differences between a and m
    a_start = np.byte_bounds(a)[0]
    m_start = np.byte_bounds(m)[0]
    offset = a_start - m_start

    # offset from the backing memmap
    offset += m.offset

    if m.flags['F_CONTIGUOUS']:
        order = 'F'
    else:
        # The backing memmap buffer is necessarily contiguous hence C if not
        # Fortran
        order = 'C'

    # If array is a contiguous view, no need to pass the strides
    if a.flags['F_CONTIGUOUS'] or a.flags['C_CONTIGUOUS']:
        strides = None
    else:
        strides = a.strides
    return (strided_from_memmap,
            (m.filename, a.dtype, m.mode, offset, order, a.shape, strides))


def reduce_memmap(a):
    """Pickle the descriptors of a memmap instance to reopen on same file"""
    m = _get_backing_memmap(a)
    if m is not None:
        # m is a real mmap backed memmap instance, reduce a preserving striding
        # information
        return _reduce_memmap_backed(a, m)
    else:
        # This memmap instance is actually backed by a regular in-memory
        # buffer: this can happen when using binary operators on numpy.memmap
        # instances
        return (loads, (dumps(np.asarray(a), protocol=HIGHEST_PROTOCOL),))


class ArrayMemmapReducer(object):
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmaping of large arrays to files created
        a folder.
    temp_folder: str
        Path of a folder where files for backing memmaped arrays are created.
    mmap_mode: 'r', 'r+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details. Note: 'w+' is coerced to 'r+'
        automatically to avoid zeroing the data on unpickling.
    verbose: int, optional, 0 by default
        If verbose > 0, memmap creations are logged.
        If verbose > 1, both memmap creations, reuse and array pickling are
        logged.

    """

    def __init__(self, max_nbytes, temp_folder, mmap_mode, verbose=0):
        self._max_nbytes = max_nbytes
        self._temp_folder = temp_folder
        self._mmap_mode = mmap_mode
        self.verbose = int(verbose)

    def __call__(self, a):
        m = _get_backing_memmap(a)
        if m is not None:
            # a is already backed by a memmap file, let's reuse it directly
            return _reduce_memmap_backed(a, m)

        if self._max_nbytes is not None and a.nbytes > self._max_nbytes:
            # check that the folder exists (lazily create the pool temp folder
            # if required)
            try:
                os.makedirs(self._temp_folder)
            except OSError as e:
                if e.errno != 17:
                    # Errno 17 corresponds to 'Directory exists'
                    raise e

            # Find a unique, concurrent safe filename for writing the
            # content of this array only once.
            basename = "%d-%d-%d-%s.pkl" % (
                os.getpid(), id(threading.current_thread()), id(a), hash(a))
            filename = os.path.join(self._temp_folder, basename)

            # In case the same array with the same content is passed several
            # times to the pool subprocess children, serialize it only once
            if not os.path.exists(filename):
                if self.verbose > 0:
                    print("Memmaping (shape=%r, dtype=%s) to new file %s" % (
                        a.shape, a.dtype, filename))
                dump(a, filename)
            elif self.verbose > 1:
                print("Memmaping (shape=%s, dtype=%s) to old file %s" % (
                    a.shape, a.dtype, filename))

            # Let's use the memmap reducer
            return reduce_memmap(load(filename, mmap_mode=self._mmap_mode))
        else:
            # do not convert a into memmap, let pickler do its usual copy with
            # the default system pickler
            if self.verbose > 1:
                print("Pickling array (shape=%r, dtype=%s)." % (
                    a.shape, a.dtype))
            return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))


###############################################################################
# Enable custom pickling in Pool queues

class CustomizablePickler(_Pickler):
    """Pickler that accepts custom reducers.

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    `reducers` is expected expected to be a dictionary with key/values
    being `(type, callable)` pairs where `callable` is a function that
    give an instance of `type` will return a tuple `(constructor,
    tuple_of_objects)` to rebuild an instance out of the pickled
    `tuple_of_objects` as would return a `__reduce__` method. See the
    standard library documentation on pickling for more details.

    """

    # We override the pure Python pickler as its the only way to be able to
    # customize the dispatch table without side effects in Python 2.6
    # to 3.2. For Python 3.3+ we could leverage the new dispatch_table
    # feature from http://bugs.python.org/issue14166 but that would render
    # the code more complex.

    def __init__(self, writer, reducers=None, protocol=HIGHEST_PROTOCOL):
        _Pickler.__init__(self, writer, protocol=protocol)
        # Make the dispatch registry an instance level attribute instead of a
        # reference to the class dictionary
        self.dispatch = _Pickler.dispatch.copy()
        for type, reduce_func in reducers.items():
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        def dispatcher(self, obj):
            reduced = reduce_func(obj)
            self.save_reduce(obj=obj, *reduced)
        self.dispatch[type] = dispatcher


class CustomizablePicklingQueue(object):
    """Locked Pipe implementation that uses a customizable pickler.

    This class is an alternative to the multiprocessing implementation
    of SimpleQueue in order to make it possible to pass custom
    pickling reducers, for instance to avoid memory copy when passing
    memmory mapped datastructures.

    `reducers` is expected expected to be a dictionary with key/values
    being `(type, callable)` pairs where `callable` is a function that
    give an instance of `type` will return a tuple `(constructor,
    tuple_of_objects)` to rebuild an instance out of the pickled
    `tuple_of_objects` as would return a `__reduce__` method. See the
    standard library documentation on pickling for more details.
    """

    def __init__(self, reducers=None):
        self._reducers = reducers
        self._reader, self._writer = Pipe(duplex=False)
        self._rlock = Lock()
        if sys.platform == 'win32':
            self._wlock = None
        else:
            self._wlock = Lock()
        self._make_methods()

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock,
                self._reducers)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock,
         self._reducers) = state
        self._make_methods()

    def empty(self):
        return not self._reader.poll()

    def _make_methods(self):
        self._recv = recv = self._reader.recv
        racquire, rrelease = self._rlock.acquire, self._rlock.release

        def get():
            racquire()
            try:
                return recv()
            finally:
                rrelease()

        self.get = get

        if self._reducers:
            def send(obj):
                buffer = BytesIO()
                CustomizablePickler(buffer, self._reducers).dump(obj)
                self._writer.send_bytes(buffer.getvalue())
            self._send = send
        else:
            self._send = send = self._writer.send
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.put = send
        else:
            wlock_acquire, wlock_release = (
                self._wlock.acquire, self._wlock.release)

            def put(obj):
                wlock_acquire()
                try:
                    return send(obj)
                finally:
                    wlock_release()

            self.put = put


class PicklingPool(Pool):
    """Pool implementation with customizable pickling reducers.

    This is useful to control how data is shipped between processes
    and makes it possible to use shared memory without useless
    copies induces by the default pickling methods of the original
    objects passed as arguments to dispatch.

    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that give an instance of `type` will return
    a tuple `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for more
    details.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 forward_reducers=None, backward_reducers=None):
        if forward_reducers is None:
            forward_reducers = dict()
        if backward_reducers is None:
            backward_reducers = dict()
        self._forward_reducers = forward_reducers
        self._backward_reducers = backward_reducers
        super(PicklingPool, self).__init__(processes=processes,
                                           initializer=initializer,
                                           initargs=initargs)

    def _setup_queues(self):
        self._inqueue = CustomizablePicklingQueue(self._forward_reducers)
        self._outqueue = CustomizablePicklingQueue(self._backward_reducers)
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv


def delete_folder(folder_path):
    """Utility function to cleanup a temporary folder if still existing"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


class MemmapingPool(PicklingPool):
    """Process pool that shares large arrays to avoid memory copy.

    This drop-in replacement for `multiprocessing.pool.Pool` makes
    it possible to work efficiently with shared memory in a numpy
    context.

    Existing instances of numpy.memmap are preserved: the child
    suprocesses will have access to the same shared memory in the
    original mode except for the 'w+' mode that is automatically
    transformed as 'r+' to avoid zeroing the original data upon
    instanciation.

    Furthermore large arrays from the parent process are automatically
    dumped to a temporary folder on the filesystem such as child
    processes to access their content via memmaping (file system
    backed shared memory).

    Note: it is important to call the terminate method to collect
    the temporary folder used by the pool.

    Parameters
    ----------
    processes: int, optional
        Number of worker processes running concurrently in the pool.
    initializer: callable, optional
        Callable executed on worker process creation.
    initargs: tuple, optional
        Arguments passed to the initializer callable.
    temp_folder: str, optional
        Folder to be used by the pool for memmaping large arrays
        for sharing memory with worker processes. If None, this will try in
        order:
        - a folder pointed by the JOBLIB_TEMP_FOLDER environment variable,
        - /dev/shm if the folder exists and is writable: this is a RAMdisk
          filesystem available by default on modern Linux distributions,
        - the default system temporary folder that can be overridden
          with TMP, TMPDIR or TEMP environment variables, typically /tmp
          under Unix operating systems.
    max_nbytes int or None, optional, 1e6 by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memmory mapping in temp_folder.
        Use None to disable memmaping of large arrays.
    forward_reducers: dictionary, optional
        Reducers used to pickle objects passed from master to worker
        processes: see below.
    backward_reducers: dictionary, optional
        Reducers used to pickle return values from workers back to the
        master process.
    verbose: int, optional
        Make it possible to monitor how the communication of numpy arrays
        with the subprocess is handled (pickling or memmaping)

    `forward_reducers` and `backward_reducers` are expected to be
    dictionaries with key/values being `(type, callable)` pairs where
    `callable` is a function that give an instance of `type` will return
    a tuple `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for more
    details.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 temp_folder=None, max_nbytes=1e6, mmap_mode='c',
                 forward_reducers=None, backward_reducers=None,
                 verbose=0):
        if forward_reducers is None:
            forward_reducers = dict()
        if backward_reducers is None:
            backward_reducers = dict()

        # Prepare a sub-folder name for the serialization of this particular
        # pool instance (do not create in advance to spare FS write access if
        # no array is to be dumped):
        if temp_folder is None:
            temp_folder = os.environ.get('JOBLIB_TEMP_FOLDER', None)
        if temp_folder is None:
            if os.path.exists(SYSTEM_SHARED_MEM_FS):
                try:
                    joblib_folder = os.path.join(
                        SYSTEM_SHARED_MEM_FS, 'joblib')
                    if not os.path.exists(joblib_folder):
                        os.makedirs(joblib_folder)
                except IOError:
                    # Missing rights in the the /dev/shm partition, ignore
                    pass
        if temp_folder is None:
            # Fallback to the default tmp folder, typically /tmp
            temp_folder = tempfile.gettempdir()
        temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
        self._temp_folder = temp_folder = os.path.join(
            temp_folder, "joblib_memmaping_pool_%d_%d" % (
                os.getpid(), id(self)))

        # Register the garbage collector at program exit in case caller forgets
        # to call terminate explicitly: note we do not pass any reference to
        # self to ensure that this callback won't prevent garbage collection of
        # the pool instance and related file handler resources such as POSIX
        # semaphores and pipes
        atexit.register(lambda: delete_folder(temp_folder))

        if np is not None:
            # Register smart numpy.ndarray reducers that detects memmap
            # backed arrays and is able to dump to memmap large in-memory
            # arrays over the max_nbytes threshold
            forward_reduce_ndarray = ArrayMemmapReducer(
                max_nbytes, temp_folder, mmap_mode, verbose)
            forward_reducers[np.ndarray] = forward_reduce_ndarray
            forward_reducers[np.memmap] = reduce_memmap

            # Communication from child process to the parent process always
            # pickles in-memory numpy.ndarray without dumping them as memmap
            # to avoid confusing the caller and make it tricky to collect the
            # temporary folder
            backward_reduce_ndarray = ArrayMemmapReducer(
                None, temp_folder, mmap_mode, verbose)
            backward_reducers[np.ndarray] = backward_reduce_ndarray
            backward_reducers[np.memmap] = reduce_memmap

        super(MemmapingPool, self).__init__(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            forward_reducers=forward_reducers,
            backward_reducers=backward_reducers)

    def terminate(self):
        super(MemmapingPool, self).terminate()
        delete_folder(self._temp_folder)
