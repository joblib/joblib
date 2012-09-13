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

import os
import sys
import threading
import atexit
import tempfile
import shutil
from cPickle import loads
from cPickle import dumps
from pickle import Pickler
from pickle import HIGHEST_PROTOCOL
try:
    from io import BytesIO
except ImportError:
    # Python 2.5 compat
    from StringIO import StringIO as BytesIO
try:
    from multiprocessing.pool import Pool
    from multiprocessing import Pipe
    from multiprocessing.synchronize import Lock
    from multiprocessing.forking import assert_spawning
except ImportError:
    class Pool(object):
        """Dummy class for python 2.5 backward compat"""
        pass
try:
    import numpy as np
except ImportError:
    np = None

from .numpy_pickle import load
from .numpy_pickle import dump
from .hashing import hash


def reduce_memmap(a):
    """Pickle the descriptors of a memmap instance to reopen on same file"""
    mode = a.mode
    if mode == 'w+':
        # Do not make the subprocess erase the data from the parent memmap
        # inadvertently
        mode = 'r+'
    order = 'F' if a.flags['F_CONTIGUOUS'] else 'C'
    return (np.memmap, (a.filename, a.dtype, mode, a.offset, a.shape, order))


class ArrayMemmapReducer(object):
    """Reducer callable to dump large arrays to memmap files.

    Parameters
    ----------
    max_nbytes: int
        Threshold to trigger memmaping of large arrays to files created
        a folder.
    temp_folder: str
        Path of a folder where files for backing memmaped arrays are created.
    mmap_mode: 'r', 'r+', 'w+' or 'c'
        Mode for the created memmap datastructure. See the documentation of
        numpy.memmap for more details.
    verbse: int, optional, 0 by default
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
        if a.nbytes > self._max_nbytes:
            # check that the folder exists (lazily create the pool temp folder
            # if required)
            if not os.path.exists(self._temp_folder):
                os.makedirs(self._temp_folder)

            # Find a unique, concurrent safe filename for writing the
            # content of this array only once.
            basename = "%d-%d-%d-%s.pkl" % (
                os.getpid(), id(threading.current_thread()), id(a), hash(a))
            filename = os.path.join(self._temp_folder, basename)

            # In case the same array with the same content is passed several
            # times to the pool subprocess children, serialize it only once
            if not os.path.exists(filename):
                if self.verbose > 0:
                    print "Memmaping (shape=%r, dtype=%s) to new file %s" % (
                        a.shape, a.dtype, filename)
                dump(a, filename)
            elif self.verbose > 1:
                print "Memmaping (shape=%s, dtype=%s) to old file %s" % (
                        a.shape, a.dtype, filename)

            # Let's use the memmap reducer
            return reduce_memmap(load(filename, mmap_mode=self._mmap_mode))
        else:
            # do not convert a into memmap, let pickler do its usual copy with
            # the default system pickler
            if self.verbose > 1:
                print "Pickling array (shape=%r, dtype=%s)." % (
                    a.shape, a.dtype)
            return (loads, (dumps(a, protocol=HIGHEST_PROTOCOL),))


class CustomizablePickler(Pickler):
    """Pickler that accepts custom reducers.

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    `reducers` is expected to be a sequence of `(type, callable)`
    pairs where `callable` is a function that give an instance of
    `type` will return a tuple `(constructor, tuple_of_objects)`
    to rebuild an instance out of the pickled `tuple_of_objects`
    as would return a `__reduce__` method. See the standard library
    documentation on pickling for more details.

    """

    def __init__(self, writer, reducers=(), protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)
        # Make the dispatch registry an instance level attribute instead of a
        # reference to the class dictionary
        self.dispatch = Pickler.dispatch.copy()
        for type, reduce_func in reducers:
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

    `reducers` is expected to be a sequence of `(type, callable)`
    pairs where `callable` is a function that give an instance of
    `type` will return a tuple `(constructor, tuple_of_objects)`
    to rebuild an instance out of the pickled `tuple_of_objects`
    as would return a `__reduce__` method. See the standard library
    documentation on pickling for more details.

    """

    def __init__(self, reducers=()):
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
            wlock_acquire, wlock_release = self._wlock.acquire, \
                                                self._wlock.release

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
    sequences of `(type, callable)` pairs where `callable` is a
    function that give an instance of `type` will return a tuple
    `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for
    more details.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 forward_reducers=(), backward_reducers=()):
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
        for sharing memory with worker processes. If None, this
        will use the system temporary folder or can be overridden
        with TMP, TMPDIR or TEMP environment variables.
    max_nbytes int or None, optional, 1e6 by default
        Threshold on the size of arrays passed to the workers that
        triggers automated memmory mapping in temp_folder.
        Use None to disable memmaping of large arrays.
    forward_reducers: sequence of tuples (see bellow), optional
        Reducers used to pickle objects passed from master to worker
        processes.
    backward_reducers: sequence of tuples (see bellow), optional
        Reducers used to pickle return values from workers back to the
        master process.
    verbose: int, optional
        Make it possible to monitor how the communication of numpy arrays
        with the subprocess is handled (pickling or memmaping)

    `forward_reducers` and `backward_reducers` are expected to be
    sequences of `(type, callable)` pairs where `callable` is a
    function that give an instance of `type` will return a tuple
    `(constructor, tuple_of_objects)` to rebuild an instance out
    of the pickled `tuple_of_objects` as would return a `__reduce__`
    method. See the standard library documentation on pickling for
    more details.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 temp_folder=None, max_nbytes=1e6, mmap_mode='c',
                 forward_reducers=(), backward_reducers=(),
                 verbose=0):
        f_reducers = []
        b_reducers = []

        # Prepare a subfolder name for the serialization of this particular
        # pool instance (do not create in advance to spare FS write access if
        # no array is to be dumped):
        if temp_folder is None:
            temp_folder = tempfile.gettempdir()
        temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
        self._temp_folder = temp_folder = os.path.join(
            temp_folder, "joblib_memmaping_pool_%d_%d" % (
                os.getpid(), id(self)))

        # Register the garbage collector at program exit in case caller
        # forget's to call it earlier
        atexit.register(self._collect_tempfile)

        if np is not None:
            if max_nbytes is not None:
                reduce_ndarray = ArrayMemmapReducer(
                    max_nbytes, temp_folder, mmap_mode, verbose)
                # We only register the automatic array to memmap reducer in
                # the forward direction (from parent to child processes)
                f_reducers.append((np.ndarray, reduce_ndarray))
            f_reducers.append((np.memmap, reduce_memmap))

            # Communication from child process to the parent process always
            # pickles numpy.ndarray without dumping them as memmap to avoid
            # confusing the caller and make it tricky to collect the temporary
            # folder
            b_reducers.append((np.memmap, reduce_memmap))
        f_reducers.extend(forward_reducers)
        b_reducers.extend(backward_reducers)

        super(MemmapingPool, self).__init__(processes=processes,
                                            initializer=initializer,
                                            initargs=initargs,
                                            forward_reducers=f_reducers,
                                            backward_reducers=b_reducers)

    def _collect_tempfile(self):
        if os.path.exists(self._temp_folder):
            shutil.rmtree(self._temp_folder, ignore_errors=True)

    def terminate(self):
        super(MemmapingPool, self).terminate()
        self._collect_tempfile()
