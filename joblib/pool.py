"""Custom implementation of multiprocessing.Pool with custom pickler

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.

This module should not be imported if multiprocessing is not available.
as it implements subclasses of multiprocessing Pool and SimpleQueue.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Copyright: 2012, Olivier Grisel
# License: BSD 3 clause

from pickle import Pickler
from pickle import HIGHEST_PROTOCOL
from io import StringIO
from multiprocessing.queues import SimpleQueue
from multiprocessing import Pipe
from multiprocessing.pool import Pool
from multiprocessing.synchronize import Lock
from multiprocessing.forking import assert_spawning
try:
    from numpy import memmap
except ImportError:
    memmap = None


class PicklingQueue(SimpleQueue):
    """SimpleQueue implementation that uses a custom pickler"""

    def __init__(self, reducers):
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

    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._rlock, self._wlock)

    def __setstate__(self, state):
        (self._reader, self._writer, self._rlock, self._wlock) = state
        self._make_methods()

    def _make_methods(self):
        recv = self._reader.recv
        racquire, rrelease = self._rlock.acquire, self._rlock.release
        self._pickle = Pickler()
        def get():
            racquire()
            try:
                return recv()
            finally:
                rrelease()
        self.get = get

        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.put = self._writer.send
        else:
            send = self._writer.send
            wacquire, wrelease = self._wlock.acquire, self._wlock.release
            def put(obj):
                wacquire()
                try:
                    return send(obj)
                finally:
                    wrelease()
            self.put = put
class PicklingPool(Pool):
    """Pool implementation with custom pickling reducers

    This is useful to control how data is shipped between processes and makes it
    possible to use shared memory without useless copies induces by the default
    pickling methods of the original objects passed as arguments to dispatch.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 maxtasksperchild=None, reducers=()):
        self.reducers = reducers
        super(PicklingPool, self).__init__()

    def _setup_queues(self):
        self._inqueue = PicklingQueue(self.reducers)
        self._outqueue = PicklingQueue(self.reducers)
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv
