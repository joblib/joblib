"""Custom implementation of multiprocessing.Pool with custom pickler

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.

This module should not be imported if multiprocessing is not
available.  as it implements subclasses of multiprocessing Pool
that uses a custom alternative to SimpleQueue

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# Copyright: 2012, Olivier Grisel
# License: BSD 3 clause

import sys
from pickle import Pickler
from pickle import Unpickler
from pickle import HIGHEST_PROTOCOL
try:
    from io import BytesIO
except ImportError:
    # Python 2.5 compat
    from StringIO import StringIO as BytesIO
from multiprocessing import Pipe
from multiprocessing.pool import Pool
from multiprocessing.synchronize import Lock
from multiprocessing.forking import assert_spawning
try:
    from numpy import memmap
except ImportError:
    memmap = None


class CustomPickler(Pickler):
    """Pickler that accepts custom reducers."""

    def __init__(self, writer, reducers=()):
        Pickler.__init__(self, writer)
        for type, reduce_func in reducers:
            self.register(type, reduce_func)

    def register(self, type, reduce_func):
        def dispatcher(self, obj):
            reduced = reduce_func(obj)
            self.save_reduce(obj=obj, *reduced)
        self.dispatch[type] = dispatcher


class PicklingQueue(object):
    """Locked Pipe implementation that uses a custom pickler

    This class is an alternative to the multiprocessing implementation
    of SimpleQueue in order to make it possible to pass custom
    pickling reducers.

    """

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

    def _make_methods(self):
        if self._reducers:
            def recv():
                return Unpickler(BytesIO(self._reader.recv_bytes())).load()
            self._recv = recv
        else:
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
                CustomPickler(buffer, self._reducers).dump(obj)
                self._writer.send_bytes(buffer.getvalue())
            self._send = send
        else:
            self._send = send = self._writer.send
        if self._wlock is None:
            # writes to a message oriented win32 pipe are atomic
            self.put = send
        else:
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
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv
