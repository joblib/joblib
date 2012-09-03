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

import sys
from pickle import Pickler
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


class CustomizablePickler(Pickler):
    """Pickler that accepts custom reducers.

    HIGHEST_PROTOCOL is selected by default as this pickler is used
    to pickle ephemeral datastructures for interprocess communication
    hence no backward compatibility is required.

    """

    def __init__(self, writer, reducers=(), protocol=HIGHEST_PROTOCOL):
        Pickler.__init__(self, writer, protocol=protocol)
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
            wacquire, wrelease = self._wlock.acquire, self._wlock.release
            def put(obj):
                wacquire()
                try:
                    return send(obj)
                finally:
                    wrelease()
            self.put = put

class PicklingPool(Pool):
    """Pool implementation with customizable pickling reducers.

    This is useful to control how data is shipped between processes
    and makes it possible to use shared memory without useless
    copies induces by the default pickling methods of the original
    objects passed as arguments to dispatch.

    """

    def __init__(self, processes=None, initializer=None, initargs=(),
                 reducers=()):
        self.reducers = reducers
        super(PicklingPool, self).__init__(processes=None,
                                           initializer=initializer,
                                           initargs=initargs)

    def _setup_queues(self):
        self._inqueue = CustomizablePicklingQueue(self.reducers)
        self._outqueue = CustomizablePicklingQueue(self.reducers)
        self._quick_put = self._inqueue._send
        self._quick_get = self._outqueue._recv
