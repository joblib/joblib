
"""
lockfile.py - Platform-independent advisory file locks.

Requires Python 2.5 unless you apply 2.4.diff
Locking is done on a per-thread basis instead of a per-process basis.

Usage:

>>> lock = FileLock('somefile')
>>> try:
...     lock.acquire()
... except AlreadyLocked:
...     print 'somefile', 'is locked already.'
... except LockFailed:
...     print 'somefile', 'can\\'t be locked.'
... else:
...     print 'got lock'
got lock
>>> print lock.is_locked()
True
>>> lock.release()

>>> lock = FileLock('somefile')
>>> print lock.is_locked()
False
>>> with lock:
...    print lock.is_locked()
True
>>> print lock.is_locked()
False
>>> # It is okay to lock twice from the same thread...
>>> with lock:
...     lock.acquire()
...
>>> # Though no counter is kept, so you can't unlock multiple times...
>>> print lock.is_locked()
False

Exceptions:

    Error - base class for other exceptions
        LockError - base class for all locking exceptions
            AlreadyLocked - Another thread or process already holds the lock
            LockFailed - Lock failed for some other reason
        UnlockError - base class for all unlocking exceptions
            AlreadyUnlocked - File was not locked.
            NotMyLock - File was locked but not by the current thread/process
"""

from __future__ import division

import sys
import socket
import os
import thread
import threading
import time
import errno
import urllib

# Work with PEP8 and non-PEP8 versions of threading module.
if not hasattr(threading, "current_thread"):
    threading.current_thread = threading.currentThread
if not hasattr(threading.Thread, "get_name"):
    threading.Thread.get_name = threading.Thread.getName

__all__ = ['Error', 'LockError', 'LockTimeout', 'AlreadyLocked',
           'LockFailed', 'UnlockError', 'NotLocked', 'NotMyLock',
           'LinkFileLock', 'MkdirFileLock', 'SQLiteFileLock']

class Error(Exception):
    """
    Base class for other exceptions.

    >>> try:
    ...   raise Error
    ... except Exception:
    ...   pass
    """
    pass

class LockError(Error):
    """
    Base class for error arising from attempts to acquire the lock.

    >>> try:
    ...   raise LockError
    ... except Error:
    ...   pass
    """
    pass

class LockTimeout(LockError):
    """Raised when lock creation fails within a user-defined period of time.

    >>> try:
    ...   raise LockTimeout
    ... except LockError:
    ...   pass
    """
    pass

class AlreadyLocked(LockError):
    """Some other thread/process is locking the file.

    >>> try:
    ...   raise AlreadyLocked
    ... except LockError:
    ...   pass
    """
    pass

class LockFailed(LockError):
    """Lock file creation failed for some other reason.

    >>> try:
    ...   raise LockFailed
    ... except LockError:
    ...   pass
    """
    pass

class UnlockError(Error):
    """
    Base class for errors arising from attempts to release the lock.

    >>> try:
    ...   raise UnlockError
    ... except Error:
    ...   pass
    """
    pass

class NotLocked(UnlockError):
    """Raised when an attempt is made to unlock an unlocked file.

    >>> try:
    ...   raise NotLocked
    ... except UnlockError:
    ...   pass
    """
    pass

class NotMyLock(UnlockError):
    """Raised when an attempt is made to unlock a file someone else locked.

    >>> try:
    ...   raise NotMyLock
    ... except UnlockError:
    ...   pass
    """
    pass

class LockBase:
    """Base class for platform-specific lock classes."""
    def __init__(self, path, threaded=True):
        """
        >>> lock = LockBase('somefile')
        >>> lock = LockBase('somefile', threaded=False)
        """
        self.path = path
        self.lock_file = os.path.abspath(path) + ".lock"
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        if threaded:
            name = threading.current_thread().get_name()
            tname = "%s-" % urllib.quote(name, safe="")
        else:
            tname = ""
        dirname = os.path.dirname(self.lock_file)
        self.unique_name = os.path.join(dirname,
                                        "%s.%s%s" % (self.hostname,
                                                     tname,
                                                     self.pid))

    def acquire(self, timeout=None):
        """
        Acquire the lock.

        * If timeout is omitted (or None), wait forever trying to lock the
          file.

        * If timeout > 0, try to acquire the lock for that many seconds.  If
          the lock period expires and the file is still locked, raise
          LockTimeout.

        * If timeout <= 0, raise AlreadyLocked immediately if the file is
          already locked.
        """
        raise NotImplemented("implement in subclass")

    def release(self):
        """
        Release the lock.

        If the file is not locked, raise NotLocked.
        """
        raise NotImplemented("implement in subclass")

    def is_locked(self):
        """
        Tell whether or not the file is locked.
        """
        raise NotImplemented("implement in subclass")

    def i_am_locking(self):
        """
        Return True if this object is locking the file.
        """
        raise NotImplemented("implement in subclass")

    def break_lock(self):
        """
        Remove a lock.  Useful if a locking thread failed to unlock.
        """
        raise NotImplemented("implement in subclass")

    def __enter__(self):
        """
        Context manager support.
        """
        self.acquire()
        return self

    def __exit__(self, *_exc):
        """
        Context manager support.
        """
        self.release()

class LinkFileLock(LockBase):
    """Lock access to a file using atomic property of link(2)."""

    def acquire(self, timeout=None):
        try:
            open(self.unique_name, "wb").close()
        except IOError:
            raise LockFailed("failed to create %s" % self.unique_name)

        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        while True:
            # Try and create a hard link to it.
            try:
                os.link(self.unique_name, self.lock_file)
            except OSError:
                # Link creation failed.  Maybe we've double-locked?
                nlinks = os.stat(self.unique_name).st_nlink
                if nlinks == 2:
                    # The original link plus the one I created == 2.  We're
                    # good to go.
                    return
                else:
                    # Otherwise the lock creation failed.
                    if timeout is not None and time.time() > end_time:
                        os.unlink(self.unique_name)
                        if timeout > 0:
                            raise LockTimeout
                        else:
                            raise AlreadyLocked
                    time.sleep(timeout is not None and timeout/10 or 0.1)
            else:
                # Link creation succeeded.  We're good to go.
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked
        elif not os.path.exists(self.unique_name):
            raise NotMyLock
        os.unlink(self.unique_name)
        os.unlink(self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name) and
                os.stat(self.unique_name).st_nlink == 2)

    def break_lock(self):
        if os.path.exists(self.lock_file):
            os.unlink(self.lock_file)

class MkdirFileLock(LockBase):
    """Lock file by creating a directory."""
    def __init__(self, path, threaded=True):
        """
        >>> lock = MkdirFileLock('somefile')
        >>> lock = MkdirFileLock('somefile', threaded=False)
        """
        LockBase.__init__(self, path, threaded)
        if threaded:
            tname = "%x-" % thread.get_ident()
        else:
            tname = ""
        # Lock file itself is a directory.  Place the unique file name into
        # it.
        self.unique_name  = os.path.join(self.lock_file,
                                         "%s.%s%s" % (self.hostname,
                                                      tname,
                                                      self.pid))

    def acquire(self, timeout=None):
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        if timeout is None:
            wait = 0.1
        else:
            wait = max(0, timeout / 10)

        while True:
            try:
                os.mkdir(self.lock_file)
            except OSError:
                err = sys.exc_info()[1]
                if err.errno == errno.EEXIST:
                    # Already locked.
                    if os.path.exists(self.unique_name):
                        # Already locked by me.
                        return
                    if timeout is not None and time.time() > end_time:
                        if timeout > 0:
                            raise LockTimeout
                        else:
                            # Someone else has the lock.
                            raise AlreadyLocked
                    time.sleep(wait)
                else:
                    # Couldn't create the lock for some other reason
                    raise LockFailed("failed to create %s" % self.lock_file)
            else:
                open(self.unique_name, "wb").close()
                return

    def release(self):
        if not self.is_locked():
            raise NotLocked
        elif not os.path.exists(self.unique_name):
            raise NotMyLock
        os.unlink(self.unique_name)
        os.rmdir(self.lock_file)

    def is_locked(self):
        return os.path.exists(self.lock_file)

    def i_am_locking(self):
        return (self.is_locked() and
                os.path.exists(self.unique_name))

    def break_lock(self):
        if os.path.exists(self.lock_file):
            for name in os.listdir(self.lock_file):
                os.unlink(os.path.join(self.lock_file, name))
            os.rmdir(self.lock_file)

class SQLiteFileLock(LockBase):
    "Demonstration of using same SQL-based locking."

    import tempfile
    _fd, testdb = tempfile.mkstemp()
    os.close(_fd)
    os.unlink(testdb)
    del _fd, tempfile

    def __init__(self, path, threaded=True):
        LockBase.__init__(self, path, threaded)
        self.lock_file = unicode(self.lock_file)
        self.unique_name = unicode(self.unique_name)

        import sqlite3
        self.connection = sqlite3.connect(SQLiteFileLock.testdb)
        
        c = self.connection.cursor()
        try:
            c.execute("create table locks"
                      "("
                      "   lock_file varchar(32),"
                      "   unique_name varchar(32)"
                      ")")
        except sqlite3.OperationalError:
            pass
        else:
            self.connection.commit()
            import atexit
            atexit.register(os.unlink, SQLiteFileLock.testdb)

    def acquire(self, timeout=None):
        end_time = time.time()
        if timeout is not None and timeout > 0:
            end_time += timeout

        if timeout is None:
            wait = 0.1
        elif timeout <= 0:
            wait = 0
        else:
            wait = timeout / 10

        cursor = self.connection.cursor()

        while True:
            if not self.is_locked():
                # Not locked.  Try to lock it.
                cursor.execute("insert into locks"
                               "  (lock_file, unique_name)"
                               "  values"
                               "  (?, ?)",
                               (self.lock_file, self.unique_name))
                self.connection.commit()

                # Check to see if we are the only lock holder.
                cursor.execute("select * from locks"
                               "  where unique_name = ?",
                               (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) > 1:
                    # Nope.  Someone else got there.  Remove our lock.
                    cursor.execute("delete from locks"
                                   "  where unique_name = ?",
                                   (self.unique_name,))
                    self.connection.commit()
                else:
                    # Yup.  We're done, so go home.
                    return
            else:
                # Check to see if we are the only lock holder.
                cursor.execute("select * from locks"
                               "  where unique_name = ?",
                               (self.unique_name,))
                rows = cursor.fetchall()
                if len(rows) == 1:
                    # We're the locker, so go home.
                    return
                    
            # Maybe we should wait a bit longer.
            if timeout is not None and time.time() > end_time:
                if timeout > 0:
                    # No more waiting.
                    raise LockTimeout
                else:
                    # Someone else has the lock and we are impatient..
                    raise AlreadyLocked

            # Well, okay.  We'll give it a bit longer.
            time.sleep(wait)

    def release(self):
        if not self.is_locked():
            raise NotLocked
        if not self.i_am_locking():
            raise NotMyLock((self._who_is_locking(), self.unique_name))
        cursor = self.connection.cursor()
        cursor.execute("delete from locks"
                       "  where unique_name = ?",
                       (self.unique_name,))
        self.connection.commit()

    def _who_is_locking(self):
        cursor = self.connection.cursor()
        cursor.execute("select unique_name from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        return cursor.fetchone()[0]
        
    def is_locked(self):
        cursor = self.connection.cursor()
        cursor.execute("select * from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        rows = cursor.fetchall()
        return not not rows

    def i_am_locking(self):
        cursor = self.connection.cursor()
        cursor.execute("select * from locks"
                       "  where lock_file = ?"
                       "    and unique_name = ?",
                       (self.lock_file, self.unique_name))
        return not not cursor.fetchall()

    def break_lock(self):
        cursor = self.connection.cursor()
        cursor.execute("delete from locks"
                       "  where lock_file = ?",
                       (self.lock_file,))
        self.connection.commit()

if hasattr(os, "link"):
    FileLock = LinkFileLock
else:
    FileLock = MkdirFileLock
