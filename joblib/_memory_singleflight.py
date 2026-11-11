"""Local advisory locks for opt-in cache miss coalescing."""

import errno
import os
import time
from contextlib import contextmanager
from hashlib import sha256
from threading import local

_owned_locks = local()


@contextmanager
def cache_call_lock(location, func_id, args_id):
    # Do not unlink a released lock: waiters may still hold its inode.
    key = sha256((func_id + "\0" + args_id).encode()).hexdigest()
    token = (os.path.realpath(location), key)
    owned = getattr(_owned_locks, "keys", None)
    if owned is None:
        owned = _owned_locks.keys = set()
    if token in owned:
        raise RuntimeError("Recursive singleflight call with the same cache key")
    lock_dir = os.path.join(location, ".singleflight")
    os.makedirs(lock_dir, exist_ok=True)
    with open(os.path.join(lock_dir, key + ".lock"), "a+b") as lock:
        if os.name == "nt":
            import msvcrt

            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            owned.add(token)
            try:
                yield
            finally:
                owned.remove(token)
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            owned.add(token)
            try:
                yield
            finally:
                owned.remove(token)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
