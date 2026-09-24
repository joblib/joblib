try:
    # Python 2.7: use the C pickle to speed up
    # test_concurrency_safe_write which pickles big python objects
    import cPickle as cpickle
except ImportError:
    import pickle as cpickle
import functools
import time
from pickle import PicklingError

import pytest

from joblib import Parallel, delayed
from joblib._store_backends import (
    CacheWarning,
    FileSystemStoreBackend,
    concurrency_safe_write,
)
from joblib.backports import concurrency_safe_rename
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, timeout


def write_func(output, filename):
    with open(filename, "wb") as f:
        cpickle.dump(output, f)


def load_func(expected, filename):
    for i in range(10):
        try:
            with open(filename, "rb") as f:
                reloaded = cpickle.load(f)
            break
        except (OSError, IOError):
            # On Windows you can have WindowsError ([Error 5] Access
            # is denied or [Error 13] Permission denied) when reading the file,
            # probably because a writer process has a lock on the file
            time.sleep(0.1)
    else:
        raise
    assert expected == reloaded


def concurrency_safe_write_rename(to_write, filename, write_func):
    temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    concurrency_safe_rename(temporary_filename, filename)


@pytest.mark.thread_unsafe  # https://github.com/joblib/joblib/issues/1816
@timeout(0)  # No timeout as this test can be long
@with_multiprocessing
@parametrize("backend", ["multiprocessing", "loky", "threading"])
def test_concurrency_safe_write(tmpdir, backend):
    # Add one item to cache
    filename = tmpdir.join("test.pkl").strpath

    obj = {str(i): i for i in range(int(1e5))}
    funcs = [
        functools.partial(concurrency_safe_write_rename, write_func=write_func)
        if i % 3 != 2
        else load_func
        for i in range(12)
    ]
    Parallel(n_jobs=2, backend=backend)(delayed(func)(obj, filename) for func in funcs)


def test_store_cached_func_code_is_not_written_in_place(tmpdir):
    # Non-regression test for #1694: func_code.py used to be truncated and
    # rewritten in place, so a concurrent reader could catch it half-written,
    # read that as a changed function and wipe the whole cache directory.
    backend = FileSystemStoreBackend()
    backend.configure(tmpdir.join("store").strpath)

    call_id = ("mod", "func")
    backend.store_cached_func_code(call_id, "# first line: 1\ndef f(): pass\n")
    original = backend.get_cached_func_code(call_id)

    def exploding_open(name, mode):
        open(name, mode).close()  # truncates whatever it is pointed at
        raise RuntimeError("boom")

    old_open = backend._open_item
    backend._open_item = exploding_open
    with pytest.raises(RuntimeError, match="boom"):
        backend.store_cached_func_code(call_id, "# first line: 1\ndef g(): pass\n")

    backend._open_item = old_open
    assert backend.get_cached_func_code(call_id) == original


def test_warning_on_dump_failure(tmpdir):
    # Check that a warning is raised when the dump fails for any reason but
    # a PicklingError.
    class UnpicklableObject(object):
        def __reduce__(self):
            raise RuntimeError("some exception")

    backend = FileSystemStoreBackend()
    backend.location = tmpdir.join("test_warning_on_pickling_error").strpath
    backend.compress = None

    with pytest.warns(CacheWarning, match="some exception"):
        backend.dump_item("testpath", UnpicklableObject())


def test_warning_on_pickling_error(tmpdir):
    # This is separate from test_warning_on_dump_failure because in the
    # future we will turn this into an exception.
    class UnpicklableObject(object):
        def __reduce__(self):
            raise PicklingError("not picklable")

    backend = FileSystemStoreBackend()
    backend.location = tmpdir.join("test_warning_on_pickling_error").strpath
    backend.compress = None

    with pytest.warns(FutureWarning, match="not picklable"):
        backend.dump_item("testpath", UnpicklableObject())
