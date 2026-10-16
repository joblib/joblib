import functools
import json
import os
import pickle as cpickle
import threading
import time
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from pickle import PicklingError

import pytest

from joblib import Memory, Parallel, delayed, numpy_pickle
from joblib._store_backends import (
    CacheWarning,
    FileSystemStoreBackend,
    _old_split_id,
    concurrency_safe_write,
)
from joblib.backports import concurrency_safe_rename
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, timeout, warns


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
        backend.dump_item(("func", "input"), UnpicklableObject())


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
        backend.dump_item(("func", "input"), UnpicklableObject())


def test_cache_tree_versions(tmpdir):
    # Makes a backend using old cache tree
    old_store = FileSystemStoreBackend()
    old_store.configure(tmpdir.strpath)
    old_store._split_id = types.MethodType(_old_split_id, old_store)
    os.remove(os.path.join(old_store.location, "store_backend_info.json"))

    # Cache some items
    funs = ["fun1", "fun2", os.path.join("z" * 32, "fun3")]
    args = ["012" + "3" * 29, "012" + "4" * 29, "abcd" * 8]
    items = [0, "xyz", [42, 6.9]]
    for fun in funs:
        for arg, item in zip(args, items):
            old_store.dump_item((fun, arg), item)

    # Assert store_backend finds items from old cache tree
    old_items = old_store.get_items()
    assert len(old_items) == len(funs) * len(args)
    assert set(item.path for item in old_items) == set(
        tmpdir.join(fun, arg).strpath for fun in funs for arg in args
    )

    # Assert a warning is raised when configuring a FileSystemStoreBackend
    # with an old cache tree
    with warns(UserWarning, match="using old cache storage tree"):
        new_store = FileSystemStoreBackend()
        new_store.configure(tmpdir.strpath)

    # Pollute the backend
    # Creates items with new tree. After the creation:
    # (funs[0], args[0]) old tree version is more recent
    # (funs[1], args[0]) new tree version is more recent
    conflict_item = "bad"
    for fun in funs[:2]:
        arg = args[0]
        conflict_dir = tmpdir.join(fun, arg[:3], arg[3:]).strpath
        os.makedirs(conflict_dir)
        numpy_pickle.dump(conflict_item, os.path.join(conflict_dir, "output.pkl"))
    time.sleep(1.5)
    new_store.dump_item((funs[0], args[0]), items[0])

    # Assert (funs[0], args[0]) old tree version is more recent
    assert os.path.getmtime(
        tmpdir.join(funs[0], arg[:3], arg[3:], "output.pkl").strpath
    ) < os.path.getmtime(tmpdir.join(funs[0], arg, "output.pkl").strpath)

    # Assert update_cache_tree correctly updates the cache tree
    new_store.update_cache_tree()
    new_items = new_store.get_items()
    assert len(new_items) == len(funs) * len(args)
    assert set(item.path for item in new_items) == set(
        tmpdir.join(fun, arg[:3], arg[3:]).strpath for fun in funs for arg in args
    )

    # Assert no warning is raised when caching a function with a new cache tree
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        FileSystemStoreBackend().configure(tmpdir.strpath)
        assert len(ws) == 0

    # Check that old_store shifted to new cache tree
    assert old_store.load_item((funs[0], args[0])) == items[0]
    assert old_store.load_item((funs[1], args[0])) == conflict_item
    with raises(KeyError):
        old_store.load_item(("not_a_function", args[0]))


@parametrize("operation", ["initialize", "migrate_start", "migrate_finish"])
def test_cache_info_is_published_atomically(tmp_path, monkeypatch, operation):
    # Pause a real file write after opening it, while a second Memory opens
    # the same cache. Readers must never see a truncated metadata file.
    import joblib._store_backends as store_backends

    location = tmp_path / "cache"
    info_path = location / "joblib" / "store_backend_info.json"
    if operation != "initialize":
        memory = Memory(str(location), verbose=0)

    opened = threading.Event()
    resume = threading.Event()
    main_thread = threading.current_thread()
    write_count = 0
    target_write = 2 if operation == "migrate_finish" else 1

    def coordinated_open(filename, mode="r", *args, **kwargs):
        nonlocal write_count
        file = open(filename, mode, *args, **kwargs)
        if (
            threading.current_thread() is not main_thread
            and str(filename).startswith(str(info_path))
            and mode == "wb"
        ):
            write_count += 1
            if write_count == target_write:
                opened.set()
                if not resume.wait(timeout=10):
                    file.close()
                    raise TimeoutError("Cache metadata reader did not finish")
        return file

    monkeypatch.setattr(store_backends, "open", coordinated_open, raising=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        if operation == "initialize":
            writer = pool.submit(Memory, str(location), verbose=0)
        else:
            writer = pool.submit(memory.update_cache_tree)
        try:
            assert opened.wait(timeout=10)
            # The migration-finish reader may observe the valid in-progress
            # migration flag; it must still be able to construct the cache.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                reader = Memory(str(location), verbose=0)
        finally:
            resume.set()
        writer.result(timeout=10)

    def sum_values(values):
        return sum(values)

    assert reader.cache(sum_values)([1, 2, 3]) == 6
    assert json.loads(info_path.read_text()) == {
        "cache_version": 2,
        "require_update": False,
    }
    assert list(info_path.parent.glob("store_backend_info.json.*")) == []


def test_invalid_cache_info_is_not_silently_replaced(tmp_path):
    Memory(str(tmp_path), verbose=0)
    info_path = tmp_path / "joblib" / "store_backend_info.json"
    info_path.write_text("invalid json")
    with raises(json.JSONDecodeError):
        Memory(str(tmp_path), verbose=0)
    assert info_path.read_text() == "invalid json"
