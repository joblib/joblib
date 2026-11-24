"""
Test the shelf module.
"""

import os
import shutil
import subprocess
import sys
import time
from random import random

import joblib
from joblib import Parallel, Shelf, clear_shelf, delayed, shelve
from joblib.shelf import ShelfFuture, _futures
from joblib.testing import parametrize, raises


@parametrize("data", [42, "text", ["pi", 3.14, None], {"a": 0, 1: "b"}])
def test_shelve(data):
    future = shelve(data)
    assert future.result() == data
    id = (future.location, future.id)
    assert id in _futures
    assert _futures[id] == data


def test_bad_shelf_access():
    x, y = 42, 69
    sx, sy = map(shelve, (x, y))
    assert sx.location == sy.location
    assert sx.id != sy.id
    for id in "abc":
        if id != sx.id and id != sy.id:
            break
    for loc in "ab":
        if loc != sx.location:
            break
    with raises(KeyError, match="Non-existing item"):
        ShelfFuture(sx.location, id).result()
    with raises(KeyError, match="Non-existing item"):
        ShelfFuture(loc, sx.id).result()


def test_shelve_parallel():
    N, R = 100, 40
    S = N - R + 1
    data = [random() for _ in range(N)]
    shelved_data = shelve(data)

    def f(data, i):
        return sum(data.result()[i : i + R])

    expected = [sum(data[i : i + R]) for i in range(S)]
    out = Parallel(n_jobs=4)(delayed(f)(shelved_data, i) for i in range(S))
    assert out == expected
    clear_shelf()
    assert (
        len(os.listdir(joblib.shelf._shelf.store_backend.location)) == 1
    )  # contains .gitignore


def test_shelf(tmpdir):
    def core(shelf):
        shelf_location = shelf.store_backend.location
        assert tmpdir.strpath == shelf_location
        x, y = 42, 69
        assert len(os.listdir(shelf_location)) == 1  # contains .gitignore
        shelved_x = shelf.shelve(x)
        shelf.shelve(y)
        assert len(os.listdir(shelf_location)) == 3
        shelved_x.clear()
        assert len(os.listdir(shelf_location)) == 2
        shelved_x.clear()
        assert len(os.listdir(shelf_location)) == 2
        with raises(KeyError, match="Non-existing item"):
            shelved_x.result()
        shelf.clear()
        assert len(os.listdir(shelf_location)) == 1
        shelf.close()
        assert not os.path.exists(shelf_location)
        with raises(RuntimeError, match="already closed shelf"):
            shelf.shelve("abc")

    shelf = Shelf(tmpdir.strpath)
    core(shelf)
    shelf.close()
    assert not os.path.exists(tmpdir.strpath)
    with raises(RuntimeError, match="already closed shelf"):
        shelf.shelve("abc")

    # Testing as context
    with Shelf(tmpdir.strpath) as shelf:
        core(shelf)
    assert not os.path.exists(tmpdir.strpath)


def test_shelf_kill():
    # Check that the shelf is deleted when the process is killed
    cmd = """if 1:
    import joblib
    import time
    x = 42
    sx = joblib.shelve(x)
    print("shelved", joblib.shelf._shelf.store_backend.location, flush=True)
    time.sleep(60)
    """
    p = subprocess.Popen([sys.executable, "-c", cmd], stdout=subprocess.PIPE, text=True)
    for line in p.stdout:
        start = "shelved "
        assert line.startswith(start)
        path = line[len(start) : -1]
        p.kill()
        for _ in range(20):
            if not os.path.exists(path):
                break
            time.sleep(0.2)
        if os.path.exists(path):
            shutil.rmtree(path)
            assert False, "Shelf folder not deleted after process kill"
