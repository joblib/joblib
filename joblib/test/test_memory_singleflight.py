import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from joblib import Memory, Parallel, delayed
from joblib._memory_singleflight import cache_call_lock
from joblib._multiprocessing_helpers import mp


def _record_call(value, marker):
    with open(marker, "a") as output:
        output.write(f"{value}\n")
    time.sleep(0.1)
    return value


@pytest.mark.parametrize("backend", ["threading", "loky"])
def test_singleflight_shared_key(tmp_path, backend):
    if backend == "loky" and mp is None:
        pytest.skip("multiprocessing is disabled or unavailable")
    marker = tmp_path / "calls"
    cached = Memory(tmp_path / "cache", verbose=0).cache(
        _record_call, singleflight=True
    )
    results = Parallel(n_jobs=3, backend=backend)(
        delayed(cached)(7, marker) for _ in range(6)
    )
    assert results == [7] * 6
    assert marker.read_text().splitlines() == ["7"]


def test_singleflight_distinct_keys_run_concurrently(tmp_path):
    barrier = Barrier(2)

    def compute(value):
        barrier.wait(timeout=5)
        return value

    cached = Memory(tmp_path, verbose=0).cache(compute, singleflight=True)
    with ThreadPoolExecutor(2) as executor:
        assert list(executor.map(cached, [1, 2])) == [1, 2]


def test_singleflight_exception_releases_lock(tmp_path):
    def compute(value):
        raise RuntimeError("expected")

    cached = Memory(tmp_path, verbose=0).cache(compute, singleflight=True)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="expected"):
            cached(1)


def test_singleflight_shelving_and_pickle(tmp_path):
    cached = Memory(tmp_path / "cache", verbose=0).cache(
        _record_call, singleflight=True
    )
    marker = tmp_path / "calls"
    assert cached.call_and_shelve(5, marker).get() == 5
    restored = pickle.loads(pickle.dumps(cached))
    assert restored(5, marker) == 5
    assert marker.read_text().splitlines() == ["5"]


def _hold_lock(location, ready):
    with cache_call_lock(location, "function", "arguments"):
        ready.set()
        time.sleep(60)


def _acquire_after_crash(location, acquired):
    with cache_call_lock(location, "function", "arguments"):
        acquired.set()


@pytest.mark.skipif(mp is None, reason="multiprocessing is disabled or unavailable")
def test_singleflight_process_death_releases_lock(tmp_path):
    ctx = mp.get_context("spawn")
    ready, acquired = ctx.Event(), ctx.Event()
    holder = ctx.Process(target=_hold_lock, args=(str(tmp_path), ready))
    waiter = ctx.Process(target=_acquire_after_crash, args=(str(tmp_path), acquired))
    holder.start()
    try:
        assert ready.wait(10)
        waiter.start()
        assert not acquired.wait(0.2)
        holder.terminate()
        holder.join(10)
        assert acquired.wait(10)
        waiter.join(10)
        assert waiter.exitcode == 0
    finally:
        for process in (holder, waiter):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(10)


def test_singleflight_rejects_async(tmp_path):
    async def compute():
        return 1

    with pytest.raises(ValueError, match="synchronous"):
        Memory(tmp_path).cache(compute, singleflight=True)


def test_singleflight_requires_location():
    with pytest.raises(ValueError, match="location"):
        Memory(None).cache(_record_call, singleflight=True)


def test_singleflight_requires_boolean(tmp_path):
    with pytest.raises(TypeError, match="boolean"):
        Memory(tmp_path).cache(_record_call, singleflight="yes")


def test_singleflight_same_key_recursion_does_not_deadlock(tmp_path):
    def compute(value):
        return cached(value)

    cached = Memory(tmp_path, verbose=0).cache(compute, singleflight=True)
    with pytest.raises(RuntimeError, match="Recursive singleflight"):
        cached(1)
