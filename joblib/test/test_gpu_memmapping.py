"""Tests for sharing GPU arrays across processes via CUDA IPC.

The tests in this file are split in two groups:

- Hardware-independent unit tests for the sharing-mode resolution, the reducer
  registration and the producer-side resource manager. These run everywhere,
  including CI without a GPU.
- Hardware-gated integration tests (marked with ``with_torch_cuda`` /
  ``with_cupy``) that actually share GPU arrays with worker processes and check
  that the sharing is zero-copy.
"""

import gc
import warnings
import weakref

import pytest

from joblib import Parallel, delayed
from joblib._gpu_array_reducer import (
    CupyIpcForwardReducer,
    GpuResourcesManager,
    _rebuild_cupy_ipc_array,
    _resolve_share_gpu_arrays,
    check_share_gpu_arrays,
    get_gpu_reducers,
)
from joblib._multiprocessing_helpers import mp
from joblib.test.common import (
    with_cupy,
    with_multiprocessing,
    with_torch_cuda,
)
from joblib.testing import parametrize, raises, warns

_START_METHODS = mp.get_all_start_methods() if mp is not None else []

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy
except ImportError:
    cupy = None


###############################################################################
# Hardware-independent unit tests


@parametrize("mode", ["auto", "on", "off"])
def test_check_share_gpu_arrays_valid(mode):
    assert check_share_gpu_arrays(mode) == mode


def test_check_share_gpu_arrays_invalid():
    with raises(ValueError, match="share_gpu_arrays"):
        check_share_gpu_arrays("bogus")


@parametrize("mode", ["auto", "on", "off"])
def test_resolve_spawn_is_identity(mode):
    # With a safe start method, the mode is preserved.
    assert _resolve_share_gpu_arrays(mode, start_method="spawn") == mode
    assert _resolve_share_gpu_arrays(mode, start_method=None) == mode


def test_resolve_off_with_fork_is_silent():
    # "off" never warns, regardless of the start method.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_share_gpu_arrays("off", start_method="fork") == "off"


def test_resolve_auto_with_fork_falls_back_with_warning():
    with warns(UserWarning, match="fork"):
        assert _resolve_share_gpu_arrays("auto", start_method="fork") == "off"


def test_resolve_on_with_fork_raises():
    with raises(ValueError, match="fork"):
        _resolve_share_gpu_arrays("on", start_method="fork")


def test_get_gpu_reducers_off_registers_nothing():
    reducers = get_gpu_reducers("off", start_method="spawn")
    assert reducers == {}


def test_get_gpu_reducers_fork_on_raises():
    with raises(ValueError, match="fork"):
        get_gpu_reducers("on", start_method="fork")


def test_get_gpu_reducers_fork_auto_registers_nothing():
    with warns(UserWarning, match="fork"):
        reducers = get_gpu_reducers("auto", start_method="fork")
    assert reducers == {}


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_get_gpu_reducers_registers_torch():
    reducers = get_gpu_reducers("auto", start_method="spawn")
    assert torch.Tensor in reducers


@pytest.mark.skipif(cupy is None, reason="requires cupy")
def test_get_gpu_reducers_registers_cupy():
    reducers = get_gpu_reducers("auto", start_method="spawn")
    assert cupy.ndarray in reducers


def test_get_gpu_reducers_updates_existing_dict():
    existing = {int: "sentinel"}
    reducers = get_gpu_reducers("auto", start_method="spawn", forward_reducers=existing)
    assert reducers is existing
    assert existing[int] == "sentinel"


def test_gpu_resources_manager_anchor_and_clean():
    manager = GpuResourcesManager()
    manager.set_current_context("ctx-a")

    class _Dummy:
        pass

    a = _Dummy()
    manager.anchor(a)
    ref = weakref.ref(a)
    del a

    manager.set_current_context("ctx-b")
    b = _Dummy()
    manager.anchor(b)

    # The anchor for ctx-a keeps the object alive even after dropping our ref.
    gc.collect()
    assert ref() is not None

    # Cleaning ctx-a releases its anchors; ctx-b is untouched.
    manager.clean("ctx-a")
    gc.collect()
    assert ref() is None
    assert manager._anchors.get("ctx-b") == [b]

    manager.clean()
    assert manager._anchors == {}


###############################################################################
# Hardware-gated integration tests: PyTorch


def _torch_probe_and_add(tensor, value):
    """Worker function: report tensor info and mutate it in place."""
    info = {
        "is_cuda": bool(tensor.is_cuda),
        "device": int(tensor.device.index) if tensor.is_cuda else -1,
    }
    if tensor.is_cuda:
        tensor.add_(value)
        torch.cuda.synchronize()
    return info


@with_multiprocessing
@with_torch_cuda
def test_torch_cuda_tensor_shared_in_place():
    # 512x512 float32 = 1 MiB > default max_nbytes (1e6), 'on' forces sharing.
    x = torch.zeros(512, 512, device="cuda:0")
    out = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")(
        delayed(_torch_probe_and_add)(x, 1.0) for _ in range(1)
    )
    torch.cuda.synchronize()

    assert out[0]["is_cuda"]
    assert out[0]["device"] == 0
    # Zero-copy sharing: the worker's in-place mutation is visible in the
    # parent process (a host copy would not propagate back).
    assert float(x.sum().item()) == x.numel()


@with_multiprocessing
@with_torch_cuda
def test_torch_cuda_tensor_not_shared_when_off():
    x = torch.zeros(512, 512, device="cuda:0")
    out = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="off")(
        delayed(_torch_probe_and_add)(x, 1.0) for _ in range(1)
    )
    torch.cuda.synchronize()

    # The worker still receives a CUDA tensor, but via a (host round-trip) copy.
    assert out[0]["is_cuda"]
    assert float(x.sum().item()) == 0.0


@with_multiprocessing
@with_torch_cuda
def test_torch_cuda_auto_threshold():
    # Large array (> max_nbytes) is shared in auto mode.
    big = torch.zeros(512, 512, device="cuda:0")
    Parallel(n_jobs=2, backend="loky", share_gpu_arrays="auto", max_nbytes=1000)(
        delayed(_torch_probe_and_add)(big, 1.0) for _ in range(1)
    )
    torch.cuda.synchronize()
    assert float(big.sum().item()) == big.numel()

    # Small array (<= max_nbytes) is not shared in auto mode.
    small = torch.zeros(8, device="cuda:0")
    Parallel(
        n_jobs=2, backend="loky", share_gpu_arrays="auto", max_nbytes=10_000_000
    )(delayed(_torch_probe_and_add)(small, 1.0) for _ in range(1))
    torch.cuda.synchronize()
    assert float(small.sum().item()) == 0.0


@with_multiprocessing
@with_torch_cuda
def test_torch_cpu_tensor_falls_back():
    # CPU tensors are never IPC-shared, even in 'on' mode; they are passed
    # through regular pickling and not mutated in the parent.
    x = torch.zeros(512, 512)
    out = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")(
        delayed(_torch_probe_and_add)(x, 1.0) for _ in range(1)
    )
    assert not out[0]["is_cuda"]
    assert float(x.sum().item()) == 0.0


###############################################################################
# Hardware-gated integration tests: CuPy


def _cupy_probe_and_add(array, value):
    """Worker function: report array info and mutate it in place."""
    info = {
        "device": int(array.device.id),
        "shape": tuple(array.shape),
        "dtype": array.dtype.str,
    }
    array += value
    cupy.cuda.Stream.null.synchronize()
    return info


@with_multiprocessing
@with_cupy
def test_cupy_array_shared_in_place():
    with cupy.cuda.Device(0):
        x = cupy.zeros((512, 512), dtype=cupy.float32)  # 1 MiB
    out = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")(
        delayed(_cupy_probe_and_add)(x, 1.0) for _ in range(1)
    )
    cupy.cuda.Device(0).synchronize()

    assert out[0]["device"] == 0
    # Zero-copy sharing: the worker's in-place mutation is visible in the parent.
    assert float(x.sum().get()) == x.size


@with_multiprocessing
@with_cupy
def test_cupy_array_not_shared_when_off():
    with cupy.cuda.Device(0):
        x = cupy.zeros((512, 512), dtype=cupy.float32)
    out = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="off")(
        delayed(_cupy_probe_and_add)(x, 1.0) for _ in range(1)
    )
    cupy.cuda.Device(0).synchronize()

    assert out[0]["device"] == 0
    assert float(x.sum().get()) == 0.0


@with_multiprocessing
@with_cupy
def test_cupy_auto_threshold():
    with cupy.cuda.Device(0):
        big = cupy.zeros((512, 512), dtype=cupy.float32)
    Parallel(n_jobs=2, backend="loky", share_gpu_arrays="auto", max_nbytes=1000)(
        delayed(_cupy_probe_and_add)(big, 1.0) for _ in range(1)
    )
    cupy.cuda.Device(0).synchronize()
    assert float(big.sum().get()) == big.size

    with cupy.cuda.Device(0):
        small = cupy.zeros((8,), dtype=cupy.float32)
    Parallel(
        n_jobs=2, backend="loky", share_gpu_arrays="auto", max_nbytes=10_000_000
    )(delayed(_cupy_probe_and_add)(small, 1.0) for _ in range(1))
    cupy.cuda.Device(0).synchronize()
    assert float(small.sum().get()) == 0.0


###############################################################################
# Hardware-gated integration tests: CuPy producer-side lifetime


@with_cupy
def test_cupy_reducer_anchors_source_memory():
    manager = GpuResourcesManager(context_id="ctx")
    reducer = CupyIpcForwardReducer("on", max_nbytes=0, resources_manager=manager)

    with cupy.cuda.Device(0):
        x = cupy.ones((256, 256), dtype=cupy.float32)

    rebuild, args = reducer(x)

    # The reducer produced an IPC payload and anchored the source allocation
    # under the current context so it stays alive while workers map it.
    assert rebuild is _rebuild_cupy_ipc_array
    assert manager._anchors["ctx"], "source memory was not anchored"

    # Cleaning the context releases the producer-side keep-alive.
    manager.clean("ctx")
    assert "ctx" not in manager._anchors


@with_multiprocessing
@with_cupy
def test_cupy_sharing_repeated_calls():
    # Repeated calls reuse the loky executor; the per-call anchoring and
    # cleanup must keep working without leaking handles or stale allocations.
    for _ in range(3):
        with cupy.cuda.Device(0):
            x = cupy.zeros((512, 512), dtype=cupy.float32)
        Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")(
            delayed(_cupy_probe_and_add)(x, 1.0) for _ in range(1)
        )
        cupy.cuda.Device(0).synchronize()
        assert float(x.sum().get()) == x.size
        del x


@with_multiprocessing
@with_cupy
def test_cupy_anchors_released_after_call():
    with cupy.cuda.Device(0):
        x = cupy.ones((512, 512), dtype=cupy.float32)

    p = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")
    p(delayed(_cupy_probe_and_add)(x, 0.0) for _ in range(1))

    # Recover the (reused) executor's GPU manager and check the anchors created
    # for this Parallel call were released on termination.
    probe = Parallel(n_jobs=2, backend="loky", share_gpu_arrays="on")
    probe._initialize_backend()
    try:
        gpu_manager = probe._backend._workers._gpu_resources_manager
        assert p._id not in gpu_manager._anchors
    finally:
        probe._backend.terminate()


###############################################################################
# Cross-backend behavior


@with_multiprocessing
@with_cupy
def test_multiprocessing_spawn_shares():
    spawn_ctx = mp.get_context("spawn")
    with cupy.cuda.Device(0):
        x = cupy.zeros((512, 512), dtype=cupy.float32)
    Parallel(n_jobs=2, backend=spawn_ctx, share_gpu_arrays="on")(
        delayed(_cupy_probe_and_add)(x, 1.0) for _ in range(1)
    )
    cupy.cuda.Device(0).synchronize()
    assert float(x.sum().get()) == x.size


@with_multiprocessing
@pytest.mark.skipif(
    "fork" not in _START_METHODS,
    reason="requires the fork start method",
)
def test_multiprocessing_fork_on_raises():
    # 'on' mode must refuse to run on a fork-based context (CUDA cannot survive
    # a fork). This does not require a GPU: the error is raised when building
    # the pool, before any worker is forked.
    fork_ctx = mp.get_context("fork")
    with raises(ValueError, match="fork"):
        Parallel(n_jobs=2, backend=fork_ctx, share_gpu_arrays="on")(
            delayed(abs)(i) for i in range(2)
        )


@with_multiprocessing
@pytest.mark.skipif(
    "fork" not in _START_METHODS,
    reason="requires the fork start method",
)
def test_multiprocessing_fork_auto_falls_back_with_warning():
    fork_ctx = mp.get_context("fork")
    with warns(UserWarning, match="fork"):
        out = Parallel(n_jobs=2, backend=fork_ctx, share_gpu_arrays="auto")(
            delayed(abs)(i) for i in range(-3, 0)
        )
    assert out == [3, 2, 1]


def test_threading_ignores_share_gpu_arrays():
    # Threads share memory directly, so GPU sharing is a transparent no-op and
    # must not interfere with the threading backend.
    out = Parallel(n_jobs=2, backend="threading", share_gpu_arrays="on")(
        delayed(abs)(i) for i in range(-3, 0)
    )
    assert out == [3, 2, 1]


def test_sequential_ignores_share_gpu_arrays():
    out = Parallel(n_jobs=1, share_gpu_arrays="on")(
        delayed(abs)(i) for i in range(-3, 0)
    )
    assert out == [3, 2, 1]
