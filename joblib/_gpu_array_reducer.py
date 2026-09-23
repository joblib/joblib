"""Reducers for sharing GPU arrays across processes via CUDA IPC.

This module provides forward (parent -> worker) pickling reducers that share
GPU arrays (PyTorch tensors and CuPy arrays) between processes without copying
their content through host memory, in the same spirit as the numpy memmapping
reducers in :mod:`joblib._memmapping_reducer`.

Instead of dumping arrays to a file in shared memory, the device memory is
shared through CUDA inter-process communication (IPC) handles:

- For PyTorch, this reuses :func:`torch.multiprocessing.reductions.reduce_tensor`
  which already implements correct CUDA IPC sharing, including the producer
  keep-alive and the synchronization event handle.
- For CuPy, a thin reducer is implemented on top of the low-level
  ``cupy.cuda.runtime.ipc*`` primitives. The producer keeps the source
  allocation alive for the duration of the call (see :class:`GpuResourcesManager`)
  and each worker closes the IPC handle when the reconstructed array is garbage
  collected.

Sharing is forward-only: results returned by workers fall back to the regular
(host round-trip) pickling of their framework.

This module is intentionally lightweight to import: it never imports ``torch``
or ``cupy`` at module load time. A GPU array can only exist if its framework is
already imported in the current process, so the frameworks are looked up in
``sys.modules`` rather than imported eagerly. This keeps ``import joblib`` cheap
and avoids importing heavy GPU libraries for users that do not use this feature.
"""

# Author: joblib developers
# License: BSD 3 clause

import sys
import warnings
import weakref
from pickle import HIGHEST_PROTOCOL, dumps, loads

# Valid values for the ``share_gpu_arrays`` parameter.
#
# - "auto": best effort. Share device arrays above ``max_nbytes`` when a CUDA
#   IPC-capable backend is used; fall back to regular pickling otherwise (e.g.
#   no GPU framework, fork start method, sharing failure). The fallback is
#   silent, except that an incompatible start method is reported when a GPU
#   framework is actually in use.
# - "on": force sharing. Share every device array regardless of ``max_nbytes``
#   and raise a clear error if sharing is requested but not feasible.
# - "off": never share. All arrays use the regular pickling of their framework.
VALID_SHARE_GPU_ARRAYS = ("auto", "on", "off")

# Internal marker for "sharing was requested in 'auto' mode but the backend uses
# the 'fork' start method". It behaves like "off" but lets get_gpu_reducers()
# warn the users that a GPU framework is in use. It is deliberately not part of
# VALID_SHARE_GPU_ARRAYS: users must not be able to pass it.
_FORK_FALLBACK = "_fork_fallback"


def check_share_gpu_arrays(value):
    """Validate the value of the ``share_gpu_arrays`` parameter."""
    if value not in VALID_SHARE_GPU_ARRAYS:
        raise ValueError(
            "share_gpu_arrays should be one of "
            f"{VALID_SHARE_GPU_ARRAYS}, got {value!r} instead."
        )
    return value


def _resolve_share_gpu_arrays(share_gpu_arrays, start_method=None):
    """Resolve the effective sharing mode given the backend start method.

    CUDA state cannot survive a ``fork``, so GPU array sharing is not supported
    with the ``fork`` start method. In ``"on"`` mode we raise, while in
    ``"auto"`` mode we return the internal ``_FORK_FALLBACK`` marker: sharing is
    disabled, but :func:`get_gpu_reducers` still needs to know that it was asked
    for so that it can warn the users that are actually affected (see there).

    The loky backend (and ``spawn`` / ``forkserver`` multiprocessing contexts)
    pass ``start_method`` values that are considered safe.
    """
    if share_gpu_arrays == _FORK_FALLBACK:
        # Already resolved: the multiprocessing backend resolves the mode once
        # before building the pool (to raise early in "on" mode) and the pool
        # resolves it again when registering the reducers.
        return _FORK_FALLBACK

    mode = check_share_gpu_arrays(share_gpu_arrays)
    if mode == "off":
        return "off"

    if start_method == "fork":
        if mode == "on":
            raise ValueError(
                "share_gpu_arrays='on' is not supported with the 'fork' start "
                "method because CUDA state cannot survive a fork. Use the loky "
                "backend or a multiprocessing context using the 'spawn' or "
                "'forkserver' start method."
            )
        return _FORK_FALLBACK

    return mode


def _gpu_framework_in_use():
    """Whether a GPU array framework is actually in use in this process.

    Used to decide whether a user is affected by a limitation of the GPU sharing
    machinery, and therefore worth warning. Frameworks are looked up in
    ``sys.modules`` and never imported here: a GPU array cannot exist unless its
    framework is already imported.
    """
    # Being imported is not enough: both frameworks are routinely imported
    # without any device memory ever being allocated, by any library that merely
    # supports them (including joblib's own test helpers). Look for an actually
    # live CUDA context instead. Every check is best effort: "auto" is a
    # best-effort mode, so staying quiet is the safe failure direction.
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            # Creating any CUDA tensor triggers torch's lazy CUDA init.
            if torch.cuda.is_initialized():
                return True
        except Exception:
            pass

    cupy = sys.modules.get("cupy")
    if cupy is not None:
        try:
            # CuPy arrays are allocated through the default memory pool, so a
            # non-zero usage means device arrays are alive right now.
            if cupy.get_default_memory_pool().used_bytes() > 0:
                return True
        except Exception:
            pass

    return False


def _regular_pickle_reduction(obj):
    """Reduction that falls back to the framework's regular pickling.

    For GPU arrays this implies a host round-trip, which preserves the array
    type but does not share memory across processes.
    """
    return (loads, (dumps(obj, protocol=HIGHEST_PROTOCOL),))


class GpuResourcesManager:
    """Keep producer-side references to shared GPU allocations alive.

    CUDA IPC requires the producer's allocation to remain valid while a consumer
    (worker) process has it mapped. This manager anchors the source memory
    objects, keyed by the id of the :class:`~joblib.Parallel` call that shared
    them, and releases them when that call terminates.

    The current context is set while pickling the batched calls of a given
    Parallel object (see the reducer callback in ``joblib.parallel``), mirroring
    the per-context behavior of
    :class:`joblib._memmapping_reducer.TemporaryResourcesManager`.
    """

    def __init__(self, context_id=None):
        self._anchors = {}
        self._current_context_id = context_id
        if context_id is not None:
            self._anchors[context_id] = []

    def set_current_context(self, context_id):
        self._current_context_id = context_id
        self._anchors.setdefault(context_id, [])

    def anchor(self, obj):
        """Keep a strong reference to ``obj`` for the current context."""
        self._anchors.setdefault(self._current_context_id, []).append(obj)

    def clean(self, context_id=None):
        """Release the anchored allocations for ``context_id`` (or all)."""
        if context_id is None:
            self._anchors.clear()
        else:
            self._anchors.pop(context_id, None)


class TorchIpcForwardReducer:
    """Forward reducer sharing torch CUDA tensors via CUDA IPC.

    CPU tensors and CUDA tensors below the ``max_nbytes`` threshold (in
    ``"auto"`` mode) fall back to torch's regular pickling.
    """

    def __init__(self, mode, max_nbytes):
        self._mode = mode
        self._max_nbytes = max_nbytes

    def __call__(self, tensor):
        is_cuda = bool(getattr(tensor, "is_cuda", False))

        share = False
        if is_cuda:
            if self._mode == "on":
                share = True
            elif self._max_nbytes is not None:
                nbytes = tensor.element_size() * tensor.nelement()
                share = nbytes > self._max_nbytes

        if share:
            try:
                from torch.multiprocessing.reductions import reduce_tensor

                # detach() shares the storage (no copy) while dropping autograd
                # information so the shared tensor does not require grad.
                return reduce_tensor(tensor.detach())
            except Exception as e:
                if self._mode == "on":
                    raise RuntimeError(
                        f"Failed to share torch CUDA tensor via CUDA IPC: {e}"
                    ) from e
                # "auto": fall back to regular pickling below.

        return _regular_pickle_reduction(tensor)


def _close_cupy_ipc_handle(base_ptr, device_id):
    """Close a CuPy CUDA IPC handle, ignoring teardown-time errors."""
    cupy = sys.modules.get("cupy")
    if cupy is None:  # pragma: no cover - cupy always present in the worker
        return
    try:
        with cupy.cuda.Device(device_id):
            cupy.cuda.runtime.ipcCloseMemHandle(base_ptr)
    except Exception:  # pragma: no cover - best effort cleanup
        pass


class _CupyIpcMemoryOwner:
    """Owner object whose finalizer closes the IPC handle on GC.

    A reconstructed CuPy array keeps a reference to this owner through its
    ``UnownedMemory``. When the array (and hence this owner) is garbage
    collected in the worker, the IPC handle is closed exactly once.
    """

    def __init__(self, base_ptr, device_id):
        self.base_ptr = base_ptr
        self.device_id = device_id
        weakref.finalize(self, _close_cupy_ipc_handle, base_ptr, device_id)


def _rebuild_cupy_ipc_array(payload):
    """Reconstruct a CuPy array from a CUDA IPC payload in a worker process."""
    # cupy is imported lazily (and not at module import time) to keep
    # ``import joblib`` lightweight. This runs in a worker process where the
    # array is being unpickled, so cupy is necessarily available.
    import cupy

    device_id = payload["device_id"]
    with cupy.cuda.Device(device_id):
        base_ptr = cupy.cuda.runtime.ipcOpenMemHandle(payload["handle"])
        owner = _CupyIpcMemoryOwner(base_ptr, device_id)
        mem = cupy.cuda.UnownedMemory(
            base_ptr, payload["base_nbytes"], owner, device_id
        )
        memptr = cupy.cuda.MemoryPointer(mem, payload["offset"])
        # cupy.ndarray accepts the dtype string directly, so numpy is not needed.
        array = cupy.ndarray(
            payload["shape"],
            dtype=payload["dtype"],
            memptr=memptr,
            strides=payload["strides"],
        )
    return array


class CupyIpcForwardReducer:
    """Forward reducer sharing CuPy arrays via CUDA IPC.

    CuPy arrays always live on a device, so every array is a sharing candidate.
    In ``"auto"`` mode, arrays at or below ``max_nbytes`` fall back to CuPy's
    regular (host round-trip) pickling.
    """

    def __init__(self, mode, max_nbytes, resources_manager=None):
        self._mode = mode
        self._max_nbytes = max_nbytes
        self._resources_manager = resources_manager

    def __call__(self, array):
        share = self._mode == "on" or (
            self._max_nbytes is not None and array.nbytes > self._max_nbytes
        )

        if share:
            try:
                return self._reduce_ipc(array)
            except Exception as e:
                if self._mode == "on":
                    raise RuntimeError(
                        f"Failed to share CuPy array via CUDA IPC: {e}"
                    ) from e
                # "auto": fall back to regular pickling below.

        return _regular_pickle_reduction(array)

    def _reduce_ipc(self, array):
        import cupy

        device_id = int(array.device.id)
        mem = array.data.mem
        base_ptr = int(mem.ptr)

        with cupy.cuda.Device(device_id):
            # Ensure all pending writes to the array are complete before the
            # workers can read the shared memory (forward-only, read sharing).
            cupy.cuda.Stream.null.synchronize()
            handle = cupy.cuda.runtime.ipcGetMemHandle(base_ptr)

        # Offset of the array data within the base allocation that the IPC
        # handle refers to.
        offset = int(array.data.ptr) - base_ptr

        # Keep the source allocation alive on the producer side until the
        # Parallel call that shared it terminates.
        if self._resources_manager is not None:
            self._resources_manager.anchor(mem)

        payload = {
            "handle": handle,
            "offset": offset,
            "base_nbytes": int(mem.size),
            "shape": tuple(array.shape),
            "strides": tuple(array.strides),
            "dtype": array.dtype.str,
            "device_id": device_id,
        }
        return (_rebuild_cupy_ipc_array, (payload,))


def get_gpu_reducers(
    share_gpu_arrays="auto",
    max_nbytes=1e6,
    start_method=None,
    forward_reducers=None,
    resources_manager=None,
    **kwargs,
):
    """Build forward reducers sharing GPU arrays via CUDA IPC.

    Parameters
    ----------
    share_gpu_arrays : {"auto", "on", "off"}, default="auto"
        Sharing mode, see :data:`VALID_SHARE_GPU_ARRAYS`.
    max_nbytes : int or None, default=1e6
        Threshold (in bytes) above which device arrays are shared in ``"auto"``
        mode. Ignored in ``"on"`` mode (all device arrays are shared).
    start_method : str or None, default=None
        Start method of the target process backend. ``"fork"`` disables GPU
        sharing (CUDA cannot survive a fork). ``None`` (loky) and the other
        multiprocessing start methods are considered safe.
    forward_reducers : dict or None, default=None
        An existing reducers dict to update in place. A new one is created if
        not provided.
    resources_manager : GpuResourcesManager or None, default=None
        Manager keeping shared CuPy allocations alive on the producer side.

    Returns
    -------
    forward_reducers : dict
        Mapping of ``type -> reducer`` to register on the forward queue pickler.
        Only frameworks that are already imported in the current process get a
        reducer registered.
    """
    if forward_reducers is None:
        forward_reducers = dict()

    mode = _resolve_share_gpu_arrays(share_gpu_arrays, start_method)
    if mode == "off":
        return forward_reducers

    # A GPU array can only exist if its framework is already imported, so we
    # never import torch / cupy here to keep this code path lightweight.
    torch = sys.modules.get("torch")
    cupy = sys.modules.get("cupy")

    if mode == _FORK_FALLBACK:
        # Sharing was requested in "auto" mode but the start method is 'fork'.
        # Only tell the users that can actually be affected: with no GPU
        # framework in use there is no GPU array to share, so the fallback is
        # the documented silent no-op of the "auto" mode. Warning
        # unconditionally here would fire on every multiprocessing Parallel call
        # on the platforms whose default start method is 'fork'.
        if _gpu_framework_in_use():
            warnings.warn(
                "GPU array sharing is disabled because the 'fork' start method "
                "is not compatible with CUDA. Falling back to regular pickling. "
                "Use the loky backend or a 'spawn' multiprocessing context to "
                "enable zero-copy GPU array sharing.",
                UserWarning,
                stacklevel=2,
            )
        return forward_reducers

    if torch is not None:
        forward_reducers[torch.Tensor] = TorchIpcForwardReducer(mode, max_nbytes)

    if cupy is not None:
        forward_reducers[cupy.ndarray] = CupyIpcForwardReducer(
            mode, max_nbytes, resources_manager
        )

    return forward_reducers
