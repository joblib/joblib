..
    For doctests:

    >>> import sys
    >>> setup = getfixture('parallel_numpy_fixture')
    >>> fixture = setup(sys.modules[__name__])

By default the workers of the pool are real Python processes forked using the
``multiprocessing`` module of the Python standard library when ``n_jobs != 1``.
The arguments passed as input to the ``Parallel`` call are serialized and
reallocated in the memory of each worker process.

This can be problematic for large arguments as they will be reallocated
``n_jobs`` times by the workers.

As this problem can often occur in scientific computing with ``numpy``
based datastructures, :class:`joblib.Parallel` provides a special
handling for large arrays to automatically dump them on the filesystem
and pass a reference to the worker to open them as memory map
on that file using the ``numpy.memmap`` subclass of ``numpy.ndarray``.
This makes it possible to share a segment of data between all the
worker processes.

.. note::

  The following only applies with the ``"loky"`` and
  ``'multiprocessing'`` process-backends. If your code can release the
  GIL, then using a thread-based backend by passing
  ``prefer='threads'`` is even more efficient because it makes it
  possible to avoid the communication overhead of process-based
  parallelism.

  Scientific Python libraries such as numpy, scipy, pandas and
  scikit-learn often release the GIL in performance critical code paths.
  It is therefore advised to always measure the speed of thread-based
  parallelism and use it when the scalability is not limited by the GIL.


Automated array to memmap conversion
------------------------------------

The automated array to memmap conversion is triggered by a configurable
threshold on the size of the array::

  >>> import numpy as np
  >>> from joblib import Parallel, delayed
  >>> def is_memmap(obj):
  ...     return isinstance(obj, np.memmap)

  >>> Parallel(n_jobs=2, max_nbytes=1e6)(
  ...     delayed(is_memmap)(np.ones(int(i)))
  ...     for i in [1e2, 1e4, 1e6])
  [False, False, True]

By default the data is dumped to the ``/dev/shm`` shared-memory partition if it
exists and is writable (typically the case under Linux). Otherwise the
operating system's temporary folder is used. The location of the temporary data
files can be customized by passing a ``temp_folder`` argument to the
``Parallel`` constructor.

Passing ``max_nbytes=None`` makes it possible to disable the automated array to
memmap conversion.


Manual management of memmapped input data
-----------------------------------------

For even finer tuning of the memory usage it is also possible to
dump the array as a memmap directly from the parent process to
free the memory before forking the worker processes. For instance
let's allocate a large array in the memory of the parent process::

  >>> large_array = np.ones(int(1e6))

Dump it to a local file for memmapping::

  >>> import tempfile
  >>> import os
  >>> from joblib import load, dump

  >>> temp_folder = tempfile.mkdtemp()
  >>> filename = os.path.join(temp_folder, 'joblib_test.mmap')
  >>> if os.path.exists(filename): os.unlink(filename)
  >>> _ = dump(large_array, filename)
  >>> large_memmap = load(filename, mmap_mode='r+')

The ``large_memmap`` variable is pointing to a ``numpy.memmap``
instance::

  >>> large_memmap.__class__.__name__, large_array.nbytes, large_array.shape
  ('memmap', 8000000, (1000000,))

  >>> np.allclose(large_array, large_memmap)
  True

The original array can be freed from the main process memory::

  >>> del large_array
  >>> import gc
  >>> _ = gc.collect()

It is possible to slice ``large_memmap`` into a smaller memmap::

  >>> small_memmap = large_memmap[2:5]
  >>> small_memmap.__class__.__name__, small_memmap.nbytes, small_memmap.shape
  ('memmap', 24, (3,))

Finally a ``np.ndarray`` view backed on that same memory mapped file can be
used::

  >>> small_array = np.asarray(small_memmap)
  >>> small_array.__class__.__name__, small_array.nbytes, small_array.shape
  ('ndarray', 24, (3,))

All those three datastructures point to the same memory buffer and
this same buffer will also be reused directly by the worker processes
of a ``Parallel`` call::

  >>> Parallel(n_jobs=2, max_nbytes=None)(
  ...     delayed(is_memmap)(a)
  ...     for a in [large_memmap, small_memmap, small_array])
  [True, True, True]

Note that here ``max_nbytes=None`` is used to disable the auto-dumping
feature of ``Parallel``. ``small_array`` is still in shared memory in the
worker processes because it was already backed by shared memory in the
parent process.
The pickling machinery of ``Parallel`` multiprocessing queues are
able to detect this situation and optimize it on the fly to limit
the number of memory copies.


Writing parallel computation results in shared memory
-----------------------------------------------------

If data are opened using the ``w+`` or ``r+`` mode in the main program, the
worker will get ``r+`` mode access. Thus the worker will be able to write
its results directly to the original data, alleviating the need of the
serialization to send back the results to the parent process.

Here is an example script on parallel processing with preallocated
``numpy.memmap`` datastructures
:ref:`sphx_glr_auto_examples_parallel_memmap.py`.

.. warning::

  Having concurrent workers write on overlapping shared memory data segments,
  for instance by using inplace operators and assignments on a `numpy.memmap`
  instance, can lead to data corruption as numpy does not offer atomic
  operations. The previous example does not risk that issue as each task is
  updating an exclusive segment of the shared result array.

  Some C/C++ compilers offer lock-free atomic primitives such as add-and-fetch
  or compare-and-swap that could be exposed to Python via CFFI_ for instance.
  However providing numpy-aware atomic constructs is outside of the scope
  of the joblib project.


.. _CFFI: https://cffi.readthedocs.org


Sharing GPU arrays between processes
------------------------------------

.. versionadded:: 1.6

In addition to numpy memmapping, :class:`joblib.Parallel` can share GPU arrays
(PyTorch tensors and CuPy arrays) with worker processes without copying their
content through host memory, using CUDA inter-process communication (IPC). This
is the GPU counterpart of the numpy memmapping described above and is controlled
by the ``share_gpu_arrays`` parameter:

.. code-block:: python

    import torch
    from joblib import Parallel, delayed

    def run(weights):
        # ``weights`` is a CUDA tensor backed by the same device memory as in
        # the parent process: no host round-trip and no extra device copy.
        return float(weights.sum())

    weights = torch.ones(10_000, 10_000, device="cuda:0")
    Parallel(n_jobs=4, backend="loky", share_gpu_arrays="on")(
        delayed(run)(weights) for _ in range(4)
    )

``share_gpu_arrays`` accepts three values:

- ``"auto"`` (the default): share device arrays larger than ``max_nbytes`` when
  a spawn-based process backend is used, and silently fall back to regular
  (host round-trip) pickling otherwise. This keeps existing code working
  unchanged.
- ``"on"``: share every device array regardless of ``max_nbytes`` and raise an
  error if sharing is requested but not feasible (for instance with the
  ``fork`` start method, which is incompatible with CUDA).
- ``"off"``: never share; all arrays use their framework's regular pickling.

This feature is experimental and comes with several constraints:

- It only applies to the ``"loky"`` backend and to ``"multiprocessing"`` when
  using a ``spawn`` (or ``forkserver``) start method. CUDA state cannot survive
  a ``fork``.
- Sharing is forward-only (from the parent to the workers). Arrays *returned*
  by the workers go through the regular (host round-trip) pickling of their
  framework.
- The shared array stays on a single physical GPU: all workers access the same
  device memory. This is data sharing, not multi-GPU replication.

.. warning::

  Shared GPU arrays must be treated as read-only inputs. Unlike numpy's
  ``mmap_mode='r'``, this read-only contract is **not** enforced by PyTorch or
  CuPy: writing to a shared array (or mutating it in the parent after dispatch)
  results in undefined behavior and possible data races between workers.

For PyTorch, sharing reuses ``torch.multiprocessing`` reductions, which handle
the producer-side keep-alive and the stream synchronization. For CuPy, joblib
keeps the source allocation alive for the duration of the ``Parallel`` call and
synchronizes the producer stream before sharing.


A final note: don't forget to clean up any temporary folder when you are done
with the computation::

  >>> import shutil
  >>> try:
  ...     shutil.rmtree(temp_folder)
  ... except OSError:
  ...     pass  # this can sometimes fail under Windows
