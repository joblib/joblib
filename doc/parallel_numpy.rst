Working with numerical data in shared memory (memmaping)
========================================================

By default the workers of the pool are real Python processes forked using the
``multiprocessing`` module of the Python standard library when ``n_jobs != 1``.
The arguments passed as input to the ``Parallel`` call are serialized and
reallocated in the memory of each worker process.

This can be problematic for large arguments as they will be reallocated
``n_jobs`` times by the workers.

This is both wasteful for the global memory usage but also for the
runtime because the allocation time can slow down the processing
significatively.

As this problem can often occur in scientific computing with ``numpy``
based datastructures, :class:`joblib.Parallel` provides a special
handling for large arrays to automatically dump them on the filesystem
and pass a reference to the worker to open them as memory map
on that file using the ``numpy.memmap`` subclass of ``numpy.ndarray``.
This makes it possible to share a segment of data between all the
worker processes.

The automated array to memmap conversion is triggered by a configurable
threshold on the size of the array::

  >>> import numpy as np
  >>> from joblib import Parallel, delayed
  >>> from joblib.pool import has_shareable_memory

  >>> Parallel(n_jobs=2, max_nbytes=1e6)(
  ...     delayed(has_shareable_memory)(np.ones(i)) for i in [1e2, 1e4, 1e6])
  [False, False, True]

For even finer tuning of the memory usage it is also possible to
dump the array as an memmap directly from the parent process to
free the memory before forking the worker processes. For instance
let's allocate a large array in the memory of the parent process::

  >>> large_array = np.ones(int(1e6))

Dump it to a local file for memmaping::

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

We can free the original array from the main process memory::

  >>> del large_array
  >>> import gc
  >>> _ = gc.collect()

It it possible to slice ``large_memmap`` into a smaller memmap::

  >>> small_memmap = large_memmap[2:5]
  >>> small_memmap.__class__.__name__, small_memmap.nbytes, small_memmap.shape
  ('memmap', 24, (3,))

Finally we can also take a ``np.ndarray`` view backed on that same
memory mapped file::

  >>> small_array = np.asarray(small_memmap)
  >>> small_array.__class__.__name__, small_array.nbytes, small_array.shape
  ('ndarray', 24, (3,))

All those three datastructures point to the same memory buffer and
this same buffer will also be reused directly by the worker processes
of a ``Parallel`` call::

  >>> Parallel(n_jobs=2, max_nbytes=None)(
  ...     delayed(has_shareable_memory)(a)
  ...     for a in [large_memmap, small_memmap, small_array])
  [True, True, True]

Note that here we used ``max_nbytes=None`` to disable the auto-dumping
feature of ``Parallel``. The fact that ``small_array`` is still in
shared memory in the worker processes is a consequence of the fact
that it was already backed by shared memory in the parent process.
The pickling machinery of ``Parallel`` multiprocessing queues are
able to detect this situation and optimize it on the fly to limit
the number of memory copies.

Also note that when you open your data using the ``w+`` or ``r+``
mode in the main program, the worker will have ``r+`` mode access
hence will be able to write results directly to it alleviating the
need to serialization to communicate back the results to the parent
process.

For instance see the `example script
<https://github.com/joblib/joblib/blob/master/examples/parallel_memmap.py>`_
on parallel processing with preallocated numpy.memmap datastructures.

It also makes it possible to do interprocess communication without the cost of
serializing datastructures. However the current implementation does not provide
locking tools for protecting concurrent read/write access to shared memory
chunks.

By the way, this is the end of this section, let's cleanup the temp
folder::

  >>> import shutil
  >>> shutil.rmtree(temp_folder)
