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
  >>> from joblib.pool import has_shared_memory

  >>> Parallel(n_jobs=2, max_nbytes=1e6)(
  ...     delayed(has_shared_memory)(np.ones(i)) for i in [1e2, 1e4, 1e6])
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

  >>> filename = os.path.join(tempfile.gettempdir(), 'joblib_test.mmap')
  >>> if os.path.exists(filename): os.unlink(filename)
  >>> _ = dump(large_array, filename)
  >>> large_array = load(filename, mmap_mode='r+')

Launch the parallel computation directly on the memapped data::

  >>> Parallel(n_jobs=2, max_nbytes=1e6)(
  ...     delayed(has_shared_memory)(large_array) for i in [1, 2, 3])
  [True, True, True]

  >>> os.unlink(filename)

Also note that when you open your data using the ``w+`` or ``r+``
mode in the main program, the worker will have ``r+`` mode access
hence will be able to write results directly to it aleviating the
need to serialization to communicate back the results to the parent
process.

For instance see the `example script
<https://github.com/joblib/joblib/blob/master/examples/parallel_memmap.py>`_
on parallel processing with preallocated numpy.memmap datastructures.

It also makes it possible to do interprocess communication without
the cost of serializing datastructures. However the current
implementation does not yet provide locking tools for protecting
concurrent read/write access to shared memory chunks.
