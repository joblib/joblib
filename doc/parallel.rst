
.. _parallel:

=================================
Embarrassingly parallel for loops
=================================

Common usage
============

Joblib provides a simple helper class to write parallel for loops using
multiprocessing. The core idea is to write the code to be executed as a
generator expression, and convert it to parallel computing::

    >>> from math import sqrt
    >>> [sqrt(i ** 2) for i in range(10)]
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

can be spread over 2 CPUs using the following::

    >>> from math import sqrt
    >>> from joblib import Parallel, delayed
    >>> Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

The output can be a generator that yields the results as soon as they're
available, even if the subsequent tasks aren't completed yet. The order
of the outputs always matches the order the inputs have been submitted with::

    >>> from math import sqrt
    >>> from joblib import Parallel, delayed
    >>> parallel = Parallel(n_jobs=2, return_as="generator")
    >>> output_generator = parallel(delayed(sqrt)(i ** 2) for i in range(10))
    >>> print(type(output_generator))
    <class 'generator'>
    >>> print(next(output_generator))
    0.0
    >>> print(next(output_generator))
    1.0
    >>> print(list(output_generator))
    [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

This generator enables reducing the memory footprint of
:class:`joblib.Parallel` calls in case the results can benefit from on-the-fly
aggregation, as illustrated in
:ref:`sphx_glr_auto_examples_parallel_generator.py`.

It is also possible to return a generator that yields the results in the order
of completion rather than the order of submission, by using
``return_as="generator_unordered"`` instead of ``return_as="generator"``.
In this case the order of the outputs depends on the concurrency of workers
and is not deterministic, meaning the results can be yielded with a different
order every time the code is executed.

Thread-based parallelism vs process-based parallelism
=====================================================

By default :class:`joblib.Parallel` uses the ``'loky'`` backend module to start
separate Python worker processes to execute tasks concurrently on separate CPUs.
But ``joblib`` also supports other backends to execute tasks concurrently, with
different trade-offs (see :ref:`parallel_config_backend`).

While ``'loky'`` is a reasonable default for generic Python programs , it
can induce a significant overhead as the input and output data need to be
serialized in a queue for communication with the worker processes (see
:ref:`serialization_and_processes`).

When you know that the function you are calling is based on a compiled
extension that releases the Python Global Interpreter Lock (GIL) during
most of its computation then it is more efficient to use threads instead
of Python processes as concurrent workers. For instance this is the case
if you write the CPU intensive part of your code inside a `with nogil`_
block of a Cython function.

.. _`with nogil`: https://docs.cython.org/src/userguide/external_C_code.html#acquiring-and-releasing-the-gil

To hint that your code can efficiently use threads, just pass
``prefer="threads"`` as parameter of the :class:`joblib.Parallel` constructor.
In this case joblib will automatically use the ``"threading"`` backend
instead of the default ``"loky"`` backend:

    >>> Parallel(n_jobs=2, prefer="threads")(
    ...     delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

Note that the ``prefer="threads"`` option was introduced in joblib 0.12.
In prior versions, the same effect could be achieved by hardcoding a
specific backend implementation such as ``backend="threading"`` in the
call to :class:`joblib.Parallel` but this is now considered a bad pattern
(when done in a library) as it does not make it possible to override that
choice with the :func:`~joblib.parallel_config` context manager (see the next session).

.. topic:: The loky backend may not always be available

   Some rare systems do not support multiprocessing (for instance
   Pyodide). In this case the loky backend is not available and the
   default backend falls back to threading.

.. _parallel_config_backend:

Setting up ``joblib``'s backend with ``parallel_config``
========================================================

The :func:`~joblib.parallel_config` context manager allows selecting
a specific backend implementation or setting the default parameters like the
number of jobs in the backend as:

    >>> from joblib import parallel_config
    >>> with parallel_config(backend='threading', n_jobs=2):
    ...    Parallel()(delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

This context manager is especially useful when calling a library that uses
:class:`joblib.Parallel` internally without exposing backend selection as
part of its public API.

The context manager allow to set various backend parameters:

- ``n_jobs``: Set the number of jobs to run in parallel.
- ``verbose``: Set the verbosity level of the backend.
- ``prefer``: Set the preferred backend to use. Can be ``'processes'`` or
  ``'threads'``.
- ``require``: Set the requirement for the backend. Can be ``'sharedmem'``.
- ``inner_max_num_threads``: Limit the maximum number of threads used by
  third-party libraries that manage their own C-level thread-pool. This argument
  is only supported by backend that set the ``supports_inner_max_num_threads``
  class attribute to ``True``, such as the ``'loky'`` backend.
- ``temp_folder, max_nbytes, mmap_mode``: Control the auto-memmapping behavior
  of the backend. See :ref:`auto_memmapping_doc` for more details.

Extra arguments passed in the :func:`~joblib.parallel_config` context are
passed to the backend constructor, allowing additional parameters to be set up:

- ``LokyBackend``
    - ``initializer``, ``initargs``: setup function and its arguments to call
      in each worker process.
    - ``idle_worker_timeout``: timeout in seconds for a worker to wait
      for a new task before being clean up. The default is `300 s`.
- ``MultprocessingBackend``
    - ``initializer``, ``initargs``: setup function and its arguments to call
      in each worker process.
    - ``maxtasksperchild``: maximum number of tasks a worker can
      execute before being replace with a fresh one.
    - ``context``: specify the start method to use for creating new worker
      processes.

In addition to the builtin joblib backends, there are several cluster-specific
backends you can use:

* `Dask <https://docs.dask.org/en/stable/>`_ backend for Dask clusters
  (see :ref:`sphx_glr_auto_examples_parallel_distributed_backend_simple.py` for an example),
* `Ray <https://docs.ray.io/en/latest/index.html>`_ backend for Ray clusters,
* `Joblib Apache Spark Backend <https://github.com/joblib/joblib-spark>`_
  to distribute joblib tasks on a Spark cluster.

It is also possible to provide a custom backend implementation, as described in
:ref:`custom_parallel_backend`.

.. _serialization_and_processes:

Serialization & Processes
=========================

To share function definition across multiple python processes, it is necessary to rely on a serialization protocol. The standard protocol in python is :mod:`pickle` but its default implementation in the standard library has several limitations. For instance, it cannot serialize functions which are defined interactively or in the :code:`__main__` module.

To avoid this limitation, the ``loky`` backend now relies on |cloudpickle| to serialize python objects. |cloudpickle| is an alternative implementation of the pickle protocol which allows the serialization of a greater number of objects, in particular interactively defined functions. So for most usages, the loky ``backend`` should work seamlessly.


The main drawback of |cloudpickle| is that it can be slower than the :mod:`pickle` module in the standard library. In particular, it is critical for large python dictionaries or lists, where the serialization time can be up to 100 times slower. There is two ways to alter the serialization process for the ``joblib`` to temper this issue:

- If you are on an UNIX system, you can switch back to the old ``multiprocessing`` backend. With this backend, interactively defined functions can be shared with the worker processes using the fast :mod:`pickle`. The main issue with this solution is that using ``fork`` to start the process breaks the standard POSIX and can have weird interaction with third party libraries such as ``numpy`` and ``openblas``.

- If you wish to use the ``loky`` backend with a different serialization library, you can set the ``LOKY_PICKLER=mod_pickle`` environment variable to use the ``mod_pickle`` as the serialization library for ``loky``. The module ``mod_pickle`` passed as an argument should be importable as ``import mod_pickle`` and should contain a ``Pickler`` object, which will be used to serialize to objects. It can be set to ``LOKY_PICKLER=pickle`` to use the pickling module from stdlib. The main drawback with ``LOKY_PICKLER=pickle`` is that interactively defined functions will not be serializable anymore. To cope with this, you can use this solution together with the :func:`joblib.wrap_non_picklable_objects` wrapper, which can be used as a decorator to locally enable using |cloudpickle| for specific objects. This way, you can have fast pickling of all python objects and locally enable slow pickling for interactive functions. An example is given in loky_wrapper_.

.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle"><code>cloudpickle</code></a>

.. _loky_wrapper:  auto_examples/serialization_and_wrappers.html


Shared-memory semantics
=======================

The default backend of joblib will run each function call in isolated
Python processes, therefore they cannot mutate a common Python object
defined in the main program.

However if the parallel function really needs to rely on the shared
memory semantics of threads, it should be made explicit with
``require='sharedmem'``, for instance:

    >>> shared_set = set()
    >>> def collect(x):
    ...    shared_set.add(x)
    ...
    >>> Parallel(n_jobs=2, require='sharedmem')(
    ...     delayed(collect)(i) for i in range(5))
    [None, None, None, None, None]
    >>> sorted(shared_set)
    [0, 1, 2, 3, 4]

Keep in mind that relying a on the shared-memory semantics is probably
suboptimal from a performance point of view as concurrent access to a
shared Python object will suffer from lock contention.

Reusing a pool of workers
=========================

Some algorithms require to make several consecutive calls to a parallel
function interleaved with processing of the intermediate results. Calling
:class:`joblib.Parallel` several times in a loop is sub-optimal because it will
create and destroy a pool of workers (threads or processes) several times which
can cause a significant overhead.

For this case it is more efficient to use the context manager API of the
:class:`joblib.Parallel` class to reuse the same pool of workers for several
calls to the :class:`joblib.Parallel` object::

    >>> with Parallel(n_jobs=2) as parallel:
    ...    accumulator = 0.
    ...    n_iter = 0
    ...    while accumulator < 1000:
    ...        results = parallel(delayed(sqrt)(accumulator + i ** 2)
    ...                           for i in range(5))
    ...        accumulator += sum(results)  # synchronization barrier
    ...        n_iter += 1
    ...
    >>> (accumulator, n_iter)                            # doctest: +ELLIPSIS
    (1136.596..., 14)

Note that the ``'loky'`` backend now used by default for process-based
parallelism automatically tries to maintain and reuse a pool of workers
by it-self even for calls without the context manager.


.. _auto_memmapping_doc:

Working with numerical data in shared memory (memmapping)
=========================================================

.. include:: parallel_numpy.rst

Avoiding over-subscription of CPU resources
============================================

The computation parallelism relies on the usage of multiple CPUs to perform the
operation simultaneously. When using more processes than the number of CPU on
a machine, the performance of each process is degraded as there is less
computational power available for each process. Moreover, when many processes
are running, the time taken by the OS scheduler to switch between them can
further hinder the performance of the computation. It is generally better to
avoid using significantly more processes or threads than the number of CPUs on
a machine.

Some third-party libraries -- *e.g.* the BLAS runtime used by ``numpy`` --
internally manage a thread-pool to perform their computations. The default
behavior is generally to use a number of threads equals to the number of CPUs
available. When these libraries are used with :class:`joblib.Parallel`, each
worker will spawn its own thread-pools, resulting in a massive over-subscription
of resources that can slow down the computation compared to a sequential
one. To cope with this problem, joblib tells supported third-party libraries
to use a limited number of threads in workers managed by the ``'loky'``
backend: by default each worker process will have environment variables set to
allow a maximum of ``cpu_count() // n_jobs`` so that the total number of
threads used by all the workers does not exceed the number of CPUs of the
host.

This behavior can be overridden by setting the proper environment variables to
the desired number of threads. This override is supported for the following
libraries:

    - OpenMP with the environment variable ``'OMP_NUM_THREADS'``,
    - OpenBLAS with the ``'OPENBLAS_NUM_THREADS'``,
    - MKL with the environment variable ``'MKL_NUM_THREADS'``,
    - Accelerated with the environment variable ``'VECLIB_MAXIMUM_THREADS'``,
    - Numexpr with the environment variable ``'NUMEXPR_NUM_THREADS'``.

Since joblib 0.14, it is also possible to programmatically override the default
number of threads using the ``inner_max_num_threads`` argument of the
:func:`~joblib.parallel_config` function as follows:

.. code-block:: python

    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=2):
        results = Parallel(n_jobs=4)(delayed(func)(x, y) for x, y in data)

In this example, 4 Python worker processes will be allowed to use 2 threads
each, meaning that this program will be able to use up to 8 CPUs concurrently.


Old multiprocessing backend
===========================

Prior to version 0.12, joblib used the ``'multiprocessing'`` backend as
default backend instead of ``'loky'``.

This backend creates an instance of `multiprocessing.Pool` that forks
the Python interpreter in multiple processes to execute each of the
items of the list. The `delayed` function is a simple trick to be able
to create a tuple `(function, args, kwargs)` with a function-call
syntax.

.. warning::

   Under Windows, the use of ``multiprocessing.Pool`` requires to
   protect the main loop of code to avoid recursive spawning of
   subprocesses when using :class:`joblib.Parallel`. In other words, you
   should be writing code like this when using the ``'multiprocessing'``
   backend:

   .. code-block:: python

      import ....

      def function1(...):
          ...

      def function2(...):
          ...

      ...
      if __name__ == '__main__':
          # do stuff with imports and functions defined about
          ...

   **No** code should *run* outside of the ``"if __name__ ==
   '__main__'"`` blocks, only imports and definitions.

   The ``'loky'`` backend used by default in joblib 0.12 and later does
   not impose this anymore.


Bad interaction of multiprocessing and third-party libraries
============================================================

Using the ``'multiprocessing'`` backend can cause a crash when using
third party libraries that manage their own native thread-pool if the
library is first used in the main process and subsequently called again
in a worker process (inside the :class:`joblib.Parallel` call).

Joblib version 0.12 and later are no longer subject to this problem
thanks to the use of `loky <https://github.com/tomMoral/loky>`_ as the
new default backend for process-based parallelism.

Prior to Python 3.4 the ``'multiprocessing'`` backend of joblib can only
use the ``fork`` strategy to create worker processes under non-Windows
systems. This can cause some third-party libraries to crash or freeze.
Such libraries include Apple vecLib / Accelerate (used by NumPy under
OSX), some old version of OpenBLAS (prior to 0.2.10) or the OpenMP
runtime implementation from GCC which is used internally by third-party
libraries such as XGBoost, spaCy, OpenCV...

The best way to avoid this problem is to use the ``'loky'`` backend
instead of the ``multiprocessing`` backend. Prior to joblib 0.12, it is
also possible  to get :class:`joblib.Parallel` configured to use the
``'forkserver'`` start method on Python 3.4 and later. The start method
has to be configured by setting the ``JOBLIB_START_METHOD`` environment
variable to ``'forkserver'`` instead of the default ``'fork'`` start
method. However the user should be aware that using the ``'forkserver'``
method prevents :class:`joblib.Parallel` to call function interactively
defined in a shell session.

You can read more on this topic in the `multiprocessing documentation
<https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

Under Windows the ``fork`` system call does not exist at all so this problem
does not exist (but multiprocessing has more overhead).


`Parallel` reference documentation
==================================

.. autoclass:: joblib.Parallel
   :members: dispatch_next, dispatch_one_batch, format, print_progress
   :no-inherited-members:
   :noindex:

.. autofunction:: joblib.delayed

.. autofunction:: joblib.parallel_config
   :noindex:

.. autofunction:: joblib.wrap_non_picklable_objects

.. autofunction:: joblib.register_parallel_backend

.. autoclass:: joblib.parallel.ParallelBackendBase

.. autoclass:: joblib.parallel.AutoBatchingMixin
