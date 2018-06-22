
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


Thread-based parallelism vs process-based parallelism
=====================================================

By default :class:`joblib.Parallel` uses the ``'loky'`` backend module to start
separate Python worker processes to execute tasks concurrently on
separate CPUs. This is a reasonable default for generic Python programs
but can induce a significant overhead as the input and output data need
to be serialized in a queue for communication with the worker processes.

When you know that the function you are calling is based on a compiled
extension that releases the Python Global Interpreter Lock (GIL) during
most of its computation then it is more efficient to use threads instead
of Python processes as concurrent workers. For instance this is the case
if you write the CPU intensive part of your code inside a `with nogil`_
block of a Cython function.

.. _`with nogil`: http://docs.cython.org/src/userguide/external_C_code.html#acquiring-and-releasing-the-gil

To hint that your code can efficiently use threads, just pass
``prefer="threads"`` as parameter of the :class:`joblib.Parallel` constructor.
In this case joblib will automatically use the ``"threading"`` backend
instead of the default ``"loky"`` backend:

    >>> Parallel(n_jobs=2, prefer="threads")(
    ...     delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

It is also possible to manually select a specific backend implementation
with the help of a context manager:

    >>> from joblib import parallel_backend
    >>> with parallel_backend('threading', n_jobs=2):
    ...    Parallel()(delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

The latter is especially useful when calling a library that uses
:class:`joblib.Parallel` internally without exposing backend selection as
part of its public API.

Note that the ``prefer="threads"`` option was introduced in joblib 0.12.
In prior versions, the same effect could be achieved by hardcoding a
specific backend implementation such as ``backend="threading"`` in the
call to :class:`joblib.Parallel` but this is now considered a bad pattern
(when done in a library) as it does not make it possible to override that
choice with the ``parallel_backend`` context manager.


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
:class:`joblib.Parallel` class to re-use the same pool of workers for several
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

.. include:: parallel_numpy.rst

Note that the ``'loky'`` backend now used by default for process-based
parallelism automatically tries to maintain and reuse a pool of workers
by it-self even for calls without the context manager.


Avoiding over-subscription of CPU ressources
============================================

The computation parallelism relies on the usage of multiple CPUs to perform the
operation simultaneously. When using more processes than the number of CPU on
a machine, the performance of each process is degraded as there is less
computational power available for each process. Moreover, when many processes
are running, the time taken by the OS scheduler to switch between them can
further hinder the performance of the computation. It is generally better to
avoid using significantly more processes or threads than the number of CPUs on
a machine.

Some third-partiy libraries -- *e.g.* the BLAS runtime used by ``numpy`` --
manage internally a thread-pool to perform their computations. The default
behavior is generally to use number of thread equals to the number of CPU
available. When these libraries are used with :class:`joblib.Parallel`, each
worker will spawn its thread-pools, resulting in a massive over-subscription of
the ressources that can slow down the computation compared to sequential one.
To cope with this problem, joblib forces by default supported third-party
libraries to use only one thread in workers with the ``'loky'`` backend. This
behavior can be overwritten by setting the proper environment variable to the
desired number of threads. This limitation is supported for the following
libraries:

    - OpenMP with the environment variable ``'OMP_NUM_THREADS'``,
    - OpenBLAS with the ``'OPENBLAS_NUM_THREADS'``,
    - MKL with the environment variable ``'MKL_NUM_THREADS'``,
    - Accelerated with the environment variable ``'VECLIB_MAXIMUM_THREADS'``,
    - Numexpr with the environment variable ``'NUMEXPR_NUM_THREADS'``.


Custom backend API (experimental)
=================================

.. versionadded:: 0.10

.. warning:: The custom backend API is experimental and subject to change
    without going through a deprecation cycle.

User can provide their own implementation of a parallel processing
backend in addition to the ``'loky'``, ``'threading'``,
``'multiprocessing'`` backends provided by default. A backend is
registered with the :func:`joblib.register_parallel_backend` function by
passing a name and a backend factory.

The backend factory can be any callable that returns an instance of
``ParallelBackendBase``. Please refer to the `default backends source code`_ as
a reference if you want to implement your own custom backend.

.. _`default backends source code`: https://github.com/joblib/joblib/blob/master/joblib/_parallel_backends.py

Note that it is possible to register a backend class that has some mandatory
constructor parameters such as the network address and connection credentials
for a remote cluster computing service::

    class MyCustomBackend(ParallelBackendBase):

        def __init__(self, endpoint, api_key):
           self.endpoint = endpoint
           self.api_key = api_key

        ...
        # Do something with self.endpoint and self.api_key somewhere in
        # one of the method of the class

    register_parallel_backend('custom', MyCustomBackend)

The connection parameters can then be passed to the
:func:`joblib.parallel_backend` context manager::

    with parallel_backend('custom', endpoint='http://compute', api_key='42'):
        Parallel()(delayed(some_function)(i) for i in range(10))

Using the context manager can be helpful when using a third-party library that
uses :class:`joblib.Parallel` internally while not exposing the ``backend``
argument in its own API.


A problem exists that external packages that register new parallel backends
must now be imported explicitly for their backends to be identified by joblib::

   >>> import joblib
   >>> with joblib.parallel_backend('custom'):  # doctest: +SKIP
   ...     ...  # this fails
   KeyError: 'custom'

   # Import library to register external backend
   >>> import my_custom_backend_library  # doctest: +SKIP
   >>> with joblib.parallel_backend('custom'):  # doctest: +SKIP
   ...     ... # this works

This can be confusing for users.  To resolve this, external packages can
safely register their backends directly within the joblib codebase by creating
a small function that registers their backend, and including this function
within the ``joblib.parallel.EXTERNAL_PACKAGES`` dictionary::

   def _register_custom():
       try:
           import my_custom_library
       except ImportError:
           raise ImportError("an informative error message")

   EXTERNAL_BACKENDS['custom'] = _register_custom

This is subject to community review, but can reduce the confusion for users
when relying on side effects of external package imports.


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
    :noindex:

.. autofunction:: joblib.delayed

.. autofunction:: joblib.register_parallel_backend

.. autofunction:: joblib.parallel_backend
