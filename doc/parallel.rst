
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

By default joblib uses the ``'loky'`` backend to safely and efficiently
use a pool of worker Python processes to execute the delayed functions
on each set of arguments in the comprehension.


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
   subprocesses when using ``joblib.Parallel``. In other words, you
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


Using the threading backend
===========================

By default :class:`Parallel` uses the ``'loky'`` backend module to start
separate Python worker processes to execute tasks concurrently on
separate CPUs. This is a reasonable default for generic Python programs
but it induces some overhead as the input and output data need to be
serialized in a queue for communication with the worker processes.

If you know that the function you are calling is based on a compiled extension
that releases the Python Global Interpreter Lock (GIL) during most of its
computation then it might be more efficient to use threads instead of Python
processes as concurrent workers. For instance this is the case if you write the
CPU intensive part of your code inside a `with nogil`_ block of a Cython
function.

.. _`with nogil`: http://docs.cython.org/src/userguide/external_C_code.html#acquiring-and-releasing-the-gil

To use the threads, just pass ``"threading"`` as the value of the ``backend``
parameter of the :class:`Parallel` constructor:

    >>> Parallel(n_jobs=2, backend="threading")(
    ...     delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

or alternatively using a context manager:

    >>> from joblib import parallel_backend
    >>> with parallel_backend('threading'):
    ...    Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


Reusing a pool of workers
=========================

Some algorithms require to make several consecutive calls to a parallel
function interleaved with processing of the intermediate results. Calling
``Parallel`` several times in a loop is sub-optimal because it will create and
destroy a pool of workers (threads or processes) several times which can cause
a significant overhead.

For this case it is more efficient to use the context manager API of the
``Parallel`` class to re-use the same pool of workers for several calls to
the ``Parallel`` object::

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


Bad interaction of multiprocessing and third-party libraries
============================================================

Using the ``'multiprocessing'`` backend can cause a crash when using
third party libraries that manage their own native thread-pool if the
library is first used in the main process and subsequently called again
in a worker process (inside the ``Parallel`` call).

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
also possible  to get ``joblib.Parallel`` configured to use the
``'forkserver'`` start method on Python 3.4 and later. The start method
has to be configured by setting the ``JOBLIB_START_METHOD`` environment
variable to ``'forkserver'`` instead of the default ``'fork'`` start
method. However the user should be aware that using the ``'forkserver'``
method prevents ``joblib.Parallel`` to call function interactively
defined in a shell session.

You can read more on this topic in the `multiprocessing documentation
<https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

Under Windows the ``fork`` system call does not exist at all so this problem
does not exist (but multiprocessing has more overhead).


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


`Parallel` reference documentation
==================================

.. autoclass:: joblib.Parallel
    :noindex:

.. autofunction:: joblib.delayed

.. autofunction:: joblib.register_parallel_backend

.. autofunction:: joblib.parallel_backend
