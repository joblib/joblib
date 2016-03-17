
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

Under the hood, the :class:`Parallel` object create a multiprocessing
`pool` that forks the Python interpreter in multiple processes to execute
each of the items of the list. The `delayed` function is a simple trick
to be able to create a tuple `(function, args, kwargs)` with a
function-call syntax.

.. warning::

   Under Windows, it is important to protect the main loop of code to
   avoid recursive spawning of subprocesses when using joblib.Parallel.
   In other words, you should be writing code like this:

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

   **No** code should *run* outside of the "if __name__ == '__main__'"
   blocks, only imports and definitions.


Using the threading backend
===========================

By default :class:`Parallel` uses the Python ``multiprocessing`` module to fork
separate Python worker processes to execute tasks concurrently on separate
CPUs. This is a reasonable default for generic Python programs but it induces
some overhead as the input and output data need to be serialized in a queue for
communication with the worker processes.

If you know that the function you are calling is based on a compiled extension
that releases the Python Global Interpreter Lock (GIL) during most of its
computation then it might be more efficient to use threads instead of Python
processes as concurrent workers. For instance this is the case if you write the
CPU intensive part of your code inside a `with nogil`_ block of a Cython
function.

To use the threads, just pass ``"threading"`` as the value of the ``backend``
parameter of the :class:`Parallel` constructor:

    >>> Parallel(n_jobs=2, backend="threading")(
    ...     delayed(sqrt)(i ** 2) for i in range(10))
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]


.. _`with nogil`:: http://docs.cython.org/src/userguide/external_C_code.html#acquiring-and-releasing-the-gil


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


Bad interaction of multiprocessing and third-party libraries
============================================================

Prior to Python 3.4 the ``'multiprocessing'`` backend of joblib can only use
the ``fork`` strategy to create worker processes under non-Windows systems.
This can cause some third-party libraries to crash or freeze. Such libraries
include as Apple vecLib / Accelerate (used by NumPy under OSX), some old
version of OpenBLAS (prior to 0.2.10) or the OpenMP runtime implementation from
GCC.

To avoid this problem ``joblib.Parallel`` can be configured to use the
``'forkserver'`` start method on Python 3.4 and later. The start method has to
be configured by setting the ``JOBLIB_START_METHOD`` environment variable to
``'forkserver'`` instead of the default ``'fork'`` start method. However the
user should be aware that using the ``'forkserver'`` prevents
``joblib.Parallel`` to call function interactively defined in a shell session.

You can read more on this topic in the `multiprocessing documentation
<https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

Under Windows the ``fork`` system call does not exist at all so this problem
does not exist (but multiprocessing has more overhead).


Custom backend API (experimental)
=================================

.. versionadded:: 0.10

.. warning:: The custom backend API is experimental and subject to change
    without going through a deprecation cycle.

User can provide their own implementation of a parallel processing backend in
addition to the ``'multiprocessing'`` and ``'threading'`` backends provided by
default. A backend is registered with the
:func:`joblib.register_parallel_backend` function by passing a name and a
backend factory.

The backend factory can be any callable that takes no argument and return an
instance of ``ParallelBackendBase``. Moreover, the default backend can be
overwritten globally by setting ``make_default=True``. Please refer to the
`default backends source code`_ as a reference if you want to implement your
own custom backend.

.. _`default backends source code`: https://github.com/joblib/joblib/blob/master/joblib/_parallel_backends.py

Users can pass a lambda closure as a factory when the backend needs to be
parameterized, for instance to give the network address to a remote cluster
computing service along with some connection credentials::

    class MyCustomBackend(ParallelBackendBase):
        def __init__(self, hostname, api_key):
           self.hostname = hostname
           self.api_key = api_key

        ...
        # Do something with self.hostname and self.api_key somewhere in
        # one of the method of the class

    register_parallel_backend(
       'custom', lambda: MyCustomBackend('example.com', 'deadbeefcafebabe'))

The new backend can then be enabled by passing its name as the ``backend``
argument to the constructor of the :class:`joblib.Parallel` class::

    Parallel(n_jobs=2, backend='custom')(
        delayed(some_function)(i) for i in range(10))

or equivalently by using the :func:`joblib.parallel_backend` context manager::

    with parallel_backend('custom', n_jobs=2):
        Parallel()(delayed(some_function)(i) for i in range(10))

Using the context manager can be helpful when using a third-party library that
uses :class:`joblib.Parallel` internally while not exposing the ``backend``
argument in its own API.


`Parallel` reference documentation
==================================

.. autoclass:: joblib.Parallel
   :members: auto

.. autofunction:: joblib.delayed

.. autofunction:: joblib.register_parallel_backend

.. autofunction:: joblib.parallel_backend
