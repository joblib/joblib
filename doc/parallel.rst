
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

.. include:: parallel_numpy.rst


`Parallel` reference documentation
===================================

.. autoclass:: joblib.Parallel
   :members: auto

