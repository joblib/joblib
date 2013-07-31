
=================================
Embarrassingly parallel for loops
=================================

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


.. include:: parallel_numpy.rst


`Parallel` reference documentation
===================================

.. autoclass:: joblib.Parallel
   :members: auto

