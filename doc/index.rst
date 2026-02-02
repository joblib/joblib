.. container::

   .. image:: _static/joblib_logo.svg
      :class: only-light
      :width: 40%
      :align: center

   .. image:: _static/joblib_logo_dark.svg
      :class: only-dark
      :width: 40%
      :align: center

   .. rubric:: Version |release|
      :class: center-rubric

Joblib documentation
====================

Joblib is a package for **parallel computing** and **disk-based caching** in Python.
It is optimized to be **fast** and **robust** on large data in particular
and has specific optimizations for `numpy` arrays.
Joblib leaves your code and your flow control as unmodified as possible.
It is **BSD-licensed**.

.. grid:: 3

  .. grid-item-card:: Disk-based caching
    :link: user_guide/memory.html

    Using the :class:`~joblib.Memory` for disk-based caching

  .. grid-item-card:: Embarrassingly parallel
    :link: user_guide/parallel.html

    Using :class:`~joblib.Parallel` for parallel loops using multiprocessing

  .. grid-item-card:: Persistence
    :link: user_guide/persistence.html

    A pickle replacement for large data using :func:`joblib.dump` and :func:`joblib.load`

Get Joblib
----------

.. code-block:: bash

  pip install joblib

.. toctree::
  :hidden:

  user_guide/index
  references
  ../auto_examples/index
  developing
