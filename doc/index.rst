.. container::

   .. image:: _static/joblib_logo.svg
      :class: only-light
      :width: 30%
      :align: center

   .. image:: _static/joblib_logo_dark.svg
      :class: only-dark
      :width: 30%
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
  :gutter: 3

  .. grid-item-card:: Disk-based caching
    :link: user_guide/memory.html

    Using the :class:`~joblib.Memory` for disk-based caching

  .. grid-item-card:: Embarrassingly parallel
    :link: user_guide/parallel.html

    Using :class:`~joblib.Parallel` for parallel loops using multiprocessing

  .. grid-item-card:: Parallel backend
    :link: user_guide/custom_parallel_backend.html

    Using its own parallel processing backend

  .. grid-item-card:: API
    :link: references.html

  .. grid-item-card:: Examples
    :link: auto_examples/index.html

  .. grid-item-card:: Release Notes
    :link: CHANGES.html

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
