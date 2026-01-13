.. _custom_parallel_backend:

==================================
Parallel backend customization API
==================================

User can provide their own implementation of a parallel processing backend in
addition to the ``'loky'``, ``'threading'``, ``'multiprocessing'`` backends
provided by default.

A new backend is defined by providing a backend factory, which is then
registered with the :func:`joblib.register_parallel_backend` function by
passing a name and the factory.

For instance, to register a custom backend named ``'custom'`` with a factory ``MyCustomBackend``, you can call:

.. code-block:: python

    # Register the backend so it can be used with parallel_config
    register_parallel_backend('custom', MyCustomBackend)

This backend can then be used within the :func:`~joblib.parallel_config`
context manager, as:

.. code-block:: python

    from joblib import Parallel, delayed, parallel_config

    with parallel_config("custom"):
        res = Parallel(2)(delayed(id)(i) for i in range(10))


Minimal backend factory specification
======================================

The backend factory can be any callable that returns an instance of
``ParallelBackendBase``. The complete specification of the backend factory is
referenced in the `default backends source code`_.
Here, we provide minimal details to implement a custom backend factory and
the next section describes various possible customizations.

The key methods for the backend factory are:

- ``configure``: this method is called when creating a particular instance of
  the ``Parallel`` object. It can allocate the resources needed for the
  parallel computation and return the effective number of jobs that can be run
  in parallel.
- ``effective_n_jobs``: This method is called to determine the number of jobs
  that can be run in parallel. It can be used to limit the number of jobs depending
  on the specified ``n_jobs`` and the system it runs on.
- ``terminate``: this method is called when the parallel computation is
  finished. It can be used to clean up the resources allocated by the backend.
- ``submit``: this method is called to launch the computation for a given task
  in the ``Parallel`` call. The task is a single callable that takes no
  argument and the method should return a future-like object that allows
  tracking the task's progress. The ``callback`` argument passed to this method
  is a callable that should be called when the task is completed, typically through
  the ``add_done_callback`` method of the future-like object.
- ``retrieve_result_callback``: This method is called within the callback
  function passed to ``submit``. It is called with the same arguments provided
  by the backend to the callback functionality -- typically the future-like object.
  It should retrieve and return the result of the function executed in parallel.

Below is a minimal example of a custom backend factory that uses a
``ThreadPoolExecutor`` to run the tasks in parallel:

.. code-block:: python

    from concurrent.futures import ThreadPoolExecutor

    from joblib import ParallelBackendBase
    from joblib import register_parallel_backend


    class MyCustomBackend(ParallelBackendBase):

        supports_retrieve_callback = True

        def configure(self, n_jobs=1, parallel=None, **backend_kwargs):
            """Configure the backend for a specific instance of Parallel."""
            self.n_jobs = n_jobs

            n_jobs = self.effective_n_jobs(n_jobs)
            self._executor = ThreadPoolExecutor(n_jobs)

            # Return the effective number of jobs
            return n_jobs

        def terminate(self):
            """Clean-up the resources associated with the backend."""
            self._executor.shutdown()
            self._executor = None

        def effective_n_jobs(self, n_jobs):
            """Determine the number of jobs that can be run in parallel."""
            return n_jobs

        def submit(self, func, callback):
            """Schedule a function to be run and return a future-like object.

            This method should return a future-like object that allow tracking
            the progress of the task.

            If ``supports_retrieve_callback`` is False, the return value of this
            method is passed to ``retrieve_result`` instead of calling
            ``retrieve_result_callback``.

            Parameters
            ----------
            func: callable
                The function to be run in parallel.

            callback: callable
                A callable that will be called when the task is completed. This callable
                is a wrapper around ``retrieve_result_callback``. This should be added
                to the future-like object returned by this method, so that the callback
                is called when the task is completed.

                For future-like backends, this can be achieved with something like
                ``future.add_done_callback(callback)``.

            Returns
            -------
            future: future-like
                A future-like object to track the execution of the submitted function.
            """
            future = self._executor.submit(func)
            future.add_done_callback(callback)
            return future

        def retrieve_result_callback(self, future):
            """Called within the callback function passed to `submit`.

            This method can customise how the result of the function is retrieved
            from the future-like object.

            Parameters
            ----------
            future: future-like
                The future-like object returned by the `submit` method.

            Returns
            -------
            result: object
                The result of the function executed in parallel.
            """
            return future.result()


Extra customizations
--------------------

The backend API offers several hooks that can be used to customize its behavior.

Passing extra arguments to the backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is possible to register a backend class that has some mandatory
constructor parameters such as the network address and connection credentials
for a remote cluster computing service:

.. code-block:: python

    class MyCustomBackend(ParallelBackendBase):

        def __init__(self, endpoint, api_key, nesting_level=0):
           super().__init__(nesting_level=nesting_level)
           self.endpoint = endpoint
           self.api_key = api_key

        ...
        # Do something with self.endpoint and self.api_key somewhere in
        # one of the method of the class

    register_parallel_backend('custom', MyCustomBackend)

The connection parameters can then be passed to the
:func:`~joblib.parallel_config` context manager:

.. code-block:: python

    with parallel_config(backend='custom', endpoint='http://compute',
                         api_key='42'):
        Parallel()(delayed(some_function)(i) for i in range(10))

Using the context manager can be helpful when using a third-party library that
uses :class:`joblib.Parallel` internally while not exposing the ``backend``
argument in its own API.

Cancelling tasks
~~~~~~~~~~~~~~~~

If the backend allow to cancel tasks, the method ``abort_everything`` can be
implemented to abort all the tasks that are currently running as soon as one of
the tasks raises an exception. This can be useful to avoid wasting
computational resources when the call will fail.

This method have an extra parameters ``ensure_ready`` that informs the backend
whether the error was part of a single call to ``Parallel`` or in a context
manager block. In the case of a single call (``ensure_ready=False``), there is
no need to re-spawn workers for future calls, while in the case of a context
(``ensure_ready=True``), one could call ``configure`` to re-allocate computational
resources.

.. code-block:: python

    def abort_everything(self, ensure_ready=True):
        """Abort any running tasks

        This is called when an exception has been raised when executing a task
        and all the remaining tasks will be ignored and can therefore be
        aborted to spare computation resources.

        If ensure_ready is True, the backend should be left in an operating
        state as future tasks might be re-submitted via that same backend
        instance.

        If ensure_ready is False, the implementer of this method can decide
        to leave the backend in a closed / terminated state as no new task
        are expected to be submitted to this backend.

        Setting ensure_ready to False is an optimization that can be leveraged
        when aborting tasks via killing processes from a local process pool
        managed by the backend it-self: if we expect no new tasks, there is no
        point in re-creating new workers.
        """
        pass

Setting up Nested Parallelism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backend can also provide a method ``get_nested_backend`` that will be used
to setup the default backend to be used in nested parallel calls.
By default, the default backend is set to a thread-based backend for the first
level and then falls back to a sequential backend to avoid spawning too many
threads on the host.

.. code-block:: python

    def get_nested_backend(self):
        """Backend instance to be used by nested Parallel calls.

        By default a thread-based backend is used for the first level of
        nesting. Beyond, switch to sequential backend to avoid spawning too
        many threads on the host.
        """
        nesting_level = getattr(self, "nesting_level", 0) + 1
        return LokyBackend(nesting_level=nesting_level), None

Another nested parallelism that needs to be controlled is the numbers of thread
in third-party C-level threadpools, *e.g.* OpenMP, MKL, or BLAS. In ``joblib``,
this is controlled with the ``inner_max_num_threads`` argument that can be
provided to the backend in the :func:`~joblib.parallel_config` context manager.
To support this argument, the backend should set the
``supports_inner_max_num_threads`` class attribute to ``True`` and accept the
argument in the constructor to set this up in the workers. A helper to set this
in the workers is to use environment variables provided by
``self._prepare_worker_env(n_jobs)``.

Third-party backend registration
================================

To be used seamlessly within :func:`~joblib.parallel_config`, external packages need
to register their parallel backends so they are identified by joblib. This requires
importing extra packages as in::

   >>> import joblib
   >>> with joblib.parallel_config(backend='custom'):  # doctest: +SKIP
   ...     ...  # this fails
   KeyError: 'custom'

   # Import library to register external backend
   >>> import my_custom_backend_library  # doctest: +SKIP
   >>> with joblib.parallel_config(backend='custom'):  # doctest: +SKIP
   ...     ... # this works

This can be confusing for users.  To resolve this, external packages can
safely register their backends directly within the joblib codebase by creating
a small function that registers their backend, and including this function
within the ``joblib.parallel.EXTERNAL_BACKENDS`` dictionary::

   def _register_custom():
       try:
           import my_custom_library
       except ImportError:
           raise ImportError("an informative error message")

   EXTERNAL_BACKENDS['custom'] = _register_custom

This is subject to community review, but can reduce the confusion for users
when relying on side effects of external package imports.


.. Links to various page

.. _`default backends source code`: https://github.com/joblib/joblib/blob/main/joblib/_parallel_backends.py
