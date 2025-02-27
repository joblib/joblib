Writing a new backend
=====================

.. versionadded:: 1.5

User can provide their own implementation of a parallel processing
backend in addition to the ``'loky'``, ``'threading'``,
``'multiprocessing'`` backends provided by default. A backend is
registered with the :func:`joblib.register_parallel_backend` function by
passing a name and a backend factory.

The backend factory can be any callable that returns an instance of
``ParallelBackendBase``. Please refer to the `default backends source code`_ as
a reference if you want to implement your own custom backend.

.. _`default backends source code`: https://github.com/joblib/joblib/blob/main/joblib/_parallel_backends.py

Note that it is possible to register a backend class that has some mandatory
constructor parameters such as the network address and connection credentials
for a remote cluster computing service::


..code-block:: python
    class MyCustomBackend(ParallelBackendBase):

        def __init__(self, nesting_level=None, inner_max_num_threads=None,
                     **backend_kwargs):
           super().__init__(
                nesting_level=nesting_level,
                inner_max_num_threads=inner_max_num_threads
            )
            # These arguments are the ones provided in the parallel_config
            # context manager
            self.backend_kwargs = backend_kwargs

        def configure(self, n_jobs=1, parallel=None, **backend_kwargs):
            """Configure the backend for a specific instance of Parallel."""
            self.n_jobs = n_jobs

            self._executor = ThreadPoolExecutor(n_jobs, **backend_kwargs)

        def get_effective_n_jobs(self, n_jobs):
            """Determine the number of jobs that can be run in parallel."""
            return n_jobs

        def submit(self, func, callback):
            """Submit a job to be run by the backend.

            The callback argument is a callable that should be called when the
            computation is done. The callback will be calling `retrieve_result_callback` method is supports_retrieve_callback is True.

            This method should return a future-like object that can be used to
            retrieve the result of the computation.
            """

    register_parallel_backend('custom', MyCustomBackend)
