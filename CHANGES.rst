Latest changes
==============

Development version
-------------------

- Make sure that joblib works even when multiprocessing is not available,
  for instance with Pyodide
  https://github.com/joblib/joblib/pull/1256

- Avoid unnecessary warnings when workers and main process delete
  the temporary memmap folder contents concurrently.
  https://github.com/joblib/joblib/pull/1263

- Vendor loky 3.1.0 with several fixes to more robustly forcibly terminate
  worker processes in case of a crash.
  https://github.com/joblib/joblib/pull/1269

- Fix memory alignment bug for pickles containing numpy arrays.
  This is especially important when loading the pickle with
  ``mmap_mode != None`` as the resulting ``numpy.memmap`` object
  would not be able to correct the misalignment without performing
  a memory copy.
  This bug would cause invalid computation and segmentation faults
  with native code that would directly access the underlying data
  buffer of a numpy array, for instance C/C++/Cython code compiled
  with older GCC versions or some old OpenBLAS written in platform
  specific assembly.
  https://github.com/joblib/joblib/pull/1254

Release 1.1.0
--------------

- Fix byte order inconsistency issue during deserialization using joblib.load
  in cross-endian environment: the numpy arrays are now always loaded to
  use the system byte order, independently of the byte order of the system
  that serialized the pickle.
  https://github.com/joblib/joblib/pull/1181

- Fix joblib.Memory bug with the ``ignore`` parameter when the cached function
  is a decorated function.
  https://github.com/joblib/joblib/pull/1165

- Fix `joblib.Memory` to properly handle caching for functions defined
  interactively in a IPython session or in Jupyter notebook cell.
  https://github.com/joblib/joblib/pull/1214

- Update vendored loky (from version 2.9 to 3.0) and cloudpickle (from
  version 1.6 to 2.0)
  https://github.com/joblib/joblib/pull/1218

Release 1.0.1
-------------

- Add check_call_in_cache method to check cache without calling function.
  https://github.com/joblib/joblib/pull/820
 
- dask: avoid redundant scattering of large arguments to make a more
  efficient use of the network resources and avoid crashing dask with
  "OSError: [Errno 55] No buffer space available"
  or "ConnectionResetError: [Errno 104] connection reset by peer".
  https://github.com/joblib/joblib/pull/1133

Release 1.0.0
-------------

- Make `joblib.hash` and `joblib.Memory` caching system compatible with `numpy
  >= 1.20.0`. Also make it explicit in the documentation that users should now
  expect to have their `joblib.Memory` cache invalidated when either `joblib`
  or a third party library involved in the cached values definition is
  upgraded.  In particular, users updating `joblib` to a release that includes
  this fix will see their previous cache invalidated if they contained
  reference to `numpy` objects. 
  https://github.com/joblib/joblib/pull/1136

- Remove deprecated `check_pickle` argument in `delayed`.
  https://github.com/joblib/joblib/pull/903

Release 0.17.0
--------------

- Fix a spurious invalidation of `Memory.cache`'d functions called with
  `Parallel` under Jupyter or IPython.
  https://github.com/joblib/joblib/pull/1093

- Bump vendored loky to 2.9.0 and cloudpickle to 1.6.0. In particular
  this fixes a problem to add compat for Python 3.9.

Release 0.16.0
--------------

- Fix a problem in the constructors of of Parallel backends classes that
  inherit from the `AutoBatchingMixin` that prevented the dask backend to
  properly batch short tasks.
  https://github.com/joblib/joblib/pull/1062

- Fix a problem in the way the joblib dask backend batches calls that would
  badly interact with the dask callable pickling cache and lead to wrong
  results or errors.
  https://github.com/joblib/joblib/pull/1055

- Prevent a dask.distributed bug from surfacing in joblib's dask backend
  during nested Parallel calls (due to joblib's auto-scattering feature)
  https://github.com/joblib/joblib/pull/1061

- Workaround for a race condition after Parallel calls with the dask backend
  that would cause low level warnings from asyncio coroutines:
  https://github.com/joblib/joblib/pull/1078

Release 0.15.1
--------------

- Make joblib work on Python 3 installation that do not ship with the lzma
  package in their standard library.

Release 0.15.0
--------------

- Drop support for Python 2 and Python 3.5. All objects in
  ``joblib.my_exceptions`` and ``joblib.format_stack`` are now deprecated and
  will be removed in joblib 0.16. Note that no deprecation warning will be
  raised for these objects Python < 3.7.
  https://github.com/joblib/joblib/pull/1018

- Fix many bugs related to the temporary files and folder generated when
  automatically memory mapping large numpy arrays for efficient inter-process
  communication. In particular, this would cause `PermissionError` exceptions
  to be raised under Windows and large leaked files in `/dev/shm` under Linux
  in case of crash.
  https://github.com/joblib/joblib/pull/966

- Make the dask backend collect results as soon as they complete
  leading to a performance improvement:
  https://github.com/joblib/joblib/pull/1025

- Fix the number of jobs reported by ``effective_n_jobs`` when ``n_jobs=None``
  called in a parallel backend context.
  https://github.com/joblib/joblib/pull/985

- Upgraded vendored cloupickle to 1.4.1 and loky to 2.8.0. This allows for
  Parallel calls of dynamically defined functions with type annotations
  in particular.


Release 0.14.1
--------------

- Configure the loky workers' environment to mitigate oversubsription with
  nested multi-threaded code in the following case:

  - allow for a suitable number of threads for numba (``NUMBA_NUM_THREADS``);

  - enable Interprocess Communication for scheduler coordination when the
    nested code uses Threading Building Blocks (TBB) (``ENABLE_IPC=1``)

  https://github.com/joblib/joblib/pull/951

- Fix a regression where the loky backend was not reusing previously
  spawned workers.
  https://github.com/joblib/joblib/pull/968

- Revert https://github.com/joblib/joblib/pull/847 to avoid using
  `pkg_resources` that introduced a performance regression under Windows:
  https://github.com/joblib/joblib/issues/965

Release 0.14.0
--------------

- Improved the load balancing between workers to avoid stranglers caused by an
  excessively large batch size when the task duration is varying significantly
  (because of the combined use of ``joblib.Parallel`` and ``joblib.Memory``
  with a partially warmed cache for instance).
  https://github.com/joblib/joblib/pull/899

- Add official support for Python 3.8: fixed protocol number in `Hasher`
  and updated tests.

- Fix a deadlock when using the dask backend (when scattering large numpy
  arrays).
  https://github.com/joblib/joblib/pull/914

- Warn users that they should never use `joblib.load` with files from
  untrusted sources. Fix security related API change introduced in numpy
  1.6.3 that would prevent using joblib with recent numpy versions.
  https://github.com/joblib/joblib/pull/879

- Upgrade to cloudpickle 1.1.1 that add supports for the upcoming
  Python 3.8 release among other things.
  https://github.com/joblib/joblib/pull/878

- Fix semaphore availability checker to avoid spawning resource trackers
  on module import.
  https://github.com/joblib/joblib/pull/893

- Fix the oversubscription protection to only protect against nested
  `Parallel` calls. This allows `joblib` to be run in background threads.
  https://github.com/joblib/joblib/pull/934

- Fix `ValueError` (negative dimensions) when pickling large numpy arrays on
  Windows.
  https://github.com/joblib/joblib/pull/920

- Upgrade to loky 2.6.0 that add supports for the setting environment variables
  in child before loading any module.
  https://github.com/joblib/joblib/pull/940

- Fix the oversubscription protection for native libraries using threadpools
  (OpenBLAS, MKL, Blis and OpenMP runtimes).
  The maximal number of threads is can now be set in children using the
  ``inner_max_num_threads`` in ``parallel_backend``. It defaults to
  ``cpu_count() // n_jobs``.
  https://github.com/joblib/joblib/pull/940


Release 0.13.2
--------------

Pierre Glaser

   Upgrade to cloudpickle 0.8.0

   Add a non-regression test related to joblib issues #836 and #833, reporting
   that cloudpickle versions between 0.5.4 and 0.7 introduced a bug where
   global variables changes in a parent process between two calls to
   joblib.Parallel would not be propagated into the workers


Release 0.13.1
--------------

Pierre Glaser

   Memory now accepts pathlib.Path objects as ``location`` parameter.
   Also, a warning is raised if the returned backend is None while
   ``location`` is not None.

Olivier Grisel

   Make ``Parallel`` raise an informative ``RuntimeError`` when the
   active parallel backend has zero worker.

   Make the ``DaskDistributedBackend`` wait for workers before trying to
   schedule work. This is useful in particular when the workers are
   provisionned dynamically but provisionning is not immediate (for
   instance using Kubernetes, Yarn or an HPC job queue).


Release 0.13.0
--------------

Thomas Moreau

   Include loky 2.4.2 with default serialization with ``cloudpickle``.
   This can be tweaked with the environment variable ``LOKY_PICKLER``.

Thomas Moreau

   Fix nested backend in SequentialBackend to avoid changing the default
   backend to Sequential. (#792)

Thomas Moreau, Olivier Grisel

    Fix nested_backend behavior to avoid setting the default number of
    workers to -1 when the backend is not dask. (#784)

Release 0.12.5
--------------

Thomas Moreau, Olivier Grisel

    Include loky 2.3.1 with better error reporting when a worker is
    abruptly terminated. Also fixes spurious debug output.


Pierre Glaser

    Include cloudpickle 0.5.6. Fix a bug with the handling of global
    variables by locally defined functions.


Release 0.12.4
--------------

Thomas Moreau, Pierre Glaser, Olivier Grisel

    Include loky 2.3.0 with many bugfixes, notably w.r.t. when setting
    non-default multiprocessing contexts. Also include improvement on
    memory management of long running worker processes and fixed issues
    when using the loky backend under PyPy.


Maxime Weyl

    Raises a more explicit exception when a corrupted MemorizedResult is loaded.

Maxime Weyl

    Loading a corrupted cached file with mmap mode enabled would
    recompute the results and return them without memory mapping.


Release 0.12.3
--------------

Thomas Moreau

    Fix joblib import setting the global start_method for multiprocessing.

Alexandre Abadie

    Fix MemorizedResult not picklable (#747).

Loïc Estève

    Fix Memory, MemorizedFunc and MemorizedResult round-trip pickling +
    unpickling (#746).

James Collins

    Fixed a regression in Memory when positional arguments are called as
    kwargs several times with different values (#751).

Thomas Moreau and Olivier Grisel

    Integration of loky 2.2.2 that fixes issues with the selection of the
    default start method and improve the reporting when calling functions
    with arguments that raise an exception when unpickling.


Maxime Weyl

    Prevent MemorizedFunc.call_and_shelve from loading cached results to
    RAM when not necessary. Results in big performance improvements


Release 0.12.2
--------------

Olivier Grisel

   Integrate loky 2.2.0 to fix regression with unpicklable arguments and
   functions reported by users (#723, #643).

   Loky 2.2.0 also provides a protection against memory leaks long running
   applications when psutil is installed (reported as #721).

   Joblib now includes the code for the dask backend which has been updated
   to properly handle nested parallelism and data scattering at the same
   time (#722).

Alexandre Abadie and Olivier Grisel

   Restored some private API attribute and arguments
   (`MemorizedResult.argument_hash` and `BatchedCalls.__init__`'s
   `pickle_cache`) for backward compat. (#716, #732).


Joris Van den Bossche

   Fix a deprecation warning message (for `Memory`'s `cachedir`) (#720).


Release 0.12.1
--------------

Thomas Moreau

    Make sure that any exception triggered when serializing jobs in the queue
    will be wrapped as a PicklingError as in past versions of joblib.

Noam Hershtig

    Fix kwonlydefaults key error in filter_args (#715)


Release 0.12
------------

Thomas Moreau

    Implement the ``'loky'`` backend with @ogrisel. This backend relies on
    a robust implementation of ``concurrent.futures.ProcessPoolExecutor``
    with spawned processes that can be reused across the ``Parallel``
    calls. This fixes the bad integration with third paty libraries relying on
    thread pools, described in https://pythonhosted.org/joblib/parallel.html#bad-interaction-of-multiprocessing-and-third-party-libraries

    Limit the number of threads used in worker processes by C-libraries that
    relies on threadpools. This functionality works for MKL, OpenBLAS, OpenMP
    and Accelerated.

Elizabeth Sander

    Prevent numpy arrays with the same shape and data from hashing to
    the same memmap, to prevent jobs with preallocated arrays from
    writing over each other.

Olivier Grisel

    Reduce overhead of automatic memmap by removing the need to hash the
    array.

    Make ``Memory.cache`` robust to ``PermissionError (errno 13)`` under
    Windows when run in combination with ``Parallel``.

    The automatic array memory mapping feature of ``Parallel`` does no longer
    use ``/dev/shm`` if it is too small (less than 2 GB). In particular in
    docker containers ``/dev/shm`` is only 64 MB by default which would cause
    frequent failures when running joblib in Docker containers.

    Make it possible to hint for thread-based parallelism with
    ``prefer='threads'`` or enforce shared-memory semantics with
    ``require='sharedmem'``.

    Rely on the built-in exception nesting system of Python 3 to preserve
    traceback information when an exception is raised on a remote worker
    process. This avoid verbose and redundant exception reports under
    Python 3.

    Preserve exception type information when doing nested Parallel calls
    instead of mapping the exception to the generic ``JoblibException`` type.


Alexandre Abadie

    Introduce the concept of 'store' and refactor the ``Memory`` internal
    storage implementation to make it accept extra store backends for caching
    results. ``backend`` and ``backend_options`` are the new options added to
    ``Memory`` to specify and configure a store backend.

    Add the ``register_store_backend`` function to extend the store backend
    used by default with Memory. This default store backend is named 'local'
    and corresponds to the local filesystem.

    The store backend API is experimental and thus is subject to change in the
    future without deprecation.

    The ``cachedir`` parameter of ``Memory`` is now marked as deprecated, use
    ``location`` instead.

    Add support for LZ4 compression if ``lz4`` package is installed.

    Add ``register_compressor`` function for extending available compressors.

    Allow passing a string to ``compress`` parameter in ``dump`` function. This
    string should correspond to the compressor used (e.g. zlib, gzip, lz4,
    etc). The default compression level is used in this case.

Matthew Rocklin

    Allow ``parallel_backend`` to be used globally instead of only as a context
    manager.
    Support lazy registration of external parallel backends

Release 0.11
------------

Alexandre Abadie

    Remove support for python 2.6

Alexandre Abadie

    Remove deprecated `format_signature`, `format_call` and `load_output`
    functions from Memory API.

Loïc Estève

    Add initial implementation of LRU cache cleaning. You can specify
    the size limit of a ``Memory`` object via the ``bytes_limit``
    parameter and then need to clean explicitly the cache via the
    ``Memory.reduce_size`` method.

Olivier Grisel

    Make the multiprocessing backend work even when the name of the main
    thread is not the Python default. Thanks to Roman Yurchak for the
    suggestion.

Karan Desai

    pytest is used to run the tests instead of nosetests.
    ``python setup.py test`` or ``python setup.py nosetests`` do not work
    anymore, run ``pytest joblib`` instead.

Loïc Estève

    An instance of ``joblib.ParallelBackendBase`` can be passed into
    the ``parallel`` argument in ``joblib.Parallel``.


Loïc Estève

    Fix handling of memmap objects with offsets greater than
    mmap.ALLOCATIONGRANULARITY in ``joblib.Parallel``. See
    https://github.com/joblib/joblib/issues/451 for more details.

Loïc Estève

    Fix performance regression in ``joblib.Parallel`` with
    n_jobs=1. See https://github.com/joblib/joblib/issues/483 for more
    details.

Loïc Estève

    Fix race condition when a function cached with
    ``joblib.Memory.cache`` was used inside a ``joblib.Parallel``. See
    https://github.com/joblib/joblib/issues/490 for more details.

Release 0.10.3
--------------

Loïc Estève

    Fix tests when multiprocessing is disabled via the
    JOBLIB_MULTIPROCESSING environment variable.

harishmk

    Remove warnings in nested Parallel objects when the inner Parallel
    has n_jobs=1. See https://github.com/joblib/joblib/pull/406 for
    more details.

Release 0.10.2
--------------

Loïc Estève

    FIX a bug in stack formatting when the error happens in a compiled
    extension. See https://github.com/joblib/joblib/pull/382 for more
    details.

Vincent Latrouite

    FIX a bug in the constructor of BinaryZlibFile that would throw an
    exception when passing unicode filename (Python 2 only).
    See https://github.com/joblib/joblib/pull/384 for more details.

Olivier Grisel

    Expose :class:`joblib.parallel.ParallelBackendBase` and
    :class:`joblib.parallel.AutoBatchingMixin` in the public API to
    make them officially re-usable by backend implementers.


Release 0.10.0
--------------

Alexandre Abadie

    ENH: joblib.dump/load now accept file-like objects besides filenames.
    https://github.com/joblib/joblib/pull/351 for more details.

Niels Zeilemaker and Olivier Grisel

    Refactored joblib.Parallel to enable the registration of custom
    computational backends.
    https://github.com/joblib/joblib/pull/306
    Note the API to register custom backends is considered experimental
    and subject to change without deprecation.

Alexandre Abadie

    Joblib pickle format change: joblib.dump always create a single pickle file
    and joblib.dump/joblib.save never do any memory copy when writing/reading
    pickle files. Reading pickle files generated with joblib versions prior
    to 0.10 will be supported for a limited amount of time, we advise to
    regenerate them from scratch when convenient.
    joblib.dump and joblib.load also support pickle files compressed using
    various strategies: zlib, gzip, bz2, lzma and xz. Note that lzma and xz are
    only available with python >= 3.3.
    https://github.com/joblib/joblib/pull/260 for more details.

Antony Lee

    ENH: joblib.dump/load now accept pathlib.Path objects as filenames.
    https://github.com/joblib/joblib/pull/316 for more details.

Olivier Grisel

    Workaround for "WindowsError: [Error 5] Access is denied" when trying to
    terminate a multiprocessing pool under Windows:
    https://github.com/joblib/joblib/issues/354


Release 0.9.4
-------------

Olivier Grisel

    FIX a race condition that could cause a joblib.Parallel to hang
    when collecting the result of a job that triggers an exception.
    https://github.com/joblib/joblib/pull/296

Olivier Grisel

    FIX a bug that caused joblib.Parallel to wrongly reuse previously
    memmapped arrays instead of creating new temporary files.
    https://github.com/joblib/joblib/pull/294 for more details.

Loïc Estève

    FIX for raising non inheritable exceptions in a Parallel call. See
    https://github.com/joblib/joblib/issues/269 for more details.

Alexandre Abadie

    FIX joblib.hash error with mixed types sets and dicts containing mixed
    types keys when using Python 3.
    see https://github.com/joblib/joblib/issues/254

Loïc Estève

    FIX joblib.dump/load for big numpy arrays with dtype=object. See
    https://github.com/joblib/joblib/issues/220 for more details.

Loïc Estève

    FIX joblib.Parallel hanging when used with an exhausted
    iterator. See https://github.com/joblib/joblib/issues/292 for more
    details.

Release 0.9.3
-------------

Olivier Grisel

    Revert back to the ``fork`` start method (instead of
    ``forkserver``) as the latter was found to cause crashes in
    interactive Python sessions.

Release 0.9.2
-------------

Loïc Estève

    Joblib hashing now uses the default pickle protocol (2 for Python
    2 and 3 for Python 3). This makes it very unlikely to get the same
    hash for a given object under Python 2 and Python 3.

    In particular, for Python 3 users, this means that the output of
    joblib.hash changes when switching from joblib 0.8.4 to 0.9.2 . We
    strive to ensure that the output of joblib.hash does not change
    needlessly in future versions of joblib but this is not officially
    guaranteed.

Loïc Estève

    Joblib pickles generated with Python 2 can not be loaded with
    Python 3 and the same applies for joblib pickles generated with
    Python 3 and loaded with Python 2.

    During the beta period 0.9.0b2 to 0.9.0b4, we experimented with
    a joblib serialization that aimed to make pickles serialized with
    Python 3 loadable under Python 2. Unfortunately this serialization
    strategy proved to be too fragile as far as the long-term
    maintenance was concerned (For example see
    https://github.com/joblib/joblib/pull/243). That means that joblib
    pickles generated with joblib 0.9.0bN can not be loaded under
    joblib 0.9.2. Joblib beta testers, who are the only ones likely to
    be affected by this, are advised to delete their joblib cache when
    they upgrade from 0.9.0bN to 0.9.2.

Arthur Mensch

    Fixed a bug with ``joblib.hash`` that used to return unstable values for
    strings and numpy.dtype instances depending on interning states.

Olivier Grisel

    Make joblib use the 'forkserver' start method by default under Python 3.4+
    to avoid causing crash with 3rd party libraries (such as Apple vecLib /
    Accelerate or the GCC OpenMP runtime) that use an internal thread pool that
    is not not reinitialized when a ``fork`` system call happens.

Olivier Grisel

    New context manager based API (``with`` block) to re-use
    the same pool of workers across consecutive parallel calls.

Vlad Niculae and Olivier Grisel

    Automated batching of fast tasks into longer running jobs to
    hide multiprocessing dispatching overhead when possible.

Olivier Grisel

    FIX make it possible to call ``joblib.load(filename, mmap_mode='r')``
    on pickled objects that include a mix of arrays of both
    memory memmapable dtypes and object dtype.


Release 0.8.4
-------------

2014-11-20
Olivier Grisel

    OPTIM use the C-optimized pickler under Python 3

    This makes it possible to efficiently process parallel jobs that deal with
    numerous Python objects such as large dictionaries.


Release 0.8.3
-------------

2014-08-19
Olivier Grisel

    FIX disable memmapping for object arrays

2014-08-07
Lars Buitinck

    MAINT NumPy 1.10-safe version comparisons


2014-07-11
Olivier Grisel

    FIX #146: Heisen test failure caused by thread-unsafe Python lists

    This fix uses a queue.Queue datastructure in the failing test. This
    datastructure is thread-safe thanks to an internal Lock. This Lock instance
    not picklable hence cause the picklability check of delayed to check fail.

    When using the threading backend, picklability is no longer required, hence
    this PRs give the user the ability to disable it on a case by case basis.


Release 0.8.2
-------------

2014-06-30
Olivier Grisel

    BUG: use mmap_mode='r' by default in Parallel and MemmappingPool

    The former default of mmap_mode='c' (copy-on-write) caused
    problematic use of the paging file under Windows.

2014-06-27
Olivier Grisel

    BUG: fix usage of the /dev/shm folder under Linux


Release 0.8.1
-------------

2014-05-29
Gael Varoquaux

    BUG: fix crash with high verbosity


Release 0.8.0
-------------

2014-05-14
Olivier Grisel

   Fix a bug in exception reporting under Python 3

2014-05-10
Olivier Grisel

   Fixed a potential segfault when passing non-contiguous memmap
   instances.

2014-04-22
Gael Varoquaux

    ENH: Make memory robust to modification of source files while the
    interpreter is running. Should lead to less spurious cache flushes
    and recomputations.


2014-02-24
Philippe Gervais

   New ``Memory.call_and_shelve`` API to handle memoized results by
   reference instead of by value.


Release 0.8.0a3
---------------

2014-01-10
Olivier Grisel & Gael Varoquaux

   FIX #105: Race condition in task iterable consumption when
   pre_dispatch != 'all' that could cause crash with error messages "Pools
   seems closed" and "ValueError: generator already executing".

2014-01-12
Olivier Grisel

   FIX #72: joblib cannot persist "output_dir" keyword argument.


Release 0.8.0a2
---------------

2013-12-23
Olivier Grisel

    ENH: set default value of Parallel's max_nbytes to 100MB

    Motivation: avoid introducing disk latency on medium sized
    parallel workload where memory usage is not an issue.

    FIX: properly handle the JOBLIB_MULTIPROCESSING env variable

    FIX: timeout test failures under windows


Release 0.8.0a
--------------

2013-12-19
Olivier Grisel

    FIX: support the new Python 3.4 multiprocessing API


2013-12-05
Olivier Grisel

    ENH: make Memory respect mmap_mode at first call too

    ENH: add a threading based backend to Parallel

    This is low overhead alternative backend to the default multiprocessing
    backend that is suitable when calling compiled extensions that release
    the GIL.


Author: Dan Stahlke <dan@stahlke.org>
Date:   2013-11-08

    FIX: use safe_repr to print arg vals in trace

    This fixes a problem in which extremely long (and slow) stack traces would
    be produced when function parameters are large numpy arrays.


2013-09-10
Olivier Grisel

    ENH: limit memory copy with Parallel by leveraging numpy.memmap when
    possible


Release 0.7.1
---------------

2013-07-25
Gael Varoquaux

    MISC: capture meaningless argument (n_jobs=0) in Parallel

2013-07-09
Lars Buitinck

    ENH Handles tuples, sets and Python 3's dict_keys type the same as
    lists. in pre_dispatch

2013-05-23
Martin Luessi

    ENH: fix function caching for IPython

Release 0.7.0
---------------

**This release drops support for Python 2.5 in favor of support for
Python 3.0**

2013-02-13
Gael Varoquaux

    BUG: fix nasty hash collisions

2012-11-19
Gael Varoquaux

    ENH: Parallel: Turn of pre-dispatch for already expanded lists


Gael Varoquaux
2012-11-19

    ENH: detect recursive sub-process spawning, as when people do not
    protect the __main__ in scripts under Windows, and raise a useful
    error.


Gael Varoquaux
2012-11-16

    ENH: Full python 3 support

Release 0.6.5
---------------

2012-09-15
Yannick Schwartz

    BUG: make sure that sets and dictionaries give reproducible hashes


2012-07-18
Marek Rudnicki

    BUG: make sure that object-dtype numpy array hash correctly

2012-07-12
GaelVaroquaux

    BUG: Bad default n_jobs for Parallel

Release 0.6.4
---------------

2012-05-07
Vlad Niculae

    ENH: controlled randomness in tests and doctest fix

2012-02-21
GaelVaroquaux

    ENH: add verbosity in memory

2012-02-21
GaelVaroquaux

    BUG: non-reproducible hashing: order of kwargs

    The ordering of a dictionary is random. As a result the function hashing
    was not reproducible. Pretty hard to test

Release 0.6.3
---------------

2012-02-14
GaelVaroquaux

    BUG: fix joblib Memory pickling

2012-02-11
GaelVaroquaux

    BUG: fix hasher with Python 3

2012-02-09
GaelVaroquaux

    API: filter_args:  `*args, **kwargs -> args, kwargs`

Release 0.6.2
---------------

2012-02-06
Gael Varoquaux

    BUG: make sure Memory pickles even if cachedir=None

Release 0.6.1
---------------

Bugfix release because of a merge error in release 0.6.0

Release 0.6.0
---------------

**Beta 3**

2012-01-11
Gael Varoquaux

    BUG: ensure compatibility with old numpy

    DOC: update installation instructions

    BUG: file semantic to work under Windows

2012-01-10
Yaroslav Halchenko

    BUG: a fix toward 2.5 compatibility

**Beta 2**

2012-01-07
Gael Varoquaux

    ENH: hash: bugware to be able to hash objects defined interactively
    in IPython

2012-01-07
Gael Varoquaux

    ENH: Parallel: warn and not fail for nested loops

    ENH: Parallel: n_jobs=-2 now uses all CPUs but one

2012-01-01
Juan Manuel Caicedo Carvajal and Gael Varoquaux

    ENH: add verbosity levels in Parallel

Release 0.5.7
---------------

2011-12-28
Gael varoquaux

    API: zipped -> compress

2011-12-26
Gael varoquaux

    ENH: Add a zipped option to Memory

    API: Memory no longer accepts save_npy

2011-12-22
Kenneth C. Arnold and Gael varoquaux

    BUG: fix numpy_pickle for array subclasses

2011-12-21
Gael varoquaux

    ENH: add zip-based pickling

2011-12-19
Fabian Pedregosa

    Py3k: compatibility fixes.
    This makes run fine the tests test_disk and test_parallel

Release 0.5.6
---------------

2011-12-11
Lars Buitinck

    ENH: Replace os.path.exists before makedirs with exception check
    New disk.mkdirp will fail with other errnos than EEXIST.

2011-12-10
Bala Subrahmanyam Varanasi

    MISC: pep8 compliant


Release 0.5.5
---------------

2011-19-10
Fabian Pedregosa

    ENH: Make joblib installable under Python 3.X

Release 0.5.4
---------------

2011-09-29
Jon Olav Vik

    BUG: Make mangling path to filename work on Windows

2011-09-25
Olivier Grisel

    FIX: doctest heisenfailure on execution time

2011-08-24
Ralf Gommers

    STY: PEP8 cleanup.


Release 0.5.3
---------------

2011-06-25
Gael varoquaux

   API: All the useful symbols in the __init__


Release 0.5.2
---------------

2011-06-25
Gael varoquaux

    ENH: Add cpu_count

2011-06-06
Gael varoquaux

    ENH: Make sure memory hash in a reproducible way


Release 0.5.1
---------------

2011-04-12
Gael varoquaux

    TEST: Better testing of parallel and pre_dispatch

Yaroslav Halchenko
2011-04-12

    DOC: quick pass over docs -- trailing spaces/spelling

Yaroslav Halchenko
2011-04-11

    ENH: JOBLIB_MULTIPROCESSING env var to disable multiprocessing from the
    environment

Alexandre Gramfort
2011-04-08

    ENH : adding log message to know how long it takes to load from disk the
    cache


Release 0.5.0
---------------

2011-04-01
Gael varoquaux

    BUG: pickling MemoizeFunc does not store timestamp

2011-03-31
Nicolas Pinto

    TEST: expose hashing bug with cached method

2011-03-26...2011-03-27
Pietro Berkes

    BUG: fix error management in rm_subdirs
    BUG: fix for race condition during tests in mem.clear()

Gael varoquaux
2011-03-22...2011-03-26

    TEST: Improve test coverage and robustness

Gael varoquaux
2011-03-19

    BUG: hashing functions with only \*var \**kwargs

Gael varoquaux
2011-02-01... 2011-03-22

    BUG: Many fixes to capture interprocess race condition when mem.cache
    is used by several processes on the same cache.

Fabian Pedregosa
2011-02-28

    First work on Py3K compatibility

Gael varoquaux
2011-02-27

    ENH: pre_dispatch in parallel: lazy generation of jobs in parallel
    for to avoid drowning memory.

GaelVaroquaux
2011-02-24

    ENH: Add the option of overloading the arguments of the mother
    'Memory' object in the cache method that is doing the decoration.

Gael varoquaux
2010-11-21

    ENH: Add a verbosity level for more verbosity

Release 0.4.6
----------------

Gael varoquaux
2010-11-15

    ENH: Deal with interruption in parallel

Gael varoquaux
2010-11-13

    BUG: Exceptions raised by Parallel when n_job=1 are no longer captured.

Gael varoquaux
2010-11-13

    BUG: Capture wrong arguments properly (better error message)


Release 0.4.5
----------------

Pietro Berkes
2010-09-04

    BUG: Fix Windows peculiarities with path separators and file names
    BUG: Fix more windows locking bugs

Gael varoquaux
2010-09-03

    ENH: Make sure that exceptions raised in Parallel also inherit from
    the original exception class
    ENH: Add a shadow set of exceptions

Fabian Pedregosa
2010-09-01

    ENH: Clean up the code for parallel. Thanks to Fabian Pedregosa for
    the patch.


Release 0.4.4
----------------

Gael varoquaux
2010-08-23

    BUG: Fix Parallel on computers with only one CPU, for n_jobs=-1.

Gael varoquaux
2010-08-02

    BUG: Fix setup.py for extra setuptools args.

Gael varoquaux
2010-07-29

    MISC: Silence tests (and hopefully Yaroslav :P)

Release 0.4.3
----------------

Gael Varoquaux
2010-07-22

    BUG: Fix hashing for function with a side effect modifying their input
    argument. Thanks to Pietro Berkes for reporting the bug and proving the
    patch.

Release 0.4.2
----------------

Gael Varoquaux
2010-07-16

    BUG: Make sure that joblib still works with Python2.5. => release 0.4.2

Release 0.4.1
----------------
