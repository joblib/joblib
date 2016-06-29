Latest changes
===============

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

    BUG: use mmap_mode='r' by default in Parallel and MemmapingPool

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

    BUG: make sure that sets and dictionnaries give reproducible hashes


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

    The ordering of a dictionnary is random. As a result the function hashing
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

   API: All the usefull symbols in the __init__


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

    MISC: Silence tests (and hopefuly Yaroslav :P)

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
