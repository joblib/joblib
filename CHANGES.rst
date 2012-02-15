Latest changes
===============

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

