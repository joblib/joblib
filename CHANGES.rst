Latest changes
===============

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

