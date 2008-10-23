"""
Joblib: a package for dealing with long running calculation.

Sub-modules
-------------

  * memoize: An implementation of the memoize pattern: automatic
    caching to disk of functions. Memoize does not work well with
    functions taking arrays as arguments or returning arrays.

  * run_scripts: functions for running scripts.

"""

from run_scripts import default_param, PrintTime, run_script

