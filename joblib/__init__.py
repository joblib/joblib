"""
Joblib: a package for dealing with long running calculation.

Sub-modules
-------------

  * memory: a context for caching calls to function to the disk.
    Unlike the memoize pattern, this is suited for persistent use with
    big data or arrays.

  * logger: helper objects for logging.

"""

__version__ = '0.2a'


from memory import Memory

from logger import PrintTime

