"""
This script is used to generate test data for joblib/test/test_numpy_pickle.py
"""

import sys

# nosetests needs to be able to import this module even when numpy is
# not installed
try:
    import numpy as np
except ImportError:
    np = None

import joblib


def get_joblib_version(joblib_version=joblib.__version__):
    """Normalise joblib version by removing suffix

    >>> get_joblib_version('0.8.4')
    '0.8.4'
    >>> get_joblib_version('0.8.4.dev0')
n    '0.8.4'
    >>> get_joblib_version('0.9.dev0')
    '0.9'
    """
    return '.'.join([x for x in joblib_version.split('.')
                     if not x.startswith('dev')])


def write_test_pickle(to_pickle):
    joblib_version = get_joblib_version()
    py_version = '{0[0]}{0[1]}'.format(sys.version_info)
    pickle_filename = 'joblib_{}_compressed_pickle_py{}.gz'.format(
        joblib_version, py_version)
    joblib.dump(to_pickle, pickle_filename, compress=True)


if __name__ == '__main__':
    to_pickle = [np.arange(5, dtype=np.int64),
                 np.arange(5, dtype=np.float64),
                 # all possible bytes as a byte string
                 # .tostring actually returns bytes and is a
                 # compatibility alias for .tobytes which was
                 # added in 1.9.0
                 np.arange(256, dtype=np.uint8).tostring(),
                 # unicode string with non-ascii chars
                 u"C'est l'\xe9t\xe9 !"]

    write_test_pickle(to_pickle)
