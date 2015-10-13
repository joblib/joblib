"""
This script is used to generate test data for joblib/test/test_numpy_pickle.py
"""

import sys
import re

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
    >>> get_joblib_version('0.8.4b1')
    '0.8.4'
    >>> get_joblib_version('0.9.dev0')
    '0.9'
    """
    matches = [re.match(r'(\d+).*', each)
               for each in joblib_version.split('.')]
    return '.'.join([m.group(1) for m in matches if m is not None])


def write_test_pickle(to_pickle):
    joblib_version = get_joblib_version()
    py_version = '{0[0]}{0[1]}'.format(sys.version_info)
    numpy_version = ''.join(np.__version__.split('.')[:2])
    print('file:', np.__file__)
    pickle_filename = 'joblib_{0}_compressed_pickle_py{1}_np{2}.gz'.format(
        joblib_version, py_version, numpy_version)
    joblib.dump(to_pickle, pickle_filename, compress=True)
    pickle_filename = 'joblib_{0}_pickle_py{1}_np{2}.pkl'.format(
        joblib_version, py_version, numpy_version)
    joblib.dump(to_pickle, pickle_filename, compress=False)

if __name__ == '__main__':
    to_pickle = [np.arange(5, dtype=np.int64),
                 np.arange(5, dtype=np.float64),
                 np.array([1, 'abc', {'a': 1, 'b': 2}]),
                 # all possible bytes as a byte string
                 # .tostring actually returns bytes and is a
                 # compatibility alias for .tobytes which was
                 # added in 1.9.0
                 np.arange(256, dtype=np.uint8).tostring(),
                 # unicode string with non-ascii chars
                 u"C'est l'\xe9t\xe9 !"]

    write_test_pickle(to_pickle)
