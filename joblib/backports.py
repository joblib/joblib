"""
Backports of fixes for joblib dependencies
"""

from distutils.version import LooseVersion

try:
    import numpy as np

    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C'):
        """Backport of numpy memmap offset fix.

        See https://github.com/numpy/numpy/pull/8443 for more details.

        The numpy fix will be available in numpy 1.13.
        """
        mm = np.memmap(filename, dtype=dtype, mode=mode, offset=offset,
                       shape=shape, order=order)
        if LooseVersion(np.__version__) < '1.13':
            mm.offset = offset
        return mm
except ImportError:
    def make_memmap(filename, dtype='uint8', mode='r+', offset=0,
                    shape=None, order='C'):
        raise NotImplementedError(
            "'joblib.backports.make_memmap' should not be used "
            'if numpy is not installed.')
