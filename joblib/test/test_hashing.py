"""
Test the hashing module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose
import time
import hashlib

from ..hashing import hash

################################################################################
# Helper functions for the tests 
def time_func(func, *args):
    """ Time function func on *args.
    """
    times = list()
    for _ in range(3):
        t1 = time.time()
        func(*args)
        times.append(time.time() - t1)
    return min(times)


def relative_time(func1, func2, *args):
    """ Return the relative time between func1 and func2 applied on
        *args.
    """
    time_func1 = time_func(func1, *args)
    time_func2 = time_func(func2, *args)
    relative_diff = 0.5*( abs(time_func1 - time_func2)
                          /  (time_func1 + time_func2) )
    return relative_diff


################################################################################
# Tests

def test_trival_hash():
    """ Smoke test hash on various types.
    """
    for obj in [1, 1., 1+1j,
                'a', 
                (1, ), [1, ], {1:1},
                None,
                ]:
        yield nose.tools.assert_equal, hash(obj), hash(obj)
    # XXX: Need to check that all these hashes are different, using a
    # double for loop.



def test_hash_numpy():
    """ Test hashing with numpy arrays.
    """
    try:
        import numpy as np
    except ImportError:
        return
    obj = np.random.random((10, 10))
    yield nose.tools.assert_equal, hash(obj), hash(obj)

    # Check the performance: we should not be getting more than a factor
    # of 1.1 compared to directly hashing the array
    """
    In [22]: a = np.random.random(1000000)

    In [23]: %timeit hashlib.md5(a).hexdigest()
    100 loops, best of 3: 20.7 ms per loop

    In [24]: %timeit hashlib.md5(pickle.dumps(a, protocol=2)).hexdigest()
    1 loops, best of 3: 73.1 ms per loop

    In [25]: %timeit hashlib.md5(cPickle.dumps(a, protocol=2)).hexdigest()
    10 loops, best of 3: 53.9 ms per loop

    In [26]: %timeit hash(a)
    100 loops, best of 3: 20.8 ms per loop

    """
    a = np.random.random(1000000)
    md5_hash = lambda x: hashlib.md5(x).hexdigest()

    relative_diff = relative_time(md5_hash, hash, a)
    yield nose.tools.assert_true, relative_diff < 0.05

    # Check that hashing an tuple of 3 arrays takes approximately
    # 3 times as much as hashing one array
    time_hashlib = 3*time_func(md5_hash, a)
    time_hash = time_func(hash, (a, a, a))
    relative_diff = 0.5*( abs(time_hash - time_hashlib)
                          /  (time_hash + time_hashlib) )

    yield nose.tools.assert_true, relative_diff < 0.1

