"""
Test the func_inspect module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose
import tempfile

from ..func_inspect import filter_args, get_func_name
from ..memory import Memory

################################################################################
# Module-level functions, for tests 
def f(x, y=0):
    pass

def f2(x):
    pass

# Create a Memory object to test decorated functions.
# We should be careful not to call the decorated functions, so that
# cache directories are not created in the temp dir.
mem = Memory(cachedir=tempfile.gettempdir())

@mem.cache
def g(x):
    return x

def h(x, y=0, *args, **kwargs):
    pass

def i(x=1):
    pass

class Klass(object):

    def f(self, x):
        return x

################################################################################
# Tests

def test_filter_args():
    yield nose.tools.assert_equal, filter_args(f, [], 1), {'x': 1, 'y': 0}
    yield nose.tools.assert_equal, filter_args(f, ['x'], 1), {'y': 0}
    yield nose.tools.assert_equal, filter_args(f, ['y'], 0), {'x': 0}
    yield nose.tools.assert_equal, filter_args(f, ['y'], 0, y=1), {'x': 0}
    yield nose.tools.assert_equal, filter_args(f, ['x', 'y'], 0), {}
    yield nose.tools.assert_equal, filter_args(f, [], 0, y=1), {'x':0, 'y':1}

    yield nose.tools.assert_equal, filter_args(i, [], 2), {'x': 2}
    yield nose.tools.assert_equal, filter_args(f2, [], x=1), {'x': 1}


def test_filter_args_method():
    obj = Klass()
    nose.tools.assert_equal(filter_args(obj.f, [], 1), {'x': 1})

def test_filter_varargs():
    yield nose.tools.assert_equal, filter_args(h, [], 1), \
                            {'x': 1, 'y': 0, '*':[], '**':{}}
    yield nose.tools.assert_equal, filter_args(h, [], 1, 2, 3, 4), \
                            {'x': 1, 'y': 2, '*':[3, 4], '**':{}}
    yield nose.tools.assert_equal, filter_args(h, [], 1, 25, ee=2), \
                            {'x': 1, 'y': 25, '*':[], '**':{'ee':2}}
    yield nose.tools.assert_equal, filter_args(h, ['*'], 1, 2, 25, ee=2), \
                            {'x': 1, 'y': 2, '**':{'ee':2}}
    

def test_func_name():
    yield nose.tools.assert_equal, 'f', get_func_name(f)[1]
    # Check that we are not confused by the decoration
    yield nose.tools.assert_equal, 'g', get_func_name(g)[1]

