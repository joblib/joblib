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

# Create a Memory object to test decorated functions.
# We should be careful not to call the decorated functions, so that
# cache directories are not created in the temp dir.
mem = Memory(cachedir=tempfile.gettempdir())

@mem.cache
def g(x):
    return x

################################################################################
# Tests

def test_filter_args():
    yield nose.tools.assert_equal, filter_args(f, ['x'], 0), ([], {})
    nose.tools.assert_equal(filter_args(f, ['x']), ([], {}))


def test_func_name():
    yield nose.tools.assert_equal, 'f', get_func_name(f)[1]
    # Check that we are not confused by the decoration
    yield nose.tools.assert_equal, 'g', get_func_name(g)[1]

