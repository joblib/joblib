"""
Unit tests for the stack formatting utilities
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose
import sys

from joblib.format_stack import safe_repr, _fixed_getframes, format_records


###############################################################################

class Vicious(object):
    def __repr__(self):
        raise ValueError


def test_safe_repr():
    safe_repr(Vicious())


def _change_file_extensions_to_pyc(record):
    _1, filename, _2, _3, _4, _5 = record
    if filename.endswith('.py'):
        filename += 'c'
    return _1, filename, _2, _3, _4, _5


def _raise_exception(a, b):
    """Function that raises with a non trivial call stack
    """
    def helper(a, b):
        raise ValueError('Nope, this can not work')

    helper(a, b)


def test_format_records():
    try:
        _raise_exception('a', 42)
    except ValueError:
        etb = sys.exc_info()[2]
        records = _fixed_getframes(etb)

        # Modify filenames in traceback records from .py to .pyc
        pyc_records = [_change_file_extensions_to_pyc(record)
                       for record in records]

        formatted_records = format_records(pyc_records)

        # Check that the .py file and not the .pyc one is listed in
        # the traceback
        for fmt_rec in formatted_records:
            assert 'test_format_stack.py in' in fmt_rec

        # Check exception stack
        assert "_raise_exception('a', 42)" in formatted_records[0]
        assert 'helper(a, b)' in formatted_records[1]
        assert "a = 'a'" in formatted_records[1]
        assert 'b = 42' in formatted_records[1]
        assert 'Nope, this can not work' in formatted_records[2]
