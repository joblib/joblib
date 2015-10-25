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

def test_format_records():
    try:
        raise ValueError
    except ValueError:
        etb = sys.exc_info()[2]
        records = _fixed_getframes(etb)

        # Modify filenames in traceback records from .py to .pyc
        pyc_records = []
        for record in records:
            if record[1].endswith('.py'):
                record = list(record)
                record[1] += 'c'
                record = tuple(record)
                pyc_records.append(record)

        # Check that no .pyc files are listed in traceback
        for fmt_rec in format_records(pyc_records):
            assert '.pyc in' not in fmt_rec
