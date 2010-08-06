"""
Unit tests for the disk utilities.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose

from ..disk import memstr_to_kbytes

def test_memstr_to_kbytes():
    for text, value in zip(('80G', '1.4M', '120M', '53K'),
                           (80*1024**2, int(1.4*1024), 120*1024, 53)):
        yield nose.tools.assert_equal, memstr_to_kbytes(text), value


