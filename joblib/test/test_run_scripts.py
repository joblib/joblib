"""
Test the run_scripts modules.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.



import sys
import tempfile
from StringIO import StringIO

from joblib.run_scripts import run_script

def test_default_param():
    """ Try running a script and checling that the default parameter is
        indeed modified.
    """
    script_file = tempfile.mktemp()
    print >> file(script_file, 'w'), """
from joblib.run_scripts import default_param
print default_param('a', 1)
    """
    my_stdout = StringIO()
    old_stdout = sys.stdout
    sys.stdout = my_stdout
    try:
        run_script(script_file)
        run_script(script_file, a=2)
        assert my_stdout.getvalue() == '1\n2\n'
    finally:
        sys.stdout = old_stdout
    

