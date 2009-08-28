"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
import os
from tempfile import mkdtemp

import nose

from ..logger import PrintTime

################################################################################
# Test fixtures
def setup():
    """ Test setup.
    """
    global cachedir
    cachedir = mkdtemp()
    #cachedir = 'foobar'
    if os.path.exists(cachedir):
        shutil.rmtree(cachedir)
    

def teardown():
    """ Test teardown.
    """
    #return True
    shutil.rmtree(cachedir)


################################################################################
# Tests
def smoke_test_print_time():
    """ A simple smoke test for PrintTime.
    """
    print_time = PrintTime(logfile=os.path.join(cachedir, 'test.log'))
    print_time('Foo')
    # Create a second time, to smoke test log rotation.
    print_time = PrintTime(logfile=os.path.join(cachedir, 'test.log'))
    print_time('Foo')
    # And a third time 
    print_time = PrintTime(logfile=os.path.join(cachedir, 'test.log'))
    print_time('Foo')

