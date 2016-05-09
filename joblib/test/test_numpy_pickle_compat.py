"""Test the old numpy pickler, compatibility version."""

import shutil
import os
import random
import nose

from tempfile import mkdtemp

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
from joblib import numpy_pickle_compat


###############################################################################
# Test fixtures

env = dict()


def setup_module():
    """Test setup."""
    env['dir'] = mkdtemp()
    env['filename'] = os.path.join(env['dir'], 'test.pkl')


def teardown_module():
    """Test teardown."""
    shutil.rmtree(env['dir'])


def test_z_file():
    # Test saving and loading data with Zfiles.
    filename = env['filename'] + str(random.randint(0, 1000))
    data = numpy_pickle_compat.asbytes('Foo, \n Bar, baz, \n\nfoobar')
    with open(filename, 'wb') as f:
        numpy_pickle_compat.write_zfile(f, data)
    with open(filename, 'rb') as f:
        data_read = numpy_pickle_compat.read_zfile(f)
    nose.tools.assert_equal(data, data_read)
