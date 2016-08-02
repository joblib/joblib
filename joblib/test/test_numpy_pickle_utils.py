import shutil
import os
from tempfile import mkdtemp

from joblib import numpy_pickle_utils


###############################################################################
# Test fixtures

env = dict()


def setup_module():
    """Test setup."""
    env['dir'] = mkdtemp()


def teardown_module():
    """Test teardown."""
    shutil.rmtree(env['dir'])


def test_binary_zlib_file():
    """Testing creation of files depending on the type of the filenames."""
    # Testing str filename.
    binary_file = numpy_pickle_utils.BinaryZlibFile(
        os.path.join(env['dir'], 'test'),
        mode='wb')
    binary_file.close()

    # Testing unicode filename.
    binary_file = numpy_pickle_utils.BinaryZlibFile(
        os.path.join(env['dir'], u'test'),
        mode='wb')
    binary_file.close()
