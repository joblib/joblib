from joblib import numpy_pickle_utils
from joblib.testing import parametrize


@parametrize('filename', ['test', u'test'])  # testing str and unicode names
def test_binary_zlib_file(tmpdir, filename):
    """Testing creation of files depending on the type of the filenames."""
    binary_file = numpy_pickle_utils.BinaryZlibFile(
        tmpdir.join(filename).strpath,
        mode='wb')
    binary_file.close()
