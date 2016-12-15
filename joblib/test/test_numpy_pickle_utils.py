from joblib import numpy_pickle_utils


def test_binary_zlib_file(tmpdir):
    """Testing creation of files depending on the type of the filenames."""
    # Testing str filename.
    binary_file = numpy_pickle_utils.BinaryZlibFile(
        tmpdir.join('test').strpath,
        mode='wb')
    binary_file.close()

    # Testing unicode filename.
    binary_file = numpy_pickle_utils.BinaryZlibFile(
        tmpdir.join(u'test').strpath,
        mode='wb')
    binary_file.close()
