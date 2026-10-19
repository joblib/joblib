"""Test the old numpy pickler, compatibility version."""

# numpy_pickle is not a drop-in replacement of pickle, as it takes
# filenames instead of open files as arguments.
import pickle

from joblib import numpy_pickle, numpy_pickle_compat
from joblib.test.common import np, with_numpy
from joblib.testing import warns


def test_z_file(tmpdir):
    # Test saving and loading data with Zfiles.
    filename = tmpdir.join("test.pkl").strpath
    data = numpy_pickle_compat.asbytes("Foo, \n Bar, baz, \n\nfoobar")
    with open(filename, "wb") as f:
        numpy_pickle_compat.write_zfile(f, data)
    with open(filename, "rb") as f:
        data_read = numpy_pickle_compat.read_zfile(f)
    assert data == data_read


@with_numpy
def test_matrix_in_companion_file(tmpdir):
    # The old format keeps each array in a companion .npy file and only records
    # its subclass in an NDArrayWrapper: np.matrix must be rebuilt from it.
    filename = tmpdir.join("test.pkl").strpath
    a = np.matrix([[1, 2], [3, 4]])
    np.save(tmpdir.join("test_01.npy").strpath, a)
    wrapper = numpy_pickle_compat.NDArrayWrapper("test_01.npy", np.matrix)
    with open(filename, "wb") as f:
        numpy_pickle_compat.write_zfile(f, pickle.dumps(wrapper))
    with warns(DeprecationWarning):
        c = numpy_pickle.load(filename)
    assert type(c) is np.matrix
    np.testing.assert_array_equal(c, a)
