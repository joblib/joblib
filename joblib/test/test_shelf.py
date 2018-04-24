"""
Test the shelf module.
"""

# Copyright (c) 2018 Inria
# License: BSD Style, 3 clauses.

import os
import os.path
import gc

from joblib import Parallel, delayed, shelve, shelve_mmap
from joblib.shelf import JoblibShelf, JoblibShelfFuture
from joblib.test.common import with_numpy, np
from joblib.testing import parametrize


@parametrize('data', [42, 2018.01, "some data",
                      {'a': 1, 'b': 2}, [1, 2, 3, 4]])
def test_shelve_with_standard_data(data):
    future = shelve(data)
    assert future.result() == data


@with_numpy
@parametrize('data', [42, 2018.01, "some data",
                      {'a': 1, 'b': 2}, [1, 2, 3, 4]])
def test_shelve_mmap_with_standard_data(data):
    # no memmaping is possible with standard types
    shelved_data = shelve_mmap(data)
    assert not isinstance(shelved_data, np.memmap)
    assert shelved_data.result() == data


@with_numpy
def test_shelve_with_numpy(tmpdir):
    filename = tmpdir.join('test.pkl').strpath

    for obj in (np.random.random((5, 5)),
                np.matrix(np.zeros(10)),
                np.memmap(filename + 'mmap',
                          mode='w+', shape=4, dtype=np.float)):
        future = shelve(obj)
        data_result = future.result()
        np.testing.assert_array_equal(np.asarray(data_result), obj)


@with_numpy
def test_shelve_parallel():
    data = np.random.random((50, 50))
    future = shelve(data)
    assert isinstance(future, JoblibShelfFuture)
    np.testing.assert_array_equal(future.result(), data)

    mean_expected = [np.mean(line) for line in data]

    def f(future):
        return np.mean(future.result())

    mean_result = Parallel(n_jobs=10, backend='threading')(
                           delayed(f)(shelve(line)) for line in data)
    np.testing.assert_array_equal(mean_result, mean_expected)


@with_numpy
def test_shelve_mmap_with_numpy(tmpdir):
    filename = tmpdir.join('test.pkl').strpath

    for obj in (np.random.random((5, 5)),
                np.matrix(np.zeros(10)),
                np.memmap(filename + 'mmap',
                          mode='w+', shape=4, dtype=np.float)):
        mmap = shelve_mmap(obj)
        np.testing.assert_array_equal(np.asarray(mmap), obj)


@with_numpy
def test_shelve_mmap_with_numpy_obj(tmpdir):
    obj = np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O')
    shelved_data = shelve_mmap(obj)

    assert isinstance(shelved_data, JoblibShelfFuture)
    np.testing.assert_array_equal(shelved_data.result(), obj)


@with_numpy
@parametrize('backend', ["threading", "loky", "multiprocessing"])
def test_shelve_mmap_parallel(backend):
    data = np.random.random((50, 50))
    shelved_data = shelve_mmap(data)
    assert isinstance(shelved_data, np.memmap)
    np.testing.assert_array_equal(np.asarray(shelved_data), data)

    mean_expected = [np.mean(line) for line in data]

    def f(data):
        return np.mean(data)

    mean_result = Parallel(n_jobs=10, backend=backend)(
                           delayed(f)(line) for line in shelved_data)
    np.testing.assert_array_equal(mean_result, mean_expected)


@parametrize('data', [42, 2018.01, "some data",
                      {'a': 1, 'b': 2}, [1, 2, 3, 4]])
def test_shelved_data_directory(tmpdir, data):
    shelf = JoblibShelf(location=tmpdir.strpath, verbose=10)
    shelf_path = shelf.store_backend.location
    future1 = shelf.put(data)
    data_dir_path = os.path.join(shelf.store_backend.location,
                                 future1._data_id)

    # Check pickle file and parent directory exist (with the right name)
    assert os.path.basename(data_dir_path) == future1._data_id
    assert os.path.exists(data_dir_path)
    assert os.path.exists(os.path.join(data_dir_path, 'output.pkl'))

    # Check input data is correctly reloaded
    assert future1.result() == data

    # Add a second reference of the first future
    future2 = future1

    # Deleting future1 should not remove the cached directory
    del future1
    gc.collect()
    assert os.path.exists(data_dir_path)

    # Deleting future2 should remove the cached directory since it's the last
    # reference on the initial future
    del future2
    gc.collect()
    assert not os.path.exists(data_dir_path)

    # Shelve the input data a second time
    future3 = shelf.put(data)
    data_dir_path = os.path.join(shelf.store_backend.location,
                                 future3._data_id)

    # Deleting the shelf should not remove the data on disk because the future
    # still exists and have a reference on the shelf
    del shelf
    gc.collect()
    assert os.path.exists(shelf_path)
    assert os.path.exists(data_dir_path)

    # Deleting the future now should remove the data on disk
    del future3
    gc.collect()
    print(shelf_path)
    assert not os.path.exists(shelf_path)
    assert not os.path.exists(data_dir_path)


@with_numpy
def test_shelved_array_directory(tmpdir):
    data = np.random.random((5, 5))
    shelf = JoblibShelf(location=tmpdir.strpath, verbose=10)
    shelf_path = shelf.store_backend.location
    future1 = shelf.put(data)
    data_dir_path = os.path.join(shelf.store_backend.location,
                                 future1._data_id)

    # Check pickle file and parent directory exist (with the right name)
    assert os.path.basename(data_dir_path) == future1._data_id
    assert os.path.exists(data_dir_path)
    assert os.path.exists(os.path.join(data_dir_path, 'output.pkl'))

    # Check input array is correctly reloaded
    np.testing.assert_array_equal(future1.result(), data)

    # Add a second reference of the first future
    future2 = future1

    # Deleting future1 should not remove the cached directory
    del future1
    gc.collect()
    assert os.path.exists(data_dir_path)

    # Deleting future2 should remove the cached directory since it's the last
    # reference on the initial future
    del future2
    gc.collect()
    assert not os.path.exists(data_dir_path)

    # Shelve the input data a second time
    future3 = shelf.put(data)
    data_dir_path = os.path.join(shelf.store_backend.location,
                                 future3._data_id)

    # Deleting the shelf should not remove the data on disk because the future
    # still exists and have a reference on the shelf
    del shelf
    gc.collect()
    assert os.path.exists(shelf_path)
    assert os.path.exists(data_dir_path)

    # Deleting the future now should remove the data on disk
    del future3
    gc.collect()
    print(shelf_path)
    assert not os.path.exists(shelf_path)
    assert not os.path.exists(data_dir_path)
