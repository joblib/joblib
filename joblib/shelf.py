"""
A context object for temporary shelving arbitrary objects in a data store.
"""

# Copyright (c) 2018 Inria
# License: BSD Style, 3 clauses.

import os
import tempfile
import weakref
import shutil

from uuid import uuid4

from .memory import StoreBase

_active_shelf = None
_active_shelf_mmap = None
_mmap_weak_dict = weakref.WeakKeyDictionary()


class JoblibShelfFuture():
    """A Future-like object for keeping a reference on shelved objects."""

    def __init__(self, data_id, shelf):
        self._data_id = data_id
        self._shelf = shelf

    def result(self):
        """Reloads the data from store and returns it.

        Returns
        -------
            The reloaded data.
        """
        return self._shelf.store_backend.load_item([self._data_id])


###############################################################################
# class `JoblibShelf`
###############################################################################
class JoblibShelf(StoreBase):
    """A context object for shelving an arbitrary object in a store."""

    # The root of the shelf storage is built with current Python interpreter
    # process ID
    _root = 'joblib-shelf-{}'.format(os.getpid())
    _futures = set()

    def put(self, data):
        """Put data on the shelf.

        Parameters
        ----------
        data: any
            The data to put on the shelf.

        Returns:
        --------
            A future on the shelved data.

        """
        data_id = uuid4().hex
        self.store_backend.dump_item([data_id], data)

        future = JoblibShelfFuture(data_id, self)

        def _collect(ref):
            self._on_collect(data_id)
        self._futures.add(weakref.ref(future, _collect))
        return future

    def _on_collect(self, data_id):
        self.store_backend.clear_location(
            os.path.join(self.store_backend.location, data_id))

    def __del__(self):
        # TODO: FIX self.clear()
        shutil.rmtree(self.store_backend.location)


def shelve(input_object):
    """Shelves an arbitrary object and returns a future on it.

    The input object can then be deleted at any time by the script to save
    memory. The future, a light-weight object, can be used later to reload the
    initial object.

    The input object is kept in a store (by default a file on a disk) as long
    as the future object exists (technically: as long as there is a reference
    on the future).
    To retrieve the original input object later, use the ``result`` method of
    the returned future, this call will reload the initial data from the disk
    and return it.

    Parameters
    ----------
    input_object: any
        The input object to shelve.

    Returns
    -------
        A future referencing the shelved object

    See Also
    --------
        joblib.shelve_mmap : function to shelve a numpy array. The future
        result returns a memmapped numpy array.

    Notes
    -----
        The content of the shelved object is effectively deleted from the store
        only when no reference on its future no longer exists.
        When the interpreter process exits, all shelved objects are deleted
        from the store.

    Examples
    --------

        A simple example:

        >>> from joblib import shelve
        >>> input_object = "The object on the shelf"
        >>> future = shelve(input_object)
        >>> del input_object  # input object can now be removed from memory
        >>> future  #doctest: +SKIP
        <joblib.shelf.JoblibShelfFuture at 0x7fc15fad7a90>
        >>> print(future.result())
        The object on the shelf

        Only the 'threading' backend of `Parallel` can be used with `shelve`:

        >>> from joblib import shelve, Parallel, delayed
        >>> def func(future):
        ...     return future.result()**2
        >>> r = Parallel(n_jobs=10, backend='threading')(
        ...              delayed(func)(shelve(i)) for i in range(10))
        >>> r
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

    """
    global _active_shelf
    if _active_shelf is None:
        tmp_folder = os.environ.get('JOBLIB_TEMP_FOLDER',
                                    tempfile.gettempdir())
        _active_shelf = JoblibShelf(tmp_folder)

    return _active_shelf.put(input_object)


def shelve_mmap(input_object):
    """Shelves an arbitrary object and returns a future.

    The input array can then be deleted at any time to save memory. The
    future, a light-weight object, can be used later to reload the initial
    array.

    The future returns a memmap only for input numpy arrays. On other types of
    objects, it returns the initial object.

    During the life of the future the content of the array is kept written on a
    store (by default a file on a disk).
    To retrieve the memmapped input array later, use the ``result`` method of
    the future, this call will return a numpy memmap on the input array.

    Parameters
    ----------
    input_array: any
        The input numpy array to shelve.

    Returns
    -------
        A future on the shelved array.

    See Also
    --------
        joblib.shelve : function to shelve any kind of object. The future
        result returns the original input object.

    Notes
    -----
        The content of the shelved array is effectively deleted from the store
        only when no reference on its future no longer exists.
        When the interpreter process exits, all shelved objects are deleted
        from the store.

    Examples
    --------

        A simple example:

        >>> import numpy as np
        >>> from joblib import shelve_mmap
        >>> array = np.ones((10, 10))
        >>> mmap = shelve_mmap(array)
        >>> del array  # input array can now be removed from memory
        >>> mmap
        memmap([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

        Only the 'threading' backend of `Parallel` can be used with
        `shelve_mmap`:

        >>> import numpy as np
        >>> from joblib import shelve_mmap, Parallel, delayed
        >>> def f(data):
        ...     return np.mean(data)
        >>> array = np.random.random((10, 10))
        >>> Parallel(n_jobs=10, backend='threading')(
        ...          delayed(f)(i) for i in shelve_mmap(array))  #doctest: +SKIP
        [0.5224197461540009,
         0.5529565351274045,
         0.5685303248444292,
         0.47138560341416086,
         0.5606711476847496,
         0.4467337319232188,
         0.37955295459961974,
         0.39589095750523356,
         0.46805210613397985,
         0.4432485635619462]

    """
    global _active_shelf_mmap
    if _active_shelf_mmap is None:
        tmp_folder = os.environ.get('JOBLIB_TEMP_FOLDER',
                                    tempfile.gettempdir())
        _active_shelf_mmap = JoblibShelf(tmp_folder, mmap_mode='r')

    future = _active_shelf_mmap.put(input_object)
    _mmap_weak_dict[future] = future.result()
    return _mmap_weak_dict[future]
