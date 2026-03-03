import atexit
import os
import shutil
from uuid import uuid4

from .memory import _store_backend_factory

_shelf = None
_futures = dict()
_resource_tracker = None


def _get_resource_tracker():
    global _resource_tracker
    if _resource_tracker is None:
        from .externals.loky.backend.resource_tracker import _resource_tracker
    return _resource_tracker


class ShelfFuture(object):
    """A Future-like object referencing a shelved object."""

    def __init__(self, location, id):
        self.location = location
        self.id = id

    def result(self):
        """Loads data from the shelf storage and returns it."""
        global _futures
        id = (self.location, self.id)
        value = _futures.get(id, None)
        if value is None:
            value = _store_backend_factory("local", self.location).load_item((self.id,))
            _futures[id] = value
        return value

    def clear(self):
        """Clear the referred data from the shelf."""
        _store_backend_factory("local", self.location).clear_item((self.id,))


class Shelf(object):
    """An object for storing values to be used later

    Attributes
    ----------
    location: str
        The location of the shelf folder.
    """

    def __init__(self, location, /, backend_options=None):
        if backend_options is None:
            backend_options = {}
        self.store_backend = _store_backend_factory(
            "local",
            location,
            verbose=1,
            backend_options=dict(**backend_options),
        )
        if self.store_backend is not None:
            _get_resource_tracker().register(self.store_backend.location, "folder")
            atexit.register(self.close)

    def shelve(self, data):
        """Add data to the Shelf and returns a future on the shelved data."""
        if self.store_backend is None:
            raise RuntimeError(
                "You may be trying to shelve using an already closed shelf."
            )
        id = uuid4().hex
        self.store_backend.dump_item((id,), data)
        return ShelfFuture(self.store_backend.location, id)

    def clear(self):
        """Clear all data added to this shelf.
        New data can still be added to the shelf."""
        if self.store_backend is not None:
            self.store_backend.clear()

    def close(self):
        """Close the shelf by removing the storage directory.
        It is no longer possible to add data to the shelf."""
        if self.store_backend is not None:
            shutil.rmtree(self.store_backend.location)
            _get_resource_tracker().unregister(self.store_backend.location, "folder")
            self.store_backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __repr__(self):
        return "{class_name}(location={location})".format(
            class_name=self.__class__.__name__,
            location=(
                None if self.store_backend is None else self.store_backend.location
            ),
        )


def shelve(data):
    """Shelves data and returns a future on it.

    The data can then be deleted at any time by the script to save memory.
    The future, a light-weight object, can be used later to reload the
    initial data.

    The data is kept in a store (either in shared memory or on a disk).
    To retrieve the original data later, use the ``result`` method of
    the returned future, this call will reload the initial data from the disk
    and return it.

    Parameters
    ----------
    data: any
        The data to shelve.

    Returns
    -------
        A future referencing the shelved data
    """
    global _shelf
    if _shelf is None:
        from ._memmapping_reducer import _get_temp_dir

        location = _get_temp_dir(f"joblib_shelf_{os.getpid()}")[0]
        _shelf = Shelf(location)
    return _shelf.shelve(data)


def clear_shelf():
    """Clears all data previously shelved.
    All future referencing data previously shelved are now invalid."""
    if _shelf is not None:
        _shelf.clear()
