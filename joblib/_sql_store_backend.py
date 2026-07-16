"""Store Backend using SQLite"""

import os
import sqlite3
import time

from ._store_backends import StoreBackendBase
from .logger import format_time


class SQLStoreBackend(StoreBackendBase):
    """TODO"""

    def configure(self, location, verbose=1, backend_options=None):
        """Configures the store.

        Parameters
        ----------
        location: string
            The base location used by the store. On a filesystem, this
            corresponds to a directory.
        verbose: int
            The level of verbosity of the store
        backend_options: dict
            Options to be used. The only valid option is 'compress'.
        """
        if backend_options is None:
            backend_options = {}

        self.location = location
        # self.verbose = verbose
        self.compress = backend_options.get("compress", False)

        os.makedirs(os.path.dirname(self.location), exist_ok=True)
        self.con = sqlite3.connect(self.location)

        with self.con:
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS cache (path TEXT PRIMARY KEY, data BLOB)"
            )

    def load_item(self, call_id, verbose, timestamp, metadata):
        """Load an item from the store.

        Parameters
        ----------
        call_id: list of str
            id of the call
        verbose: int
            The level of verbosity
        timestamp: float
            Time of the creation of the Memory in seconds (used for verbose logs)
        metadata: dict
            Metadata associated to the call.
            If it contains 'input_args', it will be used by the verbose logs.

        Returns
        -------
        The item associated to call_id
        """
        path = os.path.join(*call_id)

        if verbose > 1:
            ts_string = (
                f"{format_time(time.time() - timestamp): <16}"
                if timestamp is not None
                else ""
            )
            signature = os.path.basename(call_id[0])
            if metadata is not None and "input_args" in metadata:
                kwargs = ", ".join(
                    "{}={}".format(*item) for item in metadata["input_args"].items()
                )
                signature += f"({kwargs})"
            msg = f"[Memory]{ts_string}: Loading {signature}"
            if verbose < 10:
                print(f"{msg}...")
            else:
                print(f"{msg} from {self.location}::{path}")

        with self.con.cursor() as cursor:
            cursor.execute("SELECT data FROM cache WHERE path=?", (path,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(
                    "Non-existing item (may have been cleared).\n"
                    f"Path {path} does not exist"
                )

        """
        # file-like object cannot be used when mmap_mode is set
        if mmap_mode is None:
            with self._open_item(filename, "rb") as f:
                item = numpy_pickle.load(f)
        else:
            item = numpy_pickle.load(filename, mmap_mode=mmap_mode)
        return item
        """

    def dump_item(self, call_id, item, verbose):
        """Dump an item in the store.

        Parameters
        ----------
        call_id: list of str
            id of the call
        item: any
            Item to be stored
        verbose: int
            The level of verbosity
        """

    def clear_item(self, call_id):
        """Clear an item from the store.

        Parameters
        ----------
        call_id: list of str
            id to be cleared
        """

    def contains_item(self, call_id):
        """Check if the store contains an item for a given id.

        Parameters
        ----------
        call_id: list of str
            id of the item to be checked
        """

    def get_metadata(self, call_id):
        """Return actual metadata of an item.

        Parameters
        ----------
        call_id: list of str
            id of the call

        Returns
        -------
        metadata: dict
            Metadata associated to the call
        """

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation.

        Parameters
        ----------
        call_id: list of str
            id of the call
        metadata: dict
            Metadata associated to the call
        """

    def get_cached_func_code(self, func_id):
        """Get the code of the cached function.

        Parameters
        ----------
        func_id: list of str
            id of the cached function

        Returns
        -------
        func_code: str
            The code of the cached function
        """

    def store_cached_func_code(self, func_id, func_code):
        """Store the code of the cached function.

        Parameters
        ----------
        func_id: list of str
            id of the cached function
        func_code: str
            The code of the cached function
        """

    def clear_path(self, path_id):
        """Clear all items with a common path in the store.

        Parameters
        ----------
        path_id: list of str
            Prefix id of item to be cleared
        """
