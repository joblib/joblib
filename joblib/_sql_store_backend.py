"""Store Backend using SQLite"""

import io
import json
import os
import sqlite3
import time

from . import numpy_pickle
from ._store_backends import StoreBackendBase
from .logger import format_time


class SQLStoreBackend(StoreBackendBase):
    """A StoreBackend a sqlite database."""

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
        self.compress = backend_options.get("compress", False)

        os.makedirs(os.path.dirname(self.location), exist_ok=True)
        self.con = sqlite3.connect(self.location)

        with self.con:
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "path TEXT PRIMARY KEY, "
                "data BLOB,"
                "metadata TEXT)"
            )
            self.con.execute(
                "CREATE TABLE IF NOT EXISTS func_code ("
                "path TEXT PRIMARY KEY, "
                "code TEXT)"
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

        cursor = self.con.cursor()
        cursor.execute("SELECT data FROM cache WHERE path=?", (path,))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            raise KeyError(
                "Non-existing item (may have been cleared).\n"
                f"Path {path} does not exist"
            )
        serialized_data = row[0]

        file = io.BytesIO(serialized_data)
        item = numpy_pickle.load(file)
        return item

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
        path = os.path.join(*call_id)

        if verbose > 10:
            print("Persisting in %s" % path)

        file = io.BytesIO()
        numpy_pickle.dump(item, file, compress=self.compress)
        serialized_data = file.getvalue()

        with self.con:
            self.con.execute(
                "INSERT INTO cache (path, data) "
                "VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET data = excluded.data",
                (path, serialized_data),
            )

    def clear_item(self, call_id):
        """Clear an item from the store.

        Parameters
        ----------
        call_id: list of str
            id to be cleared
        """
        path = os.path.join(*call_id)
        with self.con:
            self.con.execute("DELETE FROM cache WHERE path=?", path)

    def contains_item(self, call_id):
        """Check if the store contains an item for a given id.

        Parameters
        ----------
        call_id: list of str
            id of the item to be checked
        """
        path = os.path.join(*call_id)
        cursor = self.con.cursor()
        cursor.execute("SELECT 1 FROM cache WHERE path=? and data IS NOT NULL", (path,))
        return cursor.fetchone() is not None

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
        path = os.path.join(*call_id)
        cursor = self.con.cursor()
        cursor.execute("SELECT metadata FROM cache WHERE path=?", (path,))
        metadata = cursor.fetchone()
        cursor.close()
        if metadata is None:
            return {}
        return json.loads(metadata[0])

    def store_metadata(self, call_id, metadata):
        """Store metadata of a computation.

        Parameters
        ----------
        call_id: list of str
            id of the call
        metadata: dict
            Metadata associated to the call
        """
        path = os.path.join(*call_id)
        metadata_str = json.dumps(metadata)
        with self.con:
            self.con.execute(
                "INSERT INTO cache (path, metadata) "
                "VALUES (?, ?) "
                "ON CONFLICT(path) DO UPDATE SET metadata = excluded.metadata",
                (path, metadata_str),
            )

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
        path = os.path.join(*func_id)
        cursor = self.con.cursor()
        cursor.execute("SELECT code FROM func_code WHERE path=?", (path,))
        code = cursor.fetchone()
        cursor.close()
        if code is None:
            raise IOError(f"No function with id: {func_id}, in storage {self.location}")
        return code[0]

    def store_cached_func_code(self, func_id, func_code=None):
        """Store the code of the cached function.

        Parameters
        ----------
        func_id: list of str
            id of the cached function
        func_code: str
            The code of the cached function
        """
        if func_code is None:
            return

        path = os.path.join(*func_id)
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO func_code (path, code) VALUES (?, ?)",
                (path, func_code),
            )

    def get_cached_func_info(self, func_id):
        """Return information related to the cached function if it exists.

        Parameters
        ----------
        func_id: list of str
            id of the cached function

        Returns
        -------
        func_info: dict
            Information concerning the function
        """
        path = os.path.join(*func_id)
        return {"location": f"{self.location}::{path}"}

    def clear_path(self, path_id):
        """Clear all items with a common path in the store.

        Parameters
        ----------
        path_id: list of str
            Prefix id of item to be cleared
        """
        if len(path_id) == 0:
            return self.clear()

        path = os.path.join(*path_id)
        with self.con:
            self.con.execute("DELETE FROM cache WHERE path=?", (path,))
            self.con.execute("DELETE FROM func_code WHERE path=?", (path,))

        pattern = os.path.join(path, "")
        pattern = pattern.replace("\\", "\\\\")
        pattern = pattern.replace("_", "\\_")
        pattern = pattern.replace("%", "\\%")
        pattern += "%"
        with self.con:
            self.con.execute(
                "DELETE FROM cache WHERE path LIKE ? ESCAPE '\\'", (pattern,)
            )
            self.con.execute(
                "DELETE FROM func_code WHERE path LIKE ? ESCAPE '\\'", (pattern,)
            )

    def clear(self):
        """Clear the whole store content."""
        with self.con:
            self.con.execute("DELETE FROM cache")
            self.con.execute("DELETE FROM func_code")
            self.con.execute("VACUUM")
