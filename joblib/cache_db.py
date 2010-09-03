"""
The persistence model for a joblib cache directory.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import sqlite3
import time
from .my_exceptions import JoblibException

################################################################################
# The db class
class CacheDB(object):

    tablename = 'jobcache'

    entries =  (('key', 'TEXT PRIMARY KEY'), 
                ('func_name', 'TEXT NOT NULL'), 
                ('module', 'TEXT NOT NULL'), 
                ('args', 'TEXT'), 
                ('argument_hash', 'TEXT NOT NULL'),
                ('creation_time', 'FLOAT NOT NULL'), 
                ('access_time', 'FLOAT NOT NULL'),
                ('computation_time', 'FLOAT NOT NULL'),
                ('size', 'INTEGER NOT NULL'),
                ('last_cost', 'FLOAT NOT NULL'),
               )

    def __init__(self, filename=':memory:'):
        # Store the filename to enable pickling
        self._filename = filename
        self.open()


    def _create_index_entry(self):
        self._raise_if_closed()
        # Create a key to store the global info
        create = not ('__INDEX__' in self)
        try:
            self.update_entry('__INDEX__', access_time=time.time())
        except KeyError:
            create = True

        if create:
            self.conn.commit()
            self.new_entry(dict(
                    key='__INDEX__', 
                    func_name='', 
                    module='', 
                    args='', 
                    argument_hash='None',
                    creation_time=time.time(),
                    access_time=0,
                    computation_time=0,
                    # Start with a guesstimate of the size used by the db
                    # and the directories.
                    size=-19,
                    last_cost=0,
                ))


    def get(self, key):
        self._raise_if_closed()
        item = self.conn.execute(self._GET_ITEM, (key,)).fetchone()
        if item is None:
            raise KeyError(key)
        return dict(zip(self._keys, item))


    def new_entry(self, entry):
        self._raise_if_closed()
        ADD_ITEM = 'REPLACE INTO %s (%s) VALUES (%s)' % (
                        self.tablename,
                        ', '.join(self._keys),
                        ', '.join(['?']*len(self._keys)),
                      )
        self.conn.execute(ADD_ITEM, [entry[k] for k in self._keys])
        self.conn.commit()
        

    def update_entry(self, key, **values):
        self._raise_if_closed()
        # 2/3 faster without the check: too bad for useful error messages
        #if not key in self:
        #    raise KeyError(key)
        UPDATE_ITEM = "UPDATE %s SET %s WHERE key = ?" % (
                        self.tablename,
                        ','.join("%s=%s" % (k, repr(v)) 
                                 for k, v in values.items()),
                      )
        self.conn.execute(UPDATE_ITEM, (key, ))
        self.conn.commit()
 

    def __contains__(self, key):
        self._raise_if_closed()
        HAS_ITEM = 'SELECT 1 FROM %s WHERE key = ?' % self.tablename
        return self.conn.execute(HAS_ITEM, (key,)).fetchone() is not None

    def remove(self, key):
        self._raise_if_closed()
        # Don't test to see if the key is in the DB: less access means
        # more robustness to races, and better ask for forgivness anyhow
        #if not key in self:
        #    raise KeyError(key)
        DEL_ITEM = 'DELETE FROM %s WHERE key = ?' % self.tablename
        self.conn.execute(DEL_ITEM, (key,))
        self.conn.commit()


    def clear(self):
        self._raise_if_closed()
        CLEAR_ALL = 'DELETE FROM %s; VACUUM;' % self.tablename
        self.conn.executescript(CLEAR_ALL)
        self.conn.commit()
        self._create_index_entry()


    def sync(self):
        self._raise_if_closed()
        self.conn.commit()

    def open(self):
        keys = self._keys = [k for k, v in self.entries]
        CREATE = "CREATE TABLE IF NOT EXISTS %s (%s)" % (
                    self.tablename,
                    ", ".join("%s %s" % (k, v) 
                              for k, v in self.entries),
                   )
        self.conn = sqlite3.connect(self._filename)
        self.conn.text_factory = str
        self.conn.execute(CREATE)
        # A few tweaks for faster write speed, and less safety
        # XXX: They should be optional with a 'safe' mode.
        self.conn.execute('PRAGMA temp_store=MEMORY')
        self.conn.execute('PRAGMA synchronous=OFF')
        # These ones don't seem to make much of a difference
        self.conn.execute('PRAGMA cache_size=1048576')
        self.conn.execute('PRAGMA count_changes=OFF')
        self.conn.commit()
        # We control our commit strategy ourselves, for speed.
        self.conn.isolation_level = None
        # precompute a few string, for speed
        self._GET_ITEM = 'SELECT %s FROM %s WHERE key = ?' % (
                        ', '.join(keys),
                        self.tablename,
                        )
        self._GET_ALL_ITEMS = 'SELECT %s FROM %s ORDER BY size' % (
                        ', '.join(keys),
                        self.tablename,
                        )
        self._create_index_entry()


    def close(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None


    def _raise_if_closed(self):
        if self.conn is None:
            text = ('Cache database is closed; call the open() method to '
                    're-establish a connection')
            raise JoblibException(text)


    def __del__(self):
        self.close()


    def __reduce__(self):
        """ Used by the pickler to reconstruct without trying to pickle
            the connection object.
        """
        return (self.__class__, (self._filename, ))

    def __iter__(self):
        cursor = self.conn.cursor()
        cursor.execute(self._GET_ALL_ITEMS)
        keys = self._keys
        return (dict(zip(keys, items)) for items in cursor)

