"""
The persistence model for a joblib cache directory.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import sqlite3
import time

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
        keys = self._keys = [k for k, v in self.entries]
        CREATE = "CREATE TABLE IF NOT EXISTS %s (%s)" % (
                    self.tablename,
                    ", ".join("%s %s" % (k, v) 
                              for k, v in self.entries),
                   )
        # Store the filename to enable pickling
        self._filename = filename
        self.conn = sqlite3.connect(filename)
        self.conn.text_factory = str
        self.conn.execute(CREATE)
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

    def _create_index_entry(self):
        # Create a key to store the global info
        try:
            self.update_entry('__INDEX__', access_time=time.time())
        except KeyError:
            self.conn.commit()
            self.new_entry(dict(
                    key='__INDEX__', 
                    func_name='', 
                    module='', 
                    args='', 
                    argument_hash='',
                    creation_time=time.time(),
                    access_time=0,
                    computation_time=0,
                    size=0,
                    last_cost=0,
                ))


    def get(self, key):
        item = self.conn.execute(self._GET_ITEM, (key,)).fetchone()
        if item is None:
            raise KeyError(key)
        return dict(zip(self._keys, item))


    def new_entry(self, entry):
        ADD_ITEM = 'REPLACE INTO %s (%s) VALUES (%s)' % (
                        self.tablename,
                        ', '.join(self._keys),
                        ', '.join(['?']*len(self._keys)),
                      )
        self.conn.execute(ADD_ITEM, [entry[k] for k in self._keys])
        self.conn.commit()
        

    def update_entry(self, key, **values):
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
        HAS_ITEM = 'SELECT 1 FROM %s WHERE key = ?' % self.tablename
        return self.conn.execute(HAS_ITEM, (key,)).fetchone() is not None

    def remove(self, key):
        if not key in self:
            raise KeyError(key)
        DEL_ITEM = 'DELETE FROM %s WHERE key = ?' % self.tablename
        self.conn.execute(DEL_ITEM, (key,))
        self.conn.commit()


    def clear(self):        
        CLEAR_ALL = 'DELETE FROM %s; VACUUM;' % self.tablename
        self.conn.executescript(CLEAR_ALL)
        self.conn.commit()
        self._create_index_entry()


    def sync(self):
        if self.conn is not None:    
            self.conn.commit()


    def close(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None


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

