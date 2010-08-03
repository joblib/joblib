"""
The persistence model for a joblib cache directory.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.

import sqlite3


################################################################################
# The db class
class CacheDB(object):

    tablename = 'jobcache'

    entries =  (('key', 'TEXT PRIMARY KEY'), 
                ('func_name', 'TEXT NOT NULL'), 
                ('module', 'TEXT NOT NULL'), 
                ('args', 'TEXT NOT NULL'), 
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
        # precompute a few string, for speed
        self._GET_ITEM = 'SELECT %s FROM %s WHERE key = ?' % (
                        ', '.join(keys),
                        self.tablename,
                        )


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
        UPDATE_ITEM = 'UPDATE %s SET %s WHERE key=%s' % (
                        self.tablename,
                        ','.join("%s=%s" % (k, v) for k, v in values.items()),
                        key,
                      )
        self.conn.execute(UPDATE_ITEM, )
        self.conn.commit()
 

    def remove(self, key):
        HAS_ITEM = 'SELECT 1 FROM %s WHERE key = ?' % self.tablename
        if self.conn.execute(HAS_ITEM, (key,)).fetchone() is None:
            raise KeyError(key)
        DEL_ITEM = 'DELETE FROM %s WHERE key = ?' % self.tablename
        self.conn.execute(DEL_ITEM, (key,))
        self.conn.commit()


    def clear(self):        
        CLEAR_ALL = 'DELETE FROM %s; VACUUM;' % self.tablename
        self.conn.executescript(CLEAR_ALL)
        self.conn.commit()


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

# XXX: ORDERBY
