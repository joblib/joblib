"""
Test the cache_db module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose

from ..cache_db import CacheDB


################################################################################
# Tests
def test_store_retrieve():
    """ A simple test for CacheDb
    """
    db = CacheDB()
    e = dict(key='key', func_name='foo', module='bar', 
              args='', creation_time=10, access_time=20, 
              computation_time=5, size=100, last_cost=.5)
    db.new_entry(e)
    d = db.get('key')
    yield nose.tools.assert_equal, d, e
    db.update_entry('key', args='2')
    yield nose.tools.assert_equal, '2', db.get('key')['args']
    db.remove('key')
    yield nose.tools.assert_raises, KeyError, db.get, 'key'
    yield nose.tools.assert_raises, KeyError, db.remove, 'key'
    e.pop('module')
    yield nose.tools.assert_raises, KeyError, db.new_entry, e
    # Smoke test the two remaining functions
    db.sync()
    db.clear()
 

def test_pickle():
    """ Check that cache_db objects do pickle.
    """
    import pickle
    db = CacheDB()
    pickle.dumps(db)


