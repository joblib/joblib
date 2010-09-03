"""
Test the cache_db module.
"""
from __future__ import with_statement

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import nose

from tempfile import mkdtemp

from ..cache_db import CacheDB
from ..memory import Memory
from ..my_exceptions import JoblibException

################################################################################
# Tests
def test_store_retrieve():
    """ A simple test for CacheDb
    """
    db = CacheDB()
    yield nose.tools.assert_true, '__INDEX__' in db
    e = dict(key='key', func_name='foo', module='bar', 
              args='', creation_time=10, access_time=20, 
              argument_hash='foobar',
              computation_time=5, size=100, last_cost=.5)
    db.new_entry(e)
    d = db.get('key')
    yield nose.tools.assert_equal, d, e
    db.update_entry('key', args='2')
    # Check that the update_entry also works for strings
    db.update_entry('key', argument_hash='foo')
    yield nose.tools.assert_equal, '2', db.get('key')['args']
    db.remove('key')
    yield nose.tools.assert_raises, KeyError, db.get, 'key'
    e.pop('module')
    yield nose.tools.assert_raises, KeyError, db.new_entry, e
    # Smoke test the two remaining functions
    db.sync()
    db.clear()
    # Even after clearing, we should have an '__INDEX__'
    yield nose.tools.assert_true, '__INDEX__' in db

    # Test iteration
    c1 = d.copy()
    c1['key'] = 'c1'
    db.new_entry(c1)
    c2 = d.copy()
    c2['key'] = 'c2'
    db.new_entry(c2)
    c3 = d.copy()
    c3['key'] = 'c3'
    db.new_entry(c3)

    entries = list(db)
    # We should have 4 entries, as we should always have the __INDEX__
    # entry.
    yield nose.tools.assert_equal, len(entries), 4


def test_pickle():
    """ Check that cache_db objects do pickle.
    """
    import pickle
    db = CacheDB()
    pickle.dumps(db)


def test_closed_exception():
    """ Check that a exception is raised when a closed database is accessed.
    """
    def f(x):
        return x
    
    cachedir = mkdtemp()
    with Memory(cachedir=cachedir, verbose=0) as mem:
        f = mem.cache(f)
        yield nose.tools.assert_equal, 1, f(1)
    yield nose.tools.assert_raises, JoblibException, f, 1
    # test explicitly re-opening the database
    mem.db.open()
    yield nose.tools.assert_equal, 1, f(1)

