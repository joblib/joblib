# Author: Dag Sverre Selejbotn <d.s.seljebotn@astro.uio.no>
# Copyright (c) 2011 Dag Sverre Seljebotn
# License: BSD Style, 3 clauses.

import tempfile
import shutil
import os
from nose.tools import ok_, eq_, assert_raises
from time import sleep
import functools
from functools import partial

from ..job_store import (COMPUTED, MUST_COMPUTE, WAIT, IllegalOperationError,
                         DirectoryJobStore)

#
# Test fixture
#
tempdir = store_instance = None

def setup_store(**kw):
    global tempdir, store_instance
    tempdir = tempfile.mkdtemp()
    store_instance = DirectoryJobStore(tempdir, save_npy=True, mmap_mode=None, **kw)
                                          
def teardown_store():
    global tempdir, store_instance
    shutil.rmtree(tempdir)
    tempdir = mock_logger = store_instance = None
    
def with_store():
    def with_store_dec(func):
        @functools.wraps(func)
        def inner():
            setup_store()
            try:
                for x in func():
                    yield x
            finally:
                teardown_store()
        return inner
    return with_store_dec

def ls(path):
    return set(os.listdir(path))

#
# Tests
#

def key_function():
    pass


@with_store()
def test_basic():
    job = store_instance.get_job(key_function, dict(a=1))

    #No transactional security, at least yet
    #yield eq_, job.is_computed(), False
    #yield eq_, job.load_or_lock(blocking=False)[0], MUST_COMPUTE
    #job.persist_output((1, 2, 3))
    #job.rollback()

    yield eq_, job.is_computed(), False
    yield eq_, job.load_or_lock(blocking=False)[0], MUST_COMPUTE
    job.persist_output((1, 2, 3))
    job.persist_input(34, {}, {'a': 34})
    job.commit()
    yield eq_, job.load_or_lock(blocking=False), (COMPUTED, (1, 2, 3))
    yield eq_, job.is_computed(), True

    job.close()
    
    job = store_instance.get_job(key_function, dict(a=1))
    yield eq_, job.is_computed(), True
    yield eq_, job.load_or_lock(blocking=False), (COMPUTED, (1, 2, 3))
    job.close()

    # Check contents of dir
    hash = store_instance._hash_input(dict(a=1))
    job_path = os.path.join(tempdir, 'joblib', 'test', 'test_job_store', 'key_function', hash)
    yield ok_, os.path.exists(job_path)
    if os.path.exists(job_path):
        yield eq_, ls(job_path), set(['input_args.json', 'output.pkl'])

@with_store()
def test_errors():
    job = store_instance.get_job(key_function, dict(a=1))
    yield assert_raises, IllegalOperationError, job.persist_output, (1, 2, 3)
    yield assert_raises, IllegalOperationError, job.persist_input, 34, {}, {'a':34}
    job.close()

