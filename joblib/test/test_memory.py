"""
Test the memory module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import shutil
import os
import os.path
import pickle
import sys
import time
import datetime

from joblib.memory import Memory, MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult, _FUNCTION_HASHES
from joblib.memory import _get_cache_items, _get_cache_items_to_delete
from joblib.memory import _load_output, _get_func_fullname
from joblib.memory import JobLibCollisionWarning
from joblib.test.common import with_numpy, np
from joblib.testing import raises
from joblib._compat import PY3_OR_LATER


###############################################################################
# Module-level variables for the tests
def f(x, y=1):
    """ A module-level function for testing purposes.
    """
    return x ** 2 + y


###############################################################################
# Helper function for the tests
def check_identity_lazy(func, accumulator, cachedir):
    """ Given a function and an accumulator (a list that grows every
        time the function is called), check that the function can be
        decorated by memory to be a lazy identity.
    """
    # Call each function with several arguments, and check that it is
    # evaluated only once per argument.
    memory = Memory(cachedir=cachedir, verbose=0)
    memory.clear(warn=False)
    func = memory.cache(func)
    for i in range(3):
        for _ in range(2):
            assert func(i) == i
            assert len(accumulator) == i + 1


###############################################################################
# Tests
def test_memory_integration(tmpdir):
    """ Simple test of memory lazy evaluation.
    """
    accumulator = list()
    # Rmk: this function has the same name than a module-level function,
    # thus it serves as a test to see that both are identified
    # as different.

    def f(l):
        accumulator.append(1)
        return l

    check_identity_lazy(f, accumulator, tmpdir.strpath)

    # Now test clearing
    for compress in (False, True):
        for mmap_mode in ('r', None):
            memory = Memory(cachedir=tmpdir.strpath, verbose=10,
                            mmap_mode=mmap_mode, compress=compress)
            # First clear the cache directory, to check that our code can
            # handle that
            # NOTE: this line would raise an exception, as the database file is
            # still open; we ignore the error since we want to test what
            # happens if the directory disappears
            shutil.rmtree(tmpdir.strpath, ignore_errors=True)
            g = memory.cache(f)
            g(1)
            g.clear(warn=False)
            current_accumulator = len(accumulator)
            out = g(1)

        assert len(accumulator) == current_accumulator + 1
        # Also, check that Memory.eval works similarly
        assert memory.eval(f, 1) == out
        assert len(accumulator) == current_accumulator + 1

    # Now do a smoke test with a function defined in __main__, as the name
    # mangling rules are more complex
    f.__module__ = '__main__'
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    memory.cache(f)(1)


def test_no_memory():
    """ Test memory with cachedir=None: no memoize """
    accumulator = list()

    def ff(l):
        accumulator.append(1)
        return l

    mem = Memory(cachedir=None, verbose=0)
    gg = mem.cache(ff)
    for _ in range(4):
        current_accumulator = len(accumulator)
        gg(1)
        assert len(accumulator) == current_accumulator + 1


def test_memory_kwarg(tmpdir):
    " Test memory with a function with keyword arguments."
    accumulator = list()

    def g(l=None, m=1):
        accumulator.append(1)
        return l

    check_identity_lazy(g, accumulator, tmpdir.strpath)

    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    g = memory.cache(g)
    # Smoke test with an explicit keyword argument:
    assert g(l=30, m=2) == 30


def test_memory_lambda(tmpdir):
    " Test memory with a function with a lambda."
    accumulator = list()

    def helper(x):
        """ A helper function to define l as a lambda.
        """
        accumulator.append(1)
        return x

    l = lambda x: helper(x)

    check_identity_lazy(l, accumulator, tmpdir.strpath)


def test_memory_name_collision(tmpdir, recwarn):
    " Check that name collisions with functions will raise warnings"
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)

    @memory.cache
    def name_collision(x):
        """ A first function called name_collision
        """
        return x

    a = name_collision

    @memory.cache
    def name_collision(x):
        """ A second function called name_collision
        """
        return x

    b = name_collision

    a(1)
    b(1)
    assert len(recwarn) == 1
    w = recwarn.pop(JobLibCollisionWarning)
    assert "collision" in str(w.message)


def test_memory_warning_lambda_collisions(tmpdir, recwarn):
    # Check that multiple use of lambda will raise collisions
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    # For isolation with other tests
    memory.clear()
    a = lambda x: x
    a = memory.cache(a)
    b = lambda x: x + 1
    b = memory.cache(b)


    assert a(0) == 0
    assert b(1) == 2
    assert a(1) == 1

    # In recent Python versions, we can retrieve the code of lambdas,
    # thus nothing is raised
    assert len(recwarn) == 4


def test_memory_warning_collision_detection(tmpdir, recwarn):
    # Check that collisions impossible to detect will raise appropriate
    # warnings.
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    # For isolation with other tests
    memory.clear()
    a1 = eval('lambda x: x')
    a1 = memory.cache(a1)
    b1 = eval('lambda x: x+1')
    b1 = memory.cache(b1)

    a1(1)
    b1(1)
    a1(0)

    assert len(recwarn) == 2
    w = recwarn.pop(JobLibCollisionWarning)
    assert "cannot detect" in str(w.message).lower()


def test_memory_partial(tmpdir):
    " Test memory with functools.partial."
    accumulator = list()

    def func(x, y):
        """ A helper function to define l as a lambda.
        """
        accumulator.append(1)
        return y

    import functools
    function = functools.partial(func, 1)

    check_identity_lazy(function, accumulator, tmpdir.strpath)


def test_memory_eval(tmpdir):
    " Smoke test memory with a function with a function defined in an eval."
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)

    m = eval('lambda x: x')
    mm = memory.cache(m)

    assert mm(1) == 1


def count_and_append(x=[]):
    """ A function with a side effect in its arguments.

        Return the lenght of its argument and append one element.
    """
    len_x = len(x)
    x.append(None)
    return len_x


def test_argument_change(tmpdir):
    """ Check that if a function has a side effect in its arguments, it
        should use the hash of changing arguments.
    """
    mem = Memory(cachedir=tmpdir.strpath, verbose=0)
    func = mem.cache(count_and_append)
    # call the function for the first time, is should cache it with
    # argument x=[]
    assert func() == 0
    # the second time the argument is x=[None], which is not cached
    # yet, so the functions should be called a second time
    assert func() == 1


@with_numpy
def test_memory_numpy(tmpdir):
    " Test memory with a function with numpy arrays."
    # Check with memmapping and without.
    for mmap_mode in (None, 'r'):
        accumulator = list()

        def n(l=None):
            accumulator.append(1)
            return l

        memory = Memory(cachedir=tmpdir.strpath, mmap_mode=mmap_mode,
                            verbose=0)
        memory.clear(warn=False)
        cached_n = memory.cache(n)

        rnd = np.random.RandomState(0)
        for i in range(3):
            a = rnd.random_sample((10, 10))
            for _ in range(3):
                assert np.all(cached_n(a) == a)
                assert len(accumulator) == i + 1


@with_numpy
def test_memory_numpy_check_mmap_mode(tmpdir):
    """Check that mmap_mode is respected even at the first call"""

    memory = Memory(cachedir=tmpdir.strpath, mmap_mode='r', verbose=0)
    memory.clear(warn=False)

    @memory.cache()
    def twice(a):
        return a * 2

    a = np.ones(3)

    b = twice(a)
    c = twice(a)

    assert isinstance(c, np.memmap)
    assert c.mode == 'r'

    assert isinstance(b, np.memmap)
    assert b.mode == 'r'


def test_memory_exception(tmpdir):
    """ Smoketest the exception handling of Memory.
    """
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)

    class MyException(Exception):
        pass

    @memory.cache
    def h(exc=0):
        if exc:
            raise MyException

    # Call once, to initialise the cache
    h()

    for _ in range(3):
        # Call 3 times, to be sure that the Exception is always raised
        with raises(MyException):
            h(1)


def test_memory_ignore(tmpdir):
    " Test the ignore feature of memory "
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    accumulator = list()

    @memory.cache(ignore=['y'])
    def z(x, y=1):
        accumulator.append(1)

    assert z.ignore == ['y']

    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=1)
    assert len(accumulator) == 1
    z(0, y=2)
    assert len(accumulator) == 1


def test_partial_decoration(tmpdir):
    "Check cache may be called with kwargs before decorating"
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)

    test_values = [
        (['x'], 100, 'r'),
        ([], 10, None),
    ]
    for ignore, verbose, mmap_mode in test_values:
        @memory.cache(ignore=ignore, verbose=verbose, mmap_mode=mmap_mode)
        def z(x):
            pass

        assert z.ignore == ignore
        assert z._verbose == verbose
        assert z.mmap_mode == mmap_mode


def test_func_dir(tmpdir):
    # Test the creation of the memory cache directory for the function.
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    memory.clear()
    path = __name__.split('.')
    path.append('f')
    path = tmpdir.join('joblib', *path).strpath

    g = memory.cache(f)
    # Test that the function directory is created on demand
    assert g._get_func_dir() == path
    assert os.path.exists(path)

    # Test that the code is stored.
    # For the following test to be robust to previous execution, we clear
    # the in-memory store
    _FUNCTION_HASHES.clear()
    assert not g._check_previous_func_code()
    assert os.path.exists(os.path.join(path, 'func_code.py'))
    assert g._check_previous_func_code()

    # Test the robustness to failure of loading previous results.
    dir, _ = g.get_output_dir(1)
    a = g(1)
    assert os.path.exists(dir)
    os.remove(os.path.join(dir, 'output.pkl'))
    assert a == g(1)


def test_persistence(tmpdir):
    # Test the memorized functions can be pickled and restored.
    memory = Memory(cachedir=tmpdir.strpath, verbose=0)
    g = memory.cache(f)
    output = g(1)

    h = pickle.loads(pickle.dumps(g))

    output_dir, _ = h.get_output_dir(1)
    func_name = _get_func_fullname(f)
    assert output == _load_output(output_dir, func_name)
    memory2 = pickle.loads(pickle.dumps(memory))
    assert memory.cachedir == memory2.cachedir

    # Smoke test that pickling a memory with cachedir=None works
    memory = Memory(cachedir=None, verbose=0)
    pickle.loads(pickle.dumps(memory))
    g = memory.cache(f)
    gp = pickle.loads(pickle.dumps(g))
    gp(1)


def test_call_and_shelve(tmpdir):
    """Test MemorizedFunc outputting a reference to cache.
    """

    for func, Result in zip((MemorizedFunc(f, tmpdir.strpath),
                             NotMemorizedFunc(f),
                             Memory(cachedir=tmpdir.strpath).cache(f),
                             Memory(cachedir=None).cache(f),
                             ),
                            (MemorizedResult, NotMemorizedResult,
                             MemorizedResult, NotMemorizedResult)):
        assert func(2) == 5
        result = func.call_and_shelve(2)
        assert isinstance(result, Result)
        assert result.get() == 5

        result.clear()
        with raises(KeyError):
            result.get()
        result.clear()  # Do nothing if there is no cache.


def test_memorized_pickling(tmpdir):
    for func in (MemorizedFunc(f, tmpdir.strpath), NotMemorizedFunc(f)):
        filename = tmpdir.join('pickling_test.dat').strpath
        result = func.call_and_shelve(2)
        with open(filename, 'wb') as fp:
            pickle.dump(result, fp)
        with open(filename, 'rb') as fp:
            result2 = pickle.load(fp)
        assert result2.get() == result.get()
        os.remove(filename)


def test_memorized_repr(tmpdir):
    func = MemorizedFunc(f, tmpdir.strpath)
    result = func.call_and_shelve(2)

    func2 = MemorizedFunc(f, tmpdir.strpath)
    result2 = func2.call_and_shelve(2)
    assert result.get() == result2.get()
    assert repr(func) == repr(func2)

    # Smoke test with NotMemorizedFunc
    func = NotMemorizedFunc(f)
    repr(func)
    repr(func.call_and_shelve(2))

    # Smoke test for message output (increase code coverage)
    func = MemorizedFunc(f, tmpdir.strpath, verbose=11, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=11)
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=5, timestamp=time.time())
    result = func.call_and_shelve(11)
    result.get()

    func = MemorizedFunc(f, tmpdir.strpath, verbose=5)
    result = func.call_and_shelve(11)
    result.get()


def test_memory_file_modification(capsys, tmpdir):
    # Test that modifying a Python file after loading it does not lead to
    # Recomputation
    dir_name = tmpdir.mkdir('tmp_import').strpath
    filename = os.path.join(dir_name, 'tmp_joblib_.py')
    content = 'def f(x):\n    print(x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)

    # Load the module:
    sys.path.append(dir_name)
    import tmp_joblib_ as tmp

    mem = Memory(cachedir=tmpdir.strpath, verbose=0)
    f = mem.cache(tmp.f)
    # First call f a few times
    f(1)
    f(2)
    f(1)

    # Now modify the module where f is stored without modifying f
    with open(filename, 'w') as module_file:
        module_file.write('\n\n' + content)

    # And call f a couple more times
    f(1)
    f(1)

    # Flush the .pyc files
    shutil.rmtree(dir_name)
    os.mkdir(dir_name)
    # Now modify the module where f is stored, modifying f
    content = 'def f(x):\n    print("x=%s" % x)\n    return x\n'
    with open(filename, 'w') as module_file:
        module_file.write(content)

    # And call f more times prior to reloading: the cache should not be
    # invalidated at this point as the active function definition has not
    # changed in memory yet.
    f(1)
    f(1)

    # Now reload
    sys.stdout.write('Reloading\n')
    sys.modules.pop('tmp_joblib_')
    import tmp_joblib_ as tmp
    f = mem.cache(tmp.f)

    # And call f more times
    f(1)
    f(1)

    out, err = capsys.readouterr()
    assert out == '1\n2\nReloading\nx=1\n'


def _function_to_cache(a, b):
    # Just a place holder function to be mutated by tests
    pass


def _sum(a, b):
    return a + b


def _product(a, b):
    return a * b


def test_memory_in_memory_function_code_change(tmpdir):
    _function_to_cache.__code__ = _sum.__code__

    mem = Memory(cachedir=tmpdir.strpath, verbose=0)
    f = mem.cache(_function_to_cache)

    assert f(1, 2) == 3
    assert f(1, 2) == 3

    # Check that inline function modification triggers a cache invalidation

    _function_to_cache.__code__ = _product.__code__
    assert f(1, 2) == 2
    assert f(1, 2) == 2


def test_clear_memory_with_none_cachedir():
    mem = Memory(cachedir=None)
    mem.clear()

if PY3_OR_LATER:
    exec("""
def func_with_kwonly_args(a, b, *, kw1='kw1', kw2='kw2'):
    return a, b, kw1, kw2

def func_with_signature(a: int, b: float) -> float:
    return a + b
""")

    def test_memory_func_with_kwonly_args(tmpdir):
        mem = Memory(cachedir=tmpdir.strpath, verbose=0)
        func_cached = mem.cache(func_with_kwonly_args)

        assert func_cached(1, 2, kw1=3) == (1, 2, 3, 'kw2')

        # Making sure that providing a keyword-only argument by
        # position raises an exception
        with raises(ValueError) as excinfo:
            func_cached(1, 2, 3, kw2=4)
        excinfo.match("Keyword-only parameter 'kw1' was passed as positional "
                      "parameter")

        # Keyword-only parameter passed by position with cached call
        # should still raise ValueError
        func_cached(1, 2, kw1=3, kw2=4)

        with raises(ValueError) as excinfo:
            func_cached(1, 2, 3, kw2=4)
        excinfo.match("Keyword-only parameter 'kw1' was passed as positional "
                      "parameter")

        # Test 'ignore' parameter
        func_cached = mem.cache(func_with_kwonly_args, ignore=['kw2'])
        assert func_cached(1, 2, kw1=3, kw2=4) == (1, 2, 3, 4)
        assert func_cached(1, 2, kw1=3, kw2='ignored') == (1, 2, 3, 4)

    def test_memory_func_with_signature(tmpdir):
        mem = Memory(cachedir=tmpdir.strpath, verbose=0)
        func_cached = mem.cache(func_with_signature)

        assert func_cached(1, 2.) == 3.


def _setup_temporary_cache_folder(tmpdir, num_inputs=10):
    # Use separate cache dir to avoid side-effects from other tests
    # that do not use _setup_temporary_cache_folder
    mem = Memory(cachedir=tmpdir.mkdir('separate_cache').strpath,
                 verbose=0)

    @mem.cache()
    def get_1000_bytes(arg):
        return 'a' * 1000

    inputs = list(range(num_inputs))
    for arg in inputs:
        get_1000_bytes(arg)

    hash_dirnames = [get_1000_bytes._get_output_dir(arg)[0] for arg in inputs]

    full_hashdirs = [os.path.join(get_1000_bytes.cachedir, dirname)
                     for dirname in hash_dirnames]
    return mem, full_hashdirs, get_1000_bytes


def test__get_cache_items(tmpdir):
    mem, expected_hash_cachedirs, _ = _setup_temporary_cache_folder(tmpdir)
    cachedir = mem.cachedir
    cache_items = _get_cache_items(cachedir)
    hash_cachedirs = [ci.path for ci in cache_items]
    assert set(hash_cachedirs) == set(expected_hash_cachedirs)

    def get_files_size(directory):
        full_paths = [os.path.join(directory, fn)
                      for fn in os.listdir(directory)]
        return sum(os.path.getsize(fp) for fp in full_paths)

    expected_hash_cache_sizes = [get_files_size(hash_dir)
                                 for hash_dir in hash_cachedirs]
    hash_cache_sizes = [ci.size for ci in cache_items]
    assert hash_cache_sizes == expected_hash_cache_sizes

    output_filenames = [os.path.join(hash_dir, 'output.pkl')
                        for hash_dir in hash_cachedirs]

    expected_last_accesses = [
        datetime.datetime.fromtimestamp(os.path.getatime(fn))
        for fn in output_filenames]
    last_accesses = [ci.last_access for ci in cache_items]
    assert last_accesses == expected_last_accesses


def test__get_cache_items_to_delete(tmpdir):
    mem, expected_hash_cachedirs, _ = _setup_temporary_cache_folder(tmpdir)
    cachedir = mem.cachedir
    cache_items = _get_cache_items(cachedir)
    # bytes_limit set to keep only one cache item (each hash cache
    # folder is about 1000 bytes + metadata)
    cache_items_to_delete = _get_cache_items_to_delete(cachedir, '2K')
    nb_hashes = len(expected_hash_cachedirs)
    assert set.issubset(set(cache_items_to_delete), set(cache_items))
    assert len(cache_items_to_delete) == nb_hashes - 1

    # Sanity check bytes_limit=2048 is the same as bytes_limit='2K'
    cache_items_to_delete_2048b = _get_cache_items_to_delete(cachedir, 2048)
    assert sorted(cache_items_to_delete) == sorted(cache_items_to_delete_2048b)

    # bytes_limit greater than the size of the cache
    cache_items_to_delete_empty = _get_cache_items_to_delete(cachedir, '1M')
    assert cache_items_to_delete_empty == []

    # All the cache items need to be deleted
    bytes_limit_too_small = 500
    cache_items_to_delete_500b = _get_cache_items_to_delete(
        cachedir, bytes_limit_too_small)
    assert set(cache_items_to_delete_500b), set(cache_items)

    # Test LRU property: surviving cache items should all have a more
    # recent last_access that the ones that have been deleted
    cache_items_to_delete_6000b = _get_cache_items_to_delete(cachedir, 6000)
    surviving_cache_items = set(cache_items).difference(
        cache_items_to_delete_6000b)

    assert (max(ci.last_access for ci in cache_items_to_delete_6000b) <=
            min(ci.last_access for ci in surviving_cache_items))


def test_memory_reduce_size(tmpdir):
    mem, _, _ = _setup_temporary_cache_folder(tmpdir)
    cachedir = mem.cachedir
    ref_cache_items = _get_cache_items(cachedir)

    # By default mem.bytes_limit is None and reduce_size is a noop
    mem.reduce_size()
    cache_items = _get_cache_items(cachedir)
    assert sorted(ref_cache_items) == sorted(cache_items)

    # No cache items deleted if bytes_limit greater than the size of
    # the cache
    mem.bytes_limit = '1M'
    mem.reduce_size()
    cache_items = _get_cache_items(cachedir)
    assert sorted(ref_cache_items) == sorted(cache_items)

    # bytes_limit is set so that only two cache items are kept
    mem.bytes_limit = '3K'
    mem.reduce_size()
    cache_items = _get_cache_items(cachedir)
    assert set.issubset(set(cache_items), set(ref_cache_items))
    assert len(cache_items) == 2

    # bytes_limit set so that no cache item is kept
    bytes_limit_too_small = 500
    mem.bytes_limit = bytes_limit_too_small
    mem.reduce_size()
    cache_items = _get_cache_items(cachedir)
    assert cache_items == []
