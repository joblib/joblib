"""
Test the func_inspect module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import os
import shutil
import tempfile
import functools

from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import assert_equal, assert_raises_regex, assert_raises
from joblib._compat import PY3_OR_LATER


###############################################################################
# Module-level functions, for tests
def f(x, y=0):
    pass


def g(x, y=1):
    """ A module-level function for testing purposes.
    """
    return x ** 2 + y


def f2(x):
    pass


# Create a Memory object to test decorated functions.
# We should be careful not to call the decorated functions, so that
# cache directories are not created in the temp dir.
temp_folder = tempfile.mkdtemp(prefix="joblib_test_func_inspect_")
mem = Memory(cachedir=temp_folder)


def teardown_module():
    if os.path.exists(temp_folder):
        try:
            shutil.rmtree(temp_folder)
        except Exception as e:
            print("Failed to delete temporary folder %s: %r" %
                  (temp_folder, e))


@mem.cache
def g(x):
    return x


def h(x, y=0, *args, **kwargs):
    pass


def i(x=1):
    pass


def j(x, y, **kwargs):
    pass


def k(*args, **kwargs):
    pass


class Klass(object):

    def f(self, x):
        return x


###############################################################################
# Tests

def test_filter_args():
    yield assert_equal, filter_args(f, [], (1, )), {'x': 1, 'y': 0}
    yield assert_equal, filter_args(f, ['x'], (1, )), {'y': 0}
    yield assert_equal, filter_args(f, ['y'], (0, )), {'x': 0}
    yield assert_equal, filter_args(f, ['y'], (0, ), dict(y=1)), {'x': 0}
    yield assert_equal, filter_args(f, ['x', 'y'], (0, )), {}
    yield assert_equal, filter_args(f, [], (0,), dict(y=1)), {'x': 0, 'y': 1}
    yield assert_equal, filter_args(f, ['y'], (), dict(x=2, y=1)), {'x': 2}

    yield assert_equal, filter_args(i, [], (2, )), {'x': 2}
    yield assert_equal, filter_args(f2, [], (), dict(x=1)), {'x': 1}


def test_filter_args_method():
    obj = Klass()
    assert filter_args(obj.f, [], (1, )) == {'x': 1, 'self': obj}


def test_filter_varargs():
    yield assert_equal, filter_args(h, [], (1, )), \
                            {'x': 1, 'y': 0, '*': [], '**': {}}
    yield assert_equal, filter_args(h, [], (1, 2, 3, 4)), \
                            {'x': 1, 'y': 2, '*': [3, 4], '**': {}}
    yield assert_equal, filter_args(h, [], (1, 25), dict(ee=2)), \
                            {'x': 1, 'y': 25, '*': [], '**': {'ee': 2}}
    yield assert_equal, filter_args(h, ['*'], (1, 2, 25), dict(ee=2)), \
                            {'x': 1, 'y': 2, '**': {'ee': 2}}


def test_filter_kwargs():
    assert (filter_args(k, [], (1, 2), dict(ee=2)) ==
            {'*': [1, 2], '**': {'ee': 2}})
    assert filter_args(k, [], (3, 4)) == {'*': [3, 4], '**': {}}


def test_filter_args_2():
    assert (filter_args(j, [], (1, 2), dict(ee=2)) ==
            {'x': 1, 'y': 2, '**': {'ee': 2}})

    assert_raises(ValueError, filter_args, f, 'a', (None, ))
    # Check that we capture an undefined argument
    assert_raises(ValueError, filter_args, f, ['a'], (None, ))
    ff = functools.partial(f, 1)
    # filter_args has to special-case partial
    assert filter_args(ff, [], (1, )) == {'*': [1], '**': {}}
    assert filter_args(ff, ['y'], (1, )) == {'*': [1], '**': {}}


def test_func_name():
    yield assert_equal, 'f', get_func_name(f)[1]
    # Check that we are not confused by the decoration
    yield assert_equal, 'g', get_func_name(g)[1]


def test_func_inspect_errors():
    # Check that func_inspect is robust and will work on weird objects
    assert get_func_name('a'.lower)[-1] == 'lower'
    assert get_func_code('a'.lower)[1:] == (None, -1)
    ff = lambda x: x
    assert get_func_name(ff, win_characters=False)[-1] == '<lambda>'
    assert get_func_code(ff)[1] == __file__.replace('.pyc', '.py')
    # Simulate a function defined in __main__
    ff.__module__ = '__main__'
    assert get_func_name(ff, win_characters=False)[-1] == '<lambda>'
    assert get_func_code(ff)[1] == __file__.replace('.pyc', '.py')


def func_with_kwonly_args(a, b, kw1='kw1', kw2='kw2'):
    pass


def func_with_signature(a, b):
    pass

if PY3_OR_LATER:
    exec("""
def func_with_kwonly_args(a, b, *, kw1='kw1', kw2='kw2'): pass

def func_with_signature(a: int, b: int) -> None: pass
""")

    def test_filter_args_python_3():
        assert (
            filter_args(func_with_kwonly_args, [], (1, 2),
                        {'kw1': 3, 'kw2': 4}) ==
            {'a': 1, 'b': 2, 'kw1': 3, 'kw2': 4})

        # filter_args doesn't care about keyword-only arguments so you
        # can pass 'kw1' into *args without any problem
        assert_raises_regex(
            ValueError,
            "Keyword-only parameter 'kw1' was passed as positional parameter",
            filter_args,
            func_with_kwonly_args, [], (1, 2, 3), {'kw2': 2})

        assert (
            filter_args(func_with_kwonly_args, ['b', 'kw2'], (1, 2),
                        {'kw1': 3, 'kw2': 4}) ==
            {'a': 1, 'kw1': 3})

        assert (filter_args(func_with_signature, ['b'], (1, 2)) == {'a': 1})


def test_bound_methods():
    """ Make sure that calling the same method on two different instances
        of the same class does resolv to different signatures.
    """
    a = Klass()
    b = Klass()
    assert filter_args(a.f, [], (1, )) != filter_args(b.f, [], (1, ))


def test_filter_args_error_msg():
    """ Make sure that filter_args returns decent error messages, for the
        sake of the user.
    """
    assert_raises(ValueError, filter_args, f, [])


def test_clean_win_chars():
    string = r'C:\foo\bar\main.py'
    mangled_string = _clean_win_chars(string)
    for char in ('\\', ':', '<', '>', '!'):
        assert char not in mangled_string


def test_format_signature():
    # Test signature formatting.
    path, sgn = format_signature(g, list(range(10)))
    assert sgn == 'g([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])'
    path, sgn = format_signature(g, list(range(10)), y=list(range(10)))
    assert sgn == 'g([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],' \
                  ' y=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])'


@with_numpy
def test_format_signature_numpy():
    """ Test the format signature formatting with numpy.
    """


def test_special_source_encoding():
    from joblib.test.test_func_inspect_special_encoding import big5_f
    func_code, source_file, first_line = get_func_code(big5_f)
    assert first_line == 5
    assert "def big5_f():" in func_code
    assert "test_func_inspect_special_encoding" in source_file


def _get_code():
    from joblib.test.test_func_inspect_special_encoding import big5_f
    return get_func_code(big5_f)[0]


def test_func_code_consistency():
    from joblib.parallel import Parallel, delayed
    codes = Parallel(n_jobs=2)(delayed(_get_code)() for _ in range(5))
    assert len(set(codes)) == 1
