"""
Test my automatically generate exceptions
"""
from nose.tools import assert_true

from joblib import my_exceptions


def test_inheritance():
    assert_true(isinstance(my_exceptions.JoblibNameError(), NameError))
    assert_true(isinstance(my_exceptions.JoblibNameError(),
                            my_exceptions.JoblibException))
    assert_true(my_exceptions.JoblibNameError is
                my_exceptions._mk_exception(NameError)[0])


def test__mk_exception():
    # Check that _mk_exception works on a bunch of different exceptions
    for klass in (Exception, TypeError, SyntaxError, ValueError,
                  AssertionError):
        e = my_exceptions._mk_exception(klass)[0]('Some argument')
        assert_true('Some argument' in repr(e))
