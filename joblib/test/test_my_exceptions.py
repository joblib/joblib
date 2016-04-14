"""
Test my automatically generate exceptions
"""
from nose.tools import assert_true

from joblib import my_exceptions


class CustomException(Exception):
    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d


class CustomException2(Exception):
    """A custom exception with a .args attribute

    Just to check that the JoblibException created from it
    has it args set correctly
    """
    def __init__(self, a, *args):
        self.a = a
        self.args = args


def test_inheritance():
    assert_true(isinstance(my_exceptions.JoblibNameError(), NameError))
    assert_true(isinstance(my_exceptions.JoblibNameError(),
                           my_exceptions.JoblibException))
    assert_true(my_exceptions.JoblibNameError is
                my_exceptions._mk_exception(NameError)[0])


def test_inheritance_special_cases():
    # _mk_exception should transform Exception to JoblibException
    assert_true(my_exceptions._mk_exception(Exception)[0] is
                my_exceptions.JoblibException)

    # Subclasses of JoblibException should be mapped to
    # JoblibException by _mk_exception
    for exception in [my_exceptions.JoblibException,
                      my_exceptions.TransportableException]:
        assert_true(my_exceptions._mk_exception(exception)[0] is
                    my_exceptions.JoblibException)

    # Non-inheritable exception classes should be mapped to
    # JoblibException by _mk_exception. That can happen with classes
    # generated with SWIG. See
    # https://github.com/joblib/joblib/issues/269 for a concrete
    # example.
    non_inheritable_classes = [type(lambda: None), bool]
    for exception in non_inheritable_classes:
        assert_true(my_exceptions._mk_exception(exception)[0] is
                    my_exceptions.JoblibException)


def test__mk_exception():
    # Check that _mk_exception works on a bunch of different exceptions
    for klass in (Exception, TypeError, SyntaxError, ValueError,
                  AssertionError, CustomException, CustomException2):
        message = 'This message should be in the exception repr'
        exc = my_exceptions._mk_exception(klass)[0](
            message,
            'some', 'other', 'args', 'that are not', 'in the repr')
        exc_repr = repr(exc)

        assert_true(isinstance(exc, klass))
        assert_true(isinstance(exc, my_exceptions.JoblibException))
        assert_true(exc.__class__.__name__ in exc_repr)
        assert_true(message in exc_repr)
