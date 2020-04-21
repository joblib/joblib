"""
Tests making sure that deprecated objects properly raise a deprecation warning
when imported/created.
"""
import sys

from joblib import my_exceptions
from joblib import _deprecated_my_exceptions
from joblib.my_exceptions import _deprecated_names as _deprecated_exceptions

from joblib import format_stack
from joblib import _deprecated_format_stack
from joblib.format_stack import _deprecated_names as _deprecated_format_utils


def test_deprecated_joblib_exceptions(capsys):
    assert 'JoblibException' in _deprecated_exceptions
    for name in _deprecated_exceptions:
        obj = getattr(my_exceptions, name)
        assert obj is getattr(_deprecated_my_exceptions, name)

        msg = (f'UserWarning: {name} is deprecated and will be removed from '
               f'joblib in 0.16')
        out, err = capsys.readouterr()
        if sys.version_info[:2] >= (3, 7):
            assert msg in err


def test_deprecated_formatting_utilities(capsys):
    assert 'safe_repr' in _deprecated_format_utils
    assert 'eq_repr' in _deprecated_format_utils
    for name in _deprecated_format_utils:
        obj = getattr(format_stack, name)
        assert obj is getattr(_deprecated_format_stack, name)

        out, err = capsys.readouterr()
        msg = (f'UserWarning: {name} is deprecated and will be removed from '
               f'joblib in 0.16')
        if sys.version_info[:2] >= (3, 7):
            assert msg in err
