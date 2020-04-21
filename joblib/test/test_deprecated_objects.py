"""
Tests making sure that deprecated objects properly raise a deprecation warning
when imported/created.
"""
from joblib import my_exceptions
from joblib import _deprecated_my_exceptions
from joblib.my_exceptions import _deprecated_names as _deprecated_exceptions


def test_deprecated_joblib_exceptions(capsys):
    assert 'JoblibException' in _deprecated_exceptions
    for name in _deprecated_exceptions:
        obj = getattr(my_exceptions, name)
        assert obj is getattr(_deprecated_my_exceptions, name)

        msg = (f'UserWarning: {name} is deprecated and will be removed from '
               f'joblib in 0.16')
        out, err = capsys.readouterr()
        assert msg in err
