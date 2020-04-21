"""
Tests making sure that deprecated objects properly raise a deprecation warning
when imported/created.
"""
from joblib import my_exceptions
from joblib.my_exceptions import _deprecated_names


def test_deprecated_joblib_exceptions(capsys):
    assert 'JoblibException' in _deprecated_names
    for name in _deprecated_names:
        msg = (f'UserWarning: {name} is deprecated and will be removed from '
               f'joblib in 0.16')
        _ = getattr(my_exceptions, name)
        out, err = capsys.readouterr()
        assert msg in err
