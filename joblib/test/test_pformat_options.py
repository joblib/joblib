import pytest

from joblib.logger import pformat


@pytest.mark.parametrize("error_type", [ValueError, KeyboardInterrupt])
def test_pformat_restores_numpy_options_after_error(error_type):
    np = pytest.importorskip("numpy")

    class BadRepr:
        def __repr__(self):
            raise error_type("cannot format")

    with np.printoptions(precision=9, threshold=11, edgeitems=2):
        before = np.get_printoptions()
        with pytest.raises(error_type, match="cannot format"):
            pformat(BadRepr())
        assert np.get_printoptions() == before


def test_pformat_restores_numpy_options_after_success():
    np = pytest.importorskip("numpy")
    with np.printoptions(precision=9, threshold=11, edgeitems=2):
        before = np.get_printoptions()
        assert pformat([1, 2]) == "[1, 2]"
        assert np.get_printoptions() == before
