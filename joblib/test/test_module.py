import sys
import joblib
from joblib.testing import check_subprocess_call


def test_version():
    assert hasattr(joblib, '__version__'), (
        "There are no __version__ argument on the joblib module")


def test_import():
    # check that the import does not set the start_method for multiprocessing
    code = """if True:
        import joblib
        import multiprocessing as mp
        mp.set_start_method("loky")
    """
    check_subprocess_call([sys.executable, '-c', code])
