from joblib.testing import fixture


@fixture(scope='function')
def tmpdir_path(tmpdir):
    return tmpdir.strpath
