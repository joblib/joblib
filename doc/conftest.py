from joblib.parallel import mp
from joblib.test.common import np, setup_autokill, teardown_autokill
from joblib.testing import skipif, fixture


@fixture(scope='module')
@skipif(np is None or mp is None, 'Numpy or Multiprocessing not available')
def parallel_numpy_fixture(request):
    """Fixture to skip memmapping test if numpy is not installed"""
    def setup(module):
        setup_autokill(module.__name__, timeout=300)

        def teardown():
            teardown_autokill(module.__name__)
        request.addfinalizer(teardown)

        return parallel_numpy_fixture
    return setup
