import faulthandler

from joblib.parallel import mp
from joblib.test.common import np
from joblib.testing import skipif, fixture


@fixture(scope='module')
@skipif(np is None or mp is None, 'Numpy or Multiprocessing not available')
def parallel_numpy_fixture(request):
    """Fixture to skip memmapping test if numpy or multiprocessing is not
    installed"""
    def setup(module):
        faulthandler.dump_traceback_later(timeout=300, exit=True)

        def teardown():
            faulthandler.cancel_dump_traceback_later()
        request.addfinalizer(teardown)

        return parallel_numpy_fixture
    return setup
