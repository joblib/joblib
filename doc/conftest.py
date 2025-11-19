import faulthandler

from joblib.testing import fixture


@fixture(scope="module")
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
