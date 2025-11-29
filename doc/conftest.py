import faulthandler

import pytest

from joblib.parallel import mp
from joblib.test.common import np
from joblib.testing import fixture


@fixture(scope="module")
def parallel_numpy_fixture(request):
    """Fixture to skip memmapping test if numpy or multiprocessing is not
    installed"""

    if np is None or mp is None:
        pytest.skip("Numpy or Multiprocessing not available")

    def setup(module):
        faulthandler.dump_traceback_later(timeout=300, exit=True)

        def teardown():
            faulthandler.cancel_dump_traceback_later()

        request.addfinalizer(teardown)

        return parallel_numpy_fixture

    return setup
