import faulthandler
import os

import pytest

from joblib.parallel import mp
from joblib.test.common import np
from joblib.testing import skipif, fixture

skip = (
    np is None or mp is None
    or os.getenv('JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND', None)
)
reason = 'Numpy or Multiprocessing not available or JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND set'

@fixture(scope='module')
def parallel_numpy_fixture(request):
    """Fixture to skip memmapping test if numpy or multiprocessing is not
    installed"""
    def setup(module):
        if skip:
            pytest.skip(reason)

        faulthandler.dump_traceback_later(timeout=300, exit=True)

        def teardown():
            faulthandler.cancel_dump_traceback_later()
        request.addfinalizer(teardown)

        return parallel_numpy_fixture
    return setup
