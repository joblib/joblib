import os
import time

from tempfile import mkdtemp, mkstemp
from joblib.externals.loky.backend import resource_tracker


def test_file_plus_resource_tracker():
    temp_folder = mkdtemp(dir=os.getcwd())
    fd_one, temp_file_one = mkstemp(dir=temp_folder)
    fd_two, temp_file_two = mkstemp(dir=temp_folder)

    os.close(fd_one)
    os.close(fd_two)

    assert os.path.exists(temp_folder)
    assert os.path.exists(temp_file_one)
    assert os.path.exists(temp_file_two)

    resource_tracker.register(temp_file_one, 'file_plus_plus')
    resource_tracker.register(temp_file_two, 'file_plus_plus')

    resource_tracker.maybe_unlink(temp_file_one, 'file_plus_plus')
    time.sleep(.5)

    assert not os.path.exists(temp_file_one)
    assert os.path.exists(temp_file_two)
    assert os.path.exists(temp_folder)

    resource_tracker.maybe_unlink(temp_file_two, 'file_plus_plus')
    time.sleep(.5)

    assert not os.path.exists(temp_file_two)
    assert not os.path.exists(temp_folder)
