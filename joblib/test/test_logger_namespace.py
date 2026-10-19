import logging
import subprocess
import sys

import pytest

from joblib import Logger


@pytest.mark.thread_unsafe  # caplog mutates shared logging state
@pytest.mark.parametrize("name", [None, "pipeline"])
@pytest.mark.parametrize("method", ["info", "warn", "debug"])
def test_logger_uses_its_namespace(caplog, name, method):
    expected = name or "joblib"
    with caplog.at_level(logging.DEBUG, logger=expected):
        getattr(Logger(name=name), method)("pipeline finished")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == expected
    assert record.getMessage().endswith("pipeline finished")


def test_info_does_not_configure_root_logger():
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import logging; from joblib import Logger; "
            "root = logging.getLogger(); assert not root.handlers; "
            "Logger().info('finished'); assert not root.handlers",
        ],
        check=True,
    )
