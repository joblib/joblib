"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.
import logging
import re

from joblib.logger import Logger, PrintTime


def test_print_time(tmpdir, capsys):
    # A simple smoke test for PrintTime.
    logfile = tmpdir.join("test.log").strpath
    print_time = PrintTime(logfile=logfile)
    print_time("Foo")
    # Create a second time, to smoke test log rotation.
    print_time = PrintTime(logfile=logfile)
    print_time("Foo")
    # And a third time
    print_time = PrintTime(logfile=logfile)
    print_time("Foo")

    out_printed_text, err_printed_text = capsys.readouterr()
    # Use regexps to be robust to time variations
    match = r"Foo: 0\..s, 0\..min\nFoo: 0\..s, 0..min\nFoo: " + r".\..s, 0..min\n"
    if not re.match(match, err_printed_text):
        raise AssertionError("Excepted %s, got %s" % (match, err_printed_text))


def test_logger_info_uses_named_logger(caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="pipeline")

    logger = Logger(name="pipeline")
    logger.info("finished")

    assert any(
        record.name == "pipeline" and "finished" in record.getMessage()
        for record in caplog.records
    )
