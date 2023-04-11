"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import logging
import re

from joblib.logger import Logger, PrintTime
from joblib.memory import Memory


def test_print_time(tmpdir, capsys):
    # A simple smoke test for PrintTime.
    logfile = tmpdir.join('test.log').strpath
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # Create a second time, to smoke test log rotation.
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')
    # And a third time
    print_time = PrintTime(logfile=logfile)
    print_time('Foo')

    out_printed_text, err_printed_text = capsys.readouterr()
    # Use regexps to be robust to time variations
    match = r"Foo: 0\..s, 0\..min\nFoo: 0\..s, 0..min\nFoo: " + \
            r".\..s, 0..min\n"
    if not re.match(match, err_printed_text):
        raise AssertionError('Excepted %s, got %s' %
                             (match, err_printed_text))


def test_logging_levels(caplog):
    logger = Logger()
    warn_msg = "This is a message at logging level WARNING. "
    info_msg = "This is a message at logging level INFO. "
    debug_msg = "This is a message at logging level DEBUG. "

    logging.basicConfig(level=logging.INFO)
    caplog.set_level(logging.INFO)
    logger.warn(warn_msg)
    logger.info(info_msg)
    logger.debug(debug_msg)

    assert warn_msg in caplog.text
    assert info_msg in caplog.text
    assert debug_msg not in caplog.text
    caplog.clear()

    logging.basicConfig(level=logging.WARNING)
    caplog.set_level(logging.WARNING)
    logger.warn(warn_msg)
    logger.info(info_msg)
    logger.debug(debug_msg)

    assert warn_msg in caplog.text
    assert info_msg not in caplog.text
    assert debug_msg not in caplog.text
    caplog.clear()

    logging.basicConfig(level=logging.DEBUG)
    caplog.set_level(logging.DEBUG)
    logger.warn(warn_msg)
    logger.info(info_msg)
    logger.debug(debug_msg)

    assert warn_msg in caplog.text
    assert info_msg in caplog.text
    assert debug_msg in caplog.text
    caplog.clear()


def test_info_log(tmpdir, caplog):
    caplog.set_level(logging.INFO)
    x = 3

    memory = Memory(location=tmpdir.strpath, verbose=20)

    @memory.cache
    def f(x):
        return x ** 2

    _ = f(x)
    assert "Querying" in caplog.text
    caplog.clear()

    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache
    def f(x):
        return x ** 2

    _ = f(x)
    assert "Querying" not in caplog.text
    caplog.clear()
