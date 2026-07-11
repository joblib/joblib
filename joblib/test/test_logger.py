"""
Test the logger module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.
import re

from joblib.logger import PrintTime, _squeeze_time, format_time, short_format_time


def test_format_time():
    # format_time always reports both seconds and minutes.
    t = _squeeze_time(90.0)
    assert format_time(90.0) == "%.1fs, %.1fmin" % (t, t / 60.0)


def test_short_format_time_seconds():
    # Below a minute, short_format_time reports seconds.
    t = _squeeze_time(42.0)
    assert short_format_time(42.0) == " %5.1fs" % t
    assert short_format_time(42.0).endswith("s")


def test_short_format_time_minutes():
    # Above a minute, it switches to minutes.
    t = _squeeze_time(90.0)
    assert short_format_time(90.0) == "%4.1fmin" % (t / 60.0)
    assert short_format_time(90.0).endswith("min")


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
