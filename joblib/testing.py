"""
Helper for testing.
"""

import sys
import warnings
import os.path
import re
import subprocess
import threading
import unittest

import nose
import pytest
import _pytest

from joblib._compat import PY3_OR_LATER


_dummy = unittest.TestCase('__init__')
assert_true = _dummy.assertTrue
assert_false = _dummy.assertFalse
assert_equal = _dummy.assertEqual
assert_not_equal = _dummy.assertNotEqual
assert_raises = _dummy.assertRaises

try:
    assert_raises_regex = _dummy.assertRaisesRegexp
except AttributeError:
    def assert_raises_regex(expected_exception, expected_regexp,
                            callable_obj=None, *args, **kwargs):
        """Helper function to check for message patterns in exceptions"""
        not_raised = False
        try:
            callable_obj(*args, **kwargs)
            not_raised = True
        except Exception as e:
            error_message = str(e)
            if not re.compile(expected_regexp).search(error_message):
                raise AssertionError("Error message should match pattern "
                                     "%r. %r does not." %
                                     (expected_regexp, error_message))
        if not_raised:
            raise AssertionError("Should have raised %r" %
                                 expected_exception(expected_regexp))

SkipTest = _pytest.runner.Skipped
skipif = pytest.mark.skipif
with_setup = nose.tools.with_setup
fixture = pytest.fixture


def warnings_to_stdout():
    """ Redirect all warnings to stdout.
    """
    showwarning_orig = warnings.showwarning

    def showwarning(msg, cat, fname, lno, file=None, line=0):
        showwarning_orig(msg, cat, os.path.basename(fname), line, sys.stdout)

    warnings.showwarning = showwarning
    # warnings.simplefilter('always')


def check_subprocess_call(cmd, timeout=1, stdout_regex=None,
                          stderr_regex=None):
    """Runs a command in a subprocess with timeout in seconds.

    Also checks returncode is zero, stdout if stdout_regex is set, and
    stderr if stderr_regex is set.
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    def kill_process():
        proc.kill()

    timer = threading.Timer(timeout, kill_process)
    try:
        timer.start()
        stdout, stderr = proc.communicate()

        if PY3_OR_LATER:
            stdout, stderr = stdout.decode(), stderr.decode()
        if proc.returncode != 0:
            message = (
                'Non-zero return code: {}.\nStdout:\n{}\n'
                'Stderr:\n{}').format(
                    proc.returncode, stdout, stderr)
            raise ValueError(message)

        if (stdout_regex is not None and
                not re.search(stdout_regex, stdout)):
            raise ValueError(
                "Unexpected stdout: {!r} does not match:\n{!r}".format(
                    stdout_regex, stdout))
        if (stderr_regex is not None and
                not re.search(stderr_regex, stderr)):
            raise ValueError(
                "Unexpected stderr: {!r} does not match:\n{!r}".format(
                    stderr_regex, stderr))

    finally:
        timer.cancel()
