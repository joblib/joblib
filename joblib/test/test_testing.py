import sys
import re

from nose.tools import assert_raises

from joblib.testing import assert_raises_regex, check_subprocess_call


def test_check_subprocess_call():
    code = '\n'.join(['result = 1 + 2 * 3',
                      'print(result)',
                      'my_list = [1, 2, 3]',
                      'print(my_list)'])

    check_subprocess_call([sys.executable, '-c', code])

    # Now checking stdout with a regex
    check_subprocess_call([sys.executable, '-c', code],
                          # Regex needed for platform-specific line endings
                          stdout_regex=r'7\s{1,2}\[1, 2, 3\]')


def test_check_subprocess_call_non_matching_regex():
    code = '42'
    non_matching_pattern = '_no_way_this_matches_anything_'
    assert_raises_regex(ValueError,
                        'Unexpected stdout.+{0}'.format(non_matching_pattern),
                        check_subprocess_call,
                        [sys.executable, '-c', code],
                        stdout_regex=non_matching_pattern)


def test_check_subprocess_call_wrong_command():
    wrong_command = '_a_command_that_does_not_exist_'
    assert_raises(OSError,
                  check_subprocess_call,
                  [wrong_command])


def test_check_subprocess_call_non_zero_return_code():
    code_with_non_zero_exit = '\n'.join([
        'import sys',
        'print("writing on stdout")',
        'sys.stderr.write("writing on stderr")',
        'sys.exit(123)'])

    pattern = re.compile('Non-zero return code: 123.+'
                         'Stdout:\nwriting on stdout.+'
                         'Stderr:\nwriting on stderr', re.DOTALL)
    assert_raises_regex(ValueError,
                        pattern,
                        check_subprocess_call,
                        [sys.executable, '-c', code_with_non_zero_exit])


def test_check_subprocess_call_timeout():
    code_timing_out = '\n'.join([
        'import time',
        'import sys',
        'print("before sleep on stdout")',
        'sys.stdout.flush()',
        'sys.stderr.write("before sleep on stderr")',
        'sys.stderr.flush()',
        'time.sleep(1.1)',
        'print("process should have be killed before")',
        'sys.stdout.flush()'])

    pattern = re.compile('Non-zero return code:.+'
                         'Stdout:\nbefore sleep on stdout\s+'
                         'Stderr:\nbefore sleep on stderr',
                         re.DOTALL)

    assert_raises_regex(ValueError,
                        pattern,
                        check_subprocess_call,
                        [sys.executable, '-c', code_timing_out],
                        timeout=1)
