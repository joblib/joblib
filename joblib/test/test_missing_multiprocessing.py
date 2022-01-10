"""
Pyodide and other single-threaded Python builds will be missing the
_multiprocessing module. Test that joblib still works in this environment.
"""

import os
import subprocess
import sys

def test_missing_multiprocessing():
    """
    Test that import joblib works even if _multiprocessing is missing.

    pytest has already imported everything from joblib, so the easiest way to
    test importing it, we need to invoke a separate Python process. This also
    makes it easy to ensure that we don't break other tests by importing a bad
    `_multiprocessing` module.
    """
    env = dict(os.environ)
    # For subprocess, use current sys.path with our custom version of
    # multiprocessing inserted.
    env["PYTHONPATH"] = ":".join(["./test/missing_multiprocessing"] + sys.path)
    subprocess.check_call([sys.executable, "-c", "import joblib"], env=env)


