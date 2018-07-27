import os
import sys
import errno
import psutil
import signal
import warnings
import threading
import subprocess


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def recursive_terminate(process):
    """Terminate a process and its descendants.
    """
    try:
        _recursive_terminate(process.pid)
    except OSError as e:
        import traceback
        tb = traceback.format_exc()
        warnings.warn("Failure in child introspection on this platform. You "
                      "should report it on https://github.com/tomMoral/loky "
                      "with the following traceback\n{}".format(tb))
        # In case we cannot introspect the children, we fall back to the
        # classic Process.terminate.
        process.terminate()
    process.join()


def _recursive_terminate(pid):
    """Recursively kill the descendants of a process before killing it.
    """
    process = psutil.Process(pid=pid)
    children = process.children(recursive=True)
    for pid_ in [c.pid for c in children] + [pid]:
        try:
            os.kill(pid_, signal.SIGTERM)
        except OSError as e:
            # if OSError is raised with [Errno 3] no such process, the process
            # is already terminated, else, raise the error and let the top
            # level function raise a warning and retry to kill the process.
            if e.errno != errno.ESRCH:
                raise
