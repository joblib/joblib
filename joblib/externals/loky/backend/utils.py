import os
import sys
import errno
import psutil
import signal
import warnings
import threading
import subprocess
try:
    import psutil
except ImportError:
    psutil = None


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def recursive_terminate(process, use_psutil=True):
    if use_psutil and psutil is not None:
        _recursive_terminate_with_psutil(process)
    else:
        _recursive_terminate_without_psutil(process)


def _recursive_terminate_with_psutil(process, retries=5):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        return

    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, still_alive = psutil.wait_procs(children, timeout=5)
    for child_process in still_alive:
        child_process.kill()

    process.terminate()
    process.join()


def _recursive_terminate_without_psutil(process):
    """Terminate a process and its descendants.
    """
    try:
        _recursive_terminate(process.pid)
    except OSError as e:
        warnings.warn("Failed to kill subprocesses on this platform. Please"
                      "install psutil: https://github.com/giampaolo/psutil")
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
