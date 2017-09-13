import threading


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True


def _is_terminated(thread):
    """Check if a thread has terminated for any version of python."""
    if thread is None:
        return False
    if hasattr(thread, "_started"):
        return thread._started.is_set() and not thread.is_alive()
    else:
        return thread._Thread__started.is_set() and not thread.is_alive()


def _is_crashed(thread):
    """Check if a thread has terminated unexpectedly for any version of python.

    The thread should be flagged using ``flag_current_thread_clean_exit()`` for
    all the cases where there was no crash. This permits to avoid false
    positive. If this flag is not set, this function return True whenever a
    thread is stopped.
    """
    return _is_terminated(thread) and not getattr(thread, "_clean_exit", False)
