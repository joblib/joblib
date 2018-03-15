import threading


def _flag_current_thread_clean_exit():
    """Put a ``_clean_exit`` flag on the current thread"""
    thread = threading.current_thread()
    thread._clean_exit = True
