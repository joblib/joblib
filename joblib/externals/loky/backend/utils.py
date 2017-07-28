import threading


def flag_current_thread_clean_exit():
    thread = threading.current_thread()
    thread._clean_exit = True
