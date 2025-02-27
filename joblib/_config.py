import threading
from contextlib import contextmanager

_global_config = {}

_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    return _get_threadlocal_config().copy()


def set_config(**kwargs):
    local_config = _get_threadlocal_config()
    for key, value in kwargs.items():
        local_config[key] = value


@contextmanager
def config_context(**kwargs):
    old_config = get_config()
    set_config(**kwargs)
    try:
        yield
    finally:
        set_config(**old_config)
