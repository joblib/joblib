import threading
from contextlib import contextmanager

_global_config = {"parameter": True}

_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    return _get_threadlocal_config().copy()


def set_config(*, parameter=None):
    local_config = _get_threadlocal_config()
    if parameter is not None:
        local_config["parameter"] = parameter


@contextmanager
def config_context(*, parameter=None):
    old_config = get_config()
    set_config(parameter=parameter)
    try:
        yield
    finally:
        set_config(**old_config)