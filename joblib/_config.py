import threading
from contextlib import contextmanager

_global_config = {}

_threadlocal = threading.local()


def _get_threadlocal_config():
    """Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    is_global_config_exist = True
    if not hasattr(_threadlocal, "global_config"):
        is_global_config_exist = False
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config, is_global_config_exist


def _get_config():
    threadlocal_config, is_global_config_exist = _get_threadlocal_config()
    return threadlocal_config.copy(), is_global_config_exist


def get_config():
    """Get the configuration that is propagated to the joblib parallel workers.

    Returns
    -------
    config : dict
        The configuration.
    """
    return _get_config()[0]


def set_config(**kwargs):
    """Set the configuration that is propagated to the joblib parallel workers.

    Parameters
    ----------
    config : dict
        The configuration to set.
    """
    local_config, _ = _get_threadlocal_config()
    local_config.clear()
    local_config.update(**kwargs)


@contextmanager
def config_context(**kwargs):
    """Context manager to set the configuration that is propagated to the joblib
    parallel workers.

    Parameters
    ----------
    config : dict
        The configuration to set.
    """
    old_config, is_global_config_exist = _get_config()
    set_config(**kwargs)
    try:
        yield
    finally:
        if is_global_config_exist:
            set_config(**old_config)
        else:
            del _threadlocal.global_config
