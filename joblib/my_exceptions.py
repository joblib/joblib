from warnings import warn
from . import _deprecated_my_exceptions

"""
Exceptions
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

_deprecated_names = [
    name for name in dir(_deprecated_my_exceptions) if not
    name.startswith("__")
]


def __getattr__(name):
    if not name.startswith("__") and name in _deprecated_names:
        warn(f"{name} is deprecated and will be removed from joblib in 0.16")
        return getattr(_deprecated_my_exceptions, name)
    raise AttributeError


class WorkerInterrupt(Exception):
    """ An exception that is not KeyboardInterrupt to allow subprocesses
        to be interrupted.
    """

    pass
