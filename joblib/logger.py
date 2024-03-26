"""
Helpers for logging.
"""

import time
import sys
import os
import shutil
import logging
import pprint
from .disk import mkdirp


def _squeeze_time(t):
    """Remove .1s from the time under Windows.

    This is the time it takes to stat files. This is needed to make results
    similar to timings under Unix, for tests.
    """
    if sys.platform.startswith("win"):
        return max(0, t - 0.1)
    else:
        return t


def format_time(t):
    t = _squeeze_time(t)
    return "%.1fs, %.1fmin" % (t, t / 60.0)


def short_format_time(t):
    t = _squeeze_time(t)
    if t > 60:
        return "%4.1fmin" % (t / 60.0)
    else:
        return " %5.1fs" % (t)


def pformat(obj, indent=0, depth=3):
    if "numpy" in sys.modules:
        import numpy as np

        print_options = np.get_printoptions()
        np.set_printoptions(precision=6, threshold=64, edgeitems=1)
    else:
        print_options = None
    out = pprint.pformat(obj, depth=depth, indent=indent)
    if print_options:
        np.set_printoptions(**print_options)
    return out


class Logger:
    """Base class for logging messages."""

    def __init__(self, depth=3, name=None):
        """
        Parameters
        ----------
        depth : int, optional
            The depth of objects printed.
        name : str, optional
            The namespace to log to. If None, defaults to joblib.
        """
        self.depth = depth
        self._name = name if name else "joblib"
        self._logger = logging.getLogger(self._name)

    def warn(self, msg):
        self._logger.warning("[%s]: %s", self, msg)

    def info(self, msg):
        self._logger.info("[%s]: %s", self, msg)

    def debug(self, msg):
        self._logger.debug("[%s]: %s", self, msg)

    def format(self, obj, indent=0):
        """Return the formatted representation of the object."""
        return pformat(obj, indent=indent, depth=self.depth)


class PrintTime:
    """Print and log messages while keeping track of time."""

    def __init__(self, logfile=None, logdir=None):
        if logfile is not None and logdir is not None:
            raise ValueError("Cannot specify both logfile and logdir")
        self.last_time = time.time()
        self.start_time = self.last_time
        self.logfile = None
        if logdir is not None:
            logfile = os.path.join(logdir, "joblib.log")
        if logfile is not None:
            mkdirp(os.path.dirname(logfile))
            if os.path.exists(logfile):
                # Rotate the logs
                for i in range(8, 0, -1):
                    try:
                        shutil.move(logfile + f".{i}", logfile + f".{i+1}")
                    except FileNotFoundError:
                        pass
                try:
                    shutil.copy(logfile, logfile + ".1")
                except FileNotFoundError:
                    pass
            self.logfile = logfile

    def __call__(self, msg="", total=False):
        """Print the time elapsed between the last call and the current call,
        with an optional message.
        """
        if not total:
            time_lapse = time.time() - self.last_time
            full_msg = "%s: %s" % (msg, format_time(time_lapse))
        else:
            time_lapse = time.time() - self.start_time
            full_msg = "%s: %.2fs, %.1f min" % (
                msg,
                time_lapse,
                time_lapse / 60
            )
        print(full_msg, file=sys.stderr)
        if self.logfile is not None:
            try:
                with open(self.logfile, "a") as f:
                    print(full_msg, file=f)
            except IOError:
                # Multiprocessing writing to files can create race conditions.
                # Rather fail silently than crash the calculation.
                pass
        self.last_time = time.time()
