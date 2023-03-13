"""
Exceptions
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause


class WorkerInterrupt(Exception):
    """ An exception that is not KeyboardInterrupt to allow subprocesses
        to be interrupted.
    """

    pass
