"""
Exceptions
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
#         Pietro Berkes
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

import exceptions

class JoblibException(Exception):
    """ A simple exception with an error message that you can get to.
    """

    def __init__(self, message):
        self.message = message

    def __reduce__(self):
        # For pickling
        return self.__class__, (self.message,), {}

    def __repr__(self):
        return '%s\n%s\n%s\n%s' % (
                    self.__class__.__name__,
                    75*'_',
                    self.message,
                    75*'_')

    __str__ = __repr__



