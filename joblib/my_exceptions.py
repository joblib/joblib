"""
Exceptions
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

import sys
import cPickle as pickle


class JoblibException(Exception):
    """A simple exception with an error message that you can get to."""

    def __init__(self, *args):
        self.args = args

    def __reduce__(self):
        # For pickling
        return self.__class__, self.args, {}

    def __repr__(self):
        if hasattr(self, 'args'):
            message = self.args[0]
        else:
            # Python 2 compat: instances of JoblibException can be created
            # without calling JoblibException __init__ in case of
            # multi-inheritance: in that case the message is stored as an
            # explicit attribute under Python 2 (only)
            message = self.message
        name = self.__class__.__name__
        return '%s\n%s\n%s\n%s' % (name, 75 * '_', message, 75 * '_')

    __str__ = __repr__


class TransportableException(JoblibException):
    """An exception containing all the info to wrap an original
        exception and recreate it.
    """

    def __init__(self, message, etype, cause=None):
        self.message = message
        self.etype = etype
        if isinstance(cause, Exception):
            try:
                cause = pickle.dumps(cause)
            except:
                # cause cannot be pickled
                cause = None
        self.cause = cause

    def __reduce__(self):
        # For pickling
        return self.__class__, (self.message, self.etype), {'cause': self.cause}

    def mk_cause(self):
        cause = None
        if self.cause is not None:
            try:
                cause = pickle.loads(self.cause)
            except TypeError:
                # cause is not pickle-able
                cause = None
        return cause


_exception_mapping = dict()


def _mk_exception(exception, name=None):
    # Create an exception inheriting from both JoblibException
    # and that exception
    if name is None:
        name = exception.__name__
    this_name = 'Joblib%s' % name
    if this_name in _exception_mapping:
        # Avoid creating twice the same exception
        this_exception = _exception_mapping[this_name]
    else:
        if exception is Exception:
            # We cannot create a subclass: we are already a trivial
            # subclass
            return JoblibException, this_name
        this_exception = type(this_name, (exception, JoblibException),
                    dict(__repr__=JoblibException.__repr__,
                        __str__=JoblibException.__str__),
                    )
        _exception_mapping[this_name] = this_exception
    return this_exception, this_name


def _mk_common_exceptions():
    namespace = dict()
    if sys.version_info[0] == 3:
        import builtins as _builtin_exceptions
        common_exceptions = filter(
            lambda x: x.endswith('Error'),
            dir(_builtin_exceptions))
    else:
        import exceptions as _builtin_exceptions
        common_exceptions = dir(_builtin_exceptions)

    for name in common_exceptions:
        obj = getattr(_builtin_exceptions, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            try:
                this_obj, this_name = _mk_exception(obj, name=name)
                namespace[this_name] = this_obj
            except TypeError:
                # Cannot create a consistent method resolution order:
                # a class that we can't subclass properly, probably
                # BaseException
                pass
    return namespace


# Updating module locals so that the exceptions pickle right. AFAIK this
# works only at module-creation time
locals().update(_mk_common_exceptions())
