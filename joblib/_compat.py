"""
Compatibility layer for Python 3/Python 2 single codebase
"""
_basestring = str
_bytes_or_unicode = (bytes, str)


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    return meta("NewBase", bases, {})


CompatFileExistsError = FileExistsError
