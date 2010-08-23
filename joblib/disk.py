"""
Disk management utilities.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2010 Gael Varoquaux
# License: BSD Style, 3 clauses.


import platform
import os

def disk_used(path):
    """ Return the disk usage in a directory. 
    """
    size = 0
    for file in os.listdir(path) + ['.']:
        stat =  os.stat(os.path.join(path, file))
        size += stat.st_blocks * 512
    # We need to convert to int to avoid having longs on some systems (we
    # don't want longs to avoid problems we SQLite)
    print path, size
    return int(size/1024.)


def disk_free(path):
    """ Return the disk free in bytes and percentage.
    """
    if platform.system() == 'Windows':
        import ctypes
        capacity = ctypes.c_ulonglong(0)
        available = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                                    ctypes.c_wchar_p(path),
                                    None,
                                    ctypes.pointer(capacity),
                                    ctypes.pointer(available))
        capacity  = capacity.value
        available = available.value
    else:
        cache_st = os.statvfs(path)
        capacity = cache_st.f_bsize * cache_st.f_blocks
        available = cache_st.f_bsize * cache_st.f_bavail
    # We need to convert to int to avoid having longs on some systems (we
    # don't want longs to avoid problems we SQLite)
    return int(available), 100*available/float(capacity)


def memstr_to_kbytes(text):
    """ Convert a memory text to it's value in kilobytes.
    """
    kilo = 1024
    units = dict(K=1, M=kilo, G=kilo**2)
    try:
        size = int(units[text[-1]]*float(text[:-1]))
    except (KeyError, ValueError):
        raise ValueError(
                "Invalid literal for size give: %s (type %s) should be "
                "alike '10G', '500M', '50K'." % (text, type(text))
                )
    return size

