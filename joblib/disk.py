"""
Disk management utilities.
"""
import platform
import os

def disk_used(path):
    """ Return the disk usage in a directory. 
    """
    size = 0
    for file in os.listdir(path):
        size += os.stat(os.path.join(path, file)).st_size
    return size/1024


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
    return available, 100*available/float(capacity)



