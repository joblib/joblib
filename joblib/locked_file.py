""" Efficient file locking for posix and windows.
"""
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD

import os
if os.name == 'nt':
    import msvcrt

    LK_UNLCK = 0 # unlock the file region
    LK_LOCK = 1 # lock the file region
    LK_NBLCK = 2 # non-blocking lock
    LK_RLCK = 3 # lock for writing
    LK_NBRLCK = 4 # non-blocking lock for writing

    def locker(fd):
        # We are locking only the first 512 bytes of the file
        return msvcrt.locking(fd, LK_RLCK, 512)
elif os.name == 'posix':
    import fcntl
    def locker(fd):
        try:
            return fcntl.flock(fd, fcntl.LOCK_EX)
        except:
            # This fails on an NFS drive or a BSD system:
            return fcntl.fcntl(fd, fcntl.F_EXLCK)
else:
    raise Exception('Unsupported platform')


################################################################################
class LockedFile(object):
    """ A minimalistic file-like object that establishes a OS-level lock.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(file_name, 'r+')
        locker(self.file.fileno())
        self.close = self.file.close
        self.write = self.file.write
        self.read  = self.file.read
        self.seek  = self.file.seek
        self.readline = self.file.readline

    def __enter__(self):
        """ For use in the with statement.
        """
        return self

    def __exit__(self, type, value, traceback):
        # This also unlocks the file
        self.close()

