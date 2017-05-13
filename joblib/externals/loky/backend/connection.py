###############################################################################
# Compat file to load the correct wait function
# 
# author: Thomas Moreau and Olivier grisel
#
import sys

if sys.version_info < (3, 3):
    if sys.platform == "win32":
        from ._win_wait import wait
    else:
        from ._posix_wait import wait
else:
    from multiprocessing.connection import wait
