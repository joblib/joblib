"""
Useful functions to run scripts as files.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.



import time
import sys
import os
import shutil

class Bunch(dict):
    """ A dict that exposes its keys as attributes.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__ = self
        dict.__init__(self, *args, **kwargs)


class PrintTime(object):
    """ An object to print messages while keeping track of time.
    """

    def __init__(self, logfile=None):
        self.last_time = time.time()
        self.logfile = logfile
        if logfile is not None:
            if os.path.exists(logfile):
                # Rotate the logs
                for i in range(1, 9):
                    if os.path.exists(logfile+'.%i' % i):
                        shutil.move(logfile+'.%i' % i, logfile+'.%i' % (i+1))
                # Use a copy rather than a move, so that a process
                # monitoring this file does not get lost.
                shutil.copy(logfile, logfile+'.1')
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
            logfile = file(logfile, 'w')
            logfile.write('\nLogging joblib python script\n')
            logfile.write('\n---%s---\n' % time.ctime(self.last_time))

    def __call__(self, msg=''):
        """ Print the time elapsed between the last call and the current
            call, with an optional message.
        """
        time_lapse = time.time() - self.last_time
        full_msg = "%s: %.2fs, %.1f min" % (msg, time_lapse, time_lapse/60)
        print >> sys.stderr, full_msg
        if self.logfile is not None:
            print >> file(self.logfile, 'a'), full_msg
        self.last_time = time.time()



