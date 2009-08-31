"""
Helpers for logging.

This module needs much love to become useful.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org> 
# Copyright (c) 2008 Gael Varoquaux
# License: BSD Style, 3 clauses.


import time
import sys
import os
import shutil

import logging

################################################################################
# class `Logger`
################################################################################
class Logger(object):
    """ Base class for logging messages.
    """
    
    def warn(self, msg):
        logging.warn("[%s]: %s" % (self, msg))

    def debug(self, msg):
        logging.debug("[%s]: %s" % (self, msg))



################################################################################
# class `PrintTime`
################################################################################
class PrintTime(object):
    """ Print and log messages while keeping track of time.
    """

    def __init__(self, logfile=None):
        self.last_time = time.time()
        self.logfile = logfile
        if logfile is not None:
            if not os.path.exists(os.path.dirname(logfile)):
                os.makedirs(os.path.dirname(logfile))
            if os.path.exists(logfile):
                # Rotate the logs
                for i in range(1, 9):
                    if os.path.exists(logfile+'.%i' % i):
                        try:
                            shutil.move(logfile+'.%i' % i, 
                                        logfile+'.%i' % (i+1))
                        except:
                            "No reason failing here"
                # Use a copy rather than a move, so that a process
                # monitoring this file does not get lost.
                try:
                    shutil.copy(logfile, logfile+'.1')
                except:
                    "No reason failing here"
            try:
                logfile = file(logfile, 'w')
                logfile.write('\nLogging joblib python script\n')
                logfile.write('\n---%s---\n' % time.ctime(self.last_time))
            except:
                """ Multiprocessing writing to files can create race
                    conditions. Rather fail silently than crash the
                    caculation.
                """
                # XXX: We actually need a debug flag to disable this
                # silent failure.


    def __call__(self, msg=''):
        """ Print the time elapsed between the last call and the current
            call, with an optional message.
        """
        time_lapse = time.time() - self.last_time
        full_msg = "%s: %.2fs, %.1f min" % (msg, time_lapse, time_lapse/60)
        print >> sys.stderr, full_msg
        if self.logfile is not None:
            try:
                print >> file(self.logfile, 'a'), full_msg
            except:
                """ Multiprocessing writing to files can create race
                    conditions. Rather fail silently than crash the
                    caculation.
                """
                # XXX: We actually need a debug flag to disable this
                # silent failure.
        self.last_time = time.time()



