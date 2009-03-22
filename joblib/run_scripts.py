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


def default_param(name, default_value):
    """ Returns the value of the variable 'name' if it is defined, and if 
        not the default value given.
    """
    frame = sys._getframe(1)
    if name in frame.f_locals:
        return frame.f_locals[name]
    elif name in frame.f_globals:
        return frame.f_globals[name]
    else:
        return default_value

def run_script(filename, **params):
    """ Runs the script specified by the filename, with the given
        parameters.
    """
    start_time = time.time()
    if len(params) > 0:
        print >> sys.stderr, "Running %s, with parameters %s" % (
                                filename, 
                                ", ".join("%s=%s" % (k, v) 
                                        for k, v in params.iteritems()) )
    else:
        print >> sys.stderr, "Running %s" % filename 
    namespace = params.copy()
    namespace['__name__'] = '__main__'
    namespace['__file__'] = os.path.abspath(filename)
    execfile(filename, namespace)
    time_lapse = time.time() - start_time
    print >> sys.stderr, "Ran %s in %.2fs, %.1f min" % (filename, time_lapse,
                                                    time_lapse/60)
    return Bunch(**namespace)


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



