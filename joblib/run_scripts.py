"""
Useful functions to run scripts as files.
"""
import time
import sys
import os

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
        if logfile is not None and os.path.exists(logfile):
            os.remove(logfile)

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



if __name__ == '__main__':
    # Some tests
    test_default_param()

