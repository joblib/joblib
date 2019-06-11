###############################################################################
# Server process to keep track of unlinked resources, like folders and
# semaphores and clean them.
#
# author: Thomas Moreau
#
# adapted from multiprocessing/semaphore_tracker.py  (17/02/2017)
#  * include custom spawnv_passfds to start the process
#  * use custom unlink from our own SemLock implementation
#  * add some VERBOSE logging
#

#
# On Unix we run a server process which keeps track of unlinked
# resources. The server ignores SIGINT and SIGTERM and reads from a
# pipe.  Every other process of the program has a copy of the writable
# end of the pipe, so we get EOF when all other processes have exited.
# Then the server process unlinks any remaining resources.
#
# For semaphores, this is important because the system only supports a limited
# number of named semaphores, and they will not be automatically removed till
# the next reboot.  Without this resource tracker process, "killall python"
# would probably leave unlinked semaphores.

import os
import shutil
import sys
import signal
import warnings
import threading

from . import spawn
from multiprocessing import util

if sys.platform == "win32":
    from .compat_win32 import _winapi
    from .reduction import duplicate
    import msvcrt

try:
    from _multiprocessing import sem_unlink
except ImportError:
    from .semlock import sem_unlink

if sys.version_info < (3,):
    BrokenPipeError = OSError
    from os import fdopen as open

__all__ = ['ensure_running', 'register', 'unregister']

_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

_CLEANUP_FUNCS = {
    'folder': shutil.rmtree
}

if os.name == "posix":
    _CLEANUP_FUNCS['semlock'] = sem_unlink

VERBOSE = False


class ResourceTracker(object):

    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None

    def getfd(self):
        self.ensure_running()
        return self._fd

    def ensure_running(self):
        '''Make sure that resource tracker process is running.

        This can be run from any process.  Usually a child process will use
        the resource created by its parent.'''
        with self._lock:
            if self._fd is not None:
                # resource tracker was launched before, is it still running?
                if self._check_alive():
                    # => still alive
                    return
                # => dead, launch it again
                os.close(self._fd)
                if os.name == "posix":
                    try:
                        # At this point, the resource_tracker process has been
                        # killed or crashed. Let's remove the process entry
                        # from the process table to avoid zombie processes.
                        os.waitpid(self._pid, 0)
                    except OSError:
                        # The process was terminated or is a child from an
                        # ancestor of the current process.
                        pass
                self._fd = None
                self._pid = None

                warnings.warn('resource_tracker: process died unexpectedly, '
                              'relaunching.  Some folders/sempahores might '
                              'leak.')

            fds_to_pass = []
            try:
                fds_to_pass.append(sys.stderr.fileno())
            except Exception:
                pass

            r, w = os.pipe()
            if sys.platform == "win32":
                _r = duplicate(msvcrt.get_osfhandle(r), inheritable=True)
                os.close(r)
                r = _r

            cmd = 'from {} import main; main({}, {})'.format(
                main.__module__, r, VERBOSE)
            try:
                fds_to_pass.append(r)
                # process will out live us, so no need to wait on pid
                exe = spawn.get_executable()
                args = [exe] + util._args_from_interpreter_flags()
                # In python 3.3, there is a bug which put `-RRRRR..` instead of
                # `-R` in args. Replace it to get the correct flags.
                # See https://github.com/python/cpython/blob/3.3/Lib/subprocess.py#L488
                if sys.version_info[:2] <= (3, 3):
                    import re
                    for i in range(1, len(args)):
                        args[i] = re.sub("-R+", "-R", args[i])
                args += ['-c', cmd]
                util.debug("launching resource tracker: {}".format(args))
                # bpo-33613: Register a signal mask that will block the
                # signals.  This signal mask will be inherited by the child
                # that is going to be spawned and will protect the child from a
                # race condition that can make the child die before it
                # registers signal handlers for SIGINT and SIGTERM. The mask is
                # unregistered after spawning the child.
                try:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_BLOCK,
                                               _IGNORED_SIGNALS)
                    pid = spawnv_passfds(exe, args, fds_to_pass)
                finally:
                    if _HAVE_SIGMASK:
                        signal.pthread_sigmask(signal.SIG_UNBLOCK,
                                               _IGNORED_SIGNALS)
            except BaseException:
                os.close(w)
                raise
            else:
                self._fd = w
                self._pid = pid
            finally:
                if sys.platform == "win32":
                    _winapi.CloseHandle(r)
                else:
                    os.close(r)

    def _check_alive(self):
        '''Check for the existence of the resource tracker process.'''
        try:
            self._send('PROBE', '', '')
        except BrokenPipeError:
            return False
        else:
            return True

    def register(self, name, rtype):
        '''Register a named resource with resource tracker.'''
        self.ensure_running()
        self._send('REGISTER', name, rtype)

    def unregister(self, name, rtype):
        '''Unregister a named resource with resource tracker.'''
        self.ensure_running()
        self._send('UNREGISTER', name, rtype)

    def _send(self, cmd, name, rtype):
        msg = '{0}:{1}:{2}\n'.format(cmd, name, rtype).encode('ascii')
        if len(name) > 512:
            # posix guarantees that writes to a pipe of less than PIPE_BUF
            # bytes are atomic, and that PIPE_BUF >= 512
            raise ValueError('name too long')
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg)


_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd


def main(fd, verbose=0):
    '''Run resource tracker.'''
    # protect the process from ^C and "killall python" etc
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if _HAVE_SIGMASK:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)

    for f in (sys.stdin, sys.stdout):
        try:
            f.close()
        except Exception:
            pass

    if verbose:  # pragma: no cover
        sys.stderr.write("Main resource tracker is running\n")
        sys.stderr.flush()

    cache = {rtype: set() for rtype in _CLEANUP_FUNCS.keys()}
    try:
        # keep track of registered/unregistered resources
        if sys.platform == "win32":
            fd = msvcrt.open_osfhandle(fd, os.O_RDONLY)
        with open(fd, 'rb') as f:
            for line in f:
                try:
                    splitted = line.strip().decode('ascii').split(':')
                    # name can potentially contain separator symbols (for
                    # instance folders on Windows)
                    cmd, name, rtype = (
                        splitted[0], ':'.join(splitted[1:-1]), splitted[-1])

                    if cmd == 'PROBE':
                        continue

                    if rtype not in _CLEANUP_FUNCS:
                        raise ValueError(
                            'Cannot register {} for automatic cleanup: '
                            'unknown resource type ({}). Resource type should '
                            'be one of the following: {}'.format(
                                name, rtype, list(_CLEANUP_FUNCS.keys())))

                    if cmd == 'REGISTER':
                        cache[rtype].add(name)
                        if verbose:  # pragma: no cover
                            sys.stderr.write("[ResourceTracker] register {}"
                                             " {}\n" .format(rtype, name))
                            sys.stderr.flush()
                    elif cmd == 'UNREGISTER':
                        cache[rtype].remove(name)
                        if verbose:  # pragma: no cover
                            sys.stderr.write("[ResourceTracker] unregister {}"
                                             " {}: cache({})\n"
                                             .format(name, rtype, len(cache)))
                            sys.stderr.flush()
                    else:
                        raise RuntimeError('unrecognized command %r' % cmd)
                except BaseException:
                    try:
                        sys.excepthook(*sys.exc_info())
                    except BaseException:
                        pass
    finally:
        # all processes have terminated; cleanup any remaining resources
        for rtype, rtype_cache in cache.items():
            if rtype_cache:
                try:
                    warnings.warn('resource_tracker: There appear to be %d '
                                  'leaked %s objects to clean up at shutdown' %
                                  (len(rtype_cache), rtype))
                except Exception:
                    pass
            for name in rtype_cache:
                # For some reason the process which created and registered this
                # resource has failed to unregister it. Presumably it has
                # died.  We therefore clean it up.
                try:
                    _CLEANUP_FUNCS[rtype](name)
                    if verbose:  # pragma: no cover
                        sys.stderr.write("[ResourceTracker] unlink {}\n"
                                         .format(name))
                        sys.stderr.flush()
                except Exception as e:
                    warnings.warn('resource_tracker: %s: %r' % (name, e))

    if verbose:  # pragma: no cover
        sys.stderr.write("resource tracker shut down\n")
        sys.stderr.flush()


#
# Start a program with only specified fds kept open
#

def spawnv_passfds(path, args, passfds):
    passfds = sorted(passfds)
    if sys.platform != "win32":
        errpipe_read, errpipe_write = os.pipe()
        try:
            from .reduction import _mk_inheritable
            _pass = []
            for fd in passfds:
                _pass += [_mk_inheritable(fd)]
            from .fork_exec import fork_exec
            return fork_exec(args, _pass)
        finally:
            os.close(errpipe_read)
            os.close(errpipe_write)
    else:
        cmd = ' '.join('"%s"' % x for x in args)
        try:
            hp, ht, pid, tid = _winapi.CreateProcess(
                path, cmd, None, None, True, 0, None, None, None)
            _winapi.CloseHandle(ht)
        except BaseException:
            pass
        return pid
