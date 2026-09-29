###############################################################################
# Server process to keep track of unlinked resources, like folders and
# semaphores and clean them.
#
# author: Thomas Moreau
#
# Adapted from multiprocessing/resource_tracker.py
#  * add some VERBOSE logging,
#  * add support to track folders,
#  * add Windows support,
#  * refcounting scheme to avoid unlinking resources still in use.
#
# On Unix we run a server process which keeps track of unlinked
# resources. The server ignores SIGINT and SIGTERM and reads from a
# pipe. The resource_tracker implements a reference counting scheme: each time
# a Python process anticipates the shared usage of a resource by another
# process, it signals the resource_tracker of this shared usage, and in return,
# the resource_tracker increments the resource's reference count by 1.
# Similarly, when access to a resource is closed by a Python process, the
# process notifies the resource_tracker by asking it to decrement the
# resource's reference count by 1.  When the reference count drops to 0, the
# resource_tracker attempts to clean up the underlying resource.

# Finally, every other process connected to the resource tracker has a copy of
# the writable end of the pipe used to communicate with it, so the resource
# tracker gets EOF when all other processes have exited. Then the
# resource_tracker process unlinks any remaining leaked resources (with
# reference count above 0)

# For semaphores, this is important because the system only supports a limited
# number of named semaphores, and they will not be automatically removed till
# the next reboot.  Without this resource tracker process, "killall python"
# would probably leave unlinked semaphores.

# Note that this behavior differs from CPython's resource_tracker, which only
# implements list of shared resources, and not a proper refcounting scheme.
# Also, CPython's resource tracker will only attempt to cleanup those shared
# resources once all processes connected to the resource tracker have exited.
#
# Source code organization and maintenance:
#
# This file is derived from the matching file in the standard library with
# modifications to add the new features described above.
#
# It defines a subclass of a vendored copy of the ResourceTracker from the
# standard library to make loky less likely to break when internals of
# change in a new version of CPython.
#
# The new features are implemented in the overriden method of the ResourceTracker
# class as well as in the custom `main` function.
#
# The vendored copy is minimally patched and should not be edited by hand.
# Instead, it should be revendored from time to time using the script
# in the `tools/` folder at the root of this repo.
#
# When vendoring a new copy of the resource_tracker module of the standard
# library, it is important to check of the `main` function has evolved
# upstream. If so, doing a diff of that particular function in the upstream
# file and in the current file might be helpful.
#
# Make sure to update the inline comments to help future maintainers understand
# what loky-specific changes were made.

import os
import shutil
import sys
import signal
import warnings
from multiprocessing import util
import base64
import json
import threading

from .stdlib_py314_resource_tracker import (
    ResourceTracker as StdLibResourceTracker,
    _decode_message,
)

from . import spawn

if sys.platform == "win32":
    import _winapi
    import msvcrt
    from multiprocessing.reduction import duplicate

# To minimize diff vs stdlib
# fmt:off
__all__ = ['ensure_running', 'register', 'unregister']

_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)

def cleanup_noop(name):
    raise RuntimeError('noop should never be registered or cleaned up')


_CLEANUP_FUNCS = {
    'noop': cleanup_noop,
    'dummy': lambda name: None,  # Dummy resource used in tests
    # loky: add 'folder' and 'file' resources
    'folder': shutil.rmtree,
    'file': os.unlink,
}

if os.name == 'posix':
    import _multiprocessing

    # Use sem_unlink() to clean up named semaphores.
    #
    # sem_unlink() may be missing if the Python build process detected the
    # absence of POSIX named semaphores. In that case, no named semaphores were
    # ever opened, so no cleanup would be necessary.
    if hasattr(_multiprocessing, 'sem_unlink'):
        _CLEANUP_FUNCS.update(
            {
                'semlock': _multiprocessing.sem_unlink,
            }
        )
# fmt: on

# loky: logging
VERBOSE = False


# loky: compatibility for CPython versions that don't have _RLock._recursion_count
# This was done in CPython 3.13 in
# 'Fix reentrancy issue in multiprocessing resource_tracker'
# https://github.com/python/cpython/pull/109629
# This was back-ported in 3.11.6 and 3.12.1
# TODO Remove work-around when Python 3.13 is our minimum supported version
class LokyRLock(type(threading.RLock())):
    def _recursion_count(self):
        return 1


class ResourceTracker(StdLibResourceTracker):
    """Resource tracker with refcounting scheme.

    This class is an extension of the multiprocessing ResourceTracker class
    which implements a reference counting scheme to avoid unlinking shared
    resources still in use in other processes.

    This feature is notably used by `joblib.Parallel` to share temporary
    folders and memory mapped files between the main process and the worker
    processes.

    The actual implementation of the refcounting scheme is in the main
    function, which is run in a dedicated process.
    """

    def __init__(self):
        super().__init__()
        # Windows process handle returned by CreateProcess.  A pid cannot be
        # passed to os.waitpid on Windows because the underlying _cwait expects
        # a process handle.
        self._proc_handle = None
        # TODO Remove block when Python 3.13 is our minimum supported version
        # see above comment about _recursion_count
        if not hasattr(self._lock, "_recursion_count"):
            self._lock = LokyRLock()

    def maybe_unlink(self, name, rtype):
        """Decrement the refcount of a resource, and delete it if it hits 0"""
        self._send("MAYBE_UNLINK", name, rtype)

    def _teardown_dead_process(self):
        if os.name == "posix":
            super()._teardown_dead_process()
        elif sys.platform == "win32":
            os.close(self._fd)
            if (proc_handle := self._proc_handle) is not None:
                _winapi.CloseHandle(proc_handle)
            # All 3 lines copied from stdlib _teardown_dead_processes
            self._fd = None
            self._pid = None
            self._exitcode = None
            self._proc_handle = None

            warnings.warn(
                "resource_tracker: process died unexpectedly, "
                "relaunching.  Some resources might leak."
            )

    # To minimize the diff with stdlib ResourceTracker._launch
    # fmt: off
    def _launch(self):
        # This is copied from Python 3.14.7 with loky additions/modifications
        # mostly for Windows support and logging.
        # Added or changed lines have a comment that starts with "# loky:"
        fds_to_pass = []
        try:
            fds_to_pass.append(sys.stderr.fileno())
        except Exception:
            pass
        r, w = os.pipe()
        # loky: Windows support
        if sys.platform == "win32":
            _r = duplicate(msvcrt.get_osfhandle(r), inheritable=True)
            os.close(r)
            r = _r

        try:
            fds_to_pass.append(r)
            # process will out live us, so no need to wait on pid
            exe = spawn.get_executable()
            args = [
                exe,
                *util._args_from_interpreter_flags(),
                '-c',
                # loky: use loky main function rather than stdlib one
                f'from {main.__module__} import main; main({r}, {VERBOSE})'
            ]
            # loky: logging
            util.debug(f"launching resource tracker: {args}")
            # bpo-33613: Register a signal mask that will block the signals.
            # This signal mask will be inherited by the child that is going
            # to be spawned and will protect the child from a race condition
            # that can make the child die before it registers signal handlers
            # for SIGINT and SIGTERM. The mask is unregistered after spawning
            # the child.
            prev_sigmask = None
            try:
                if _HAVE_SIGMASK:
                    prev_sigmask = signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
                # loky: call loky spawnv_passfds which supports Windows
                pid, proc_handle = spawnv_passfds(exe, args, fds_to_pass)
            finally:
                if prev_sigmask is not None:
                    signal.pthread_sigmask(signal.SIG_SETMASK, prev_sigmask)
        except:
            os.close(w)
            raise
        else:
            self._fd = w
            self._pid = pid
            self._proc_handle = proc_handle
        finally:
            # loky: Windows support
            if sys.platform == "win32":
                _winapi.CloseHandle(r)
            else:
                os.close(r)
    # fmt: on

    if sys.platform == "win32":
        # stdlib ResourceTracker._stop_locked is POSIX-specific, so we override
        # to be able to use Windows-specific primitives.
        # This is loosely inspired from stdlib ResourceTracker._stop_locked but
        # there are a number of changes. TODO: could the structure be closer?
        def _stop_locked(
            self,
            close=os.close,
            wait_timeout=None,
            wait_for_single_object=_winapi.WaitForSingleObject,
            get_exit_code_process=_winapi.GetExitCodeProcess,
            close_handle=_winapi.CloseHandle,
            winapi_infinite=_winapi.INFINITE,
            wait_timeout_code=_winapi.WAIT_TIMEOUT,
        ):
            # This shouldn't happen (it might when called by a finalizer)
            # so we check for it anyway.
            if self._lock._recursion_count() > 1:
                raise self._reentrant_call_error()
            if self._fd is None:
                # not running
                return
            if self._pid is None:
                return

            # Closing the "alive" file descriptor asks the tracker to stop.
            close(self._fd)
            self._fd = None

            proc_handle = self._proc_handle
            try:
                if proc_handle is not None:
                    timeout_ms = (
                        winapi_infinite
                        if wait_timeout is None
                        else round(wait_timeout * 1000)
                    )
                    wait_result = wait_for_single_object(
                        proc_handle, timeout_ms
                    )
                    if wait_result == wait_timeout_code:
                        self._pid = None
                        self._exitcode = None
                        self._waitpid_timed_out = True
                        return

                    self._pid = None
                    self._exitcode = get_exit_code_process(proc_handle)
                else:
                    self._pid = None
                    self._exitcode = None
            finally:
                if proc_handle is not None:
                    close_handle(proc_handle)
                self._proc_handle = None


# Copied from Python 3.14.7
# fmt: off
_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

# gh-146313: See _after_fork_in_child docstring.
if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_resource_tracker._after_fork_in_child)
# fmt: on

# fmt: off
# The main function has been copied from Python 3.14.7 and modified, mostly for
# Windows support, logging and refcount functionality.
# Added or changed lines have a comment that starts with "# loky:"
# loky: add verbose argument for logging
def main(fd, verbose=0):
    '''Run resource tracker.'''

    # loky: logging
    if verbose:
        util.log_to_stderr(level=util.DEBUG)

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

    # loky: logging
    if verbose:
        util.debug("Main resource tracker is running")

    # loky: change for refcount functionality we want a dict[str, dict] rather
    # than a dict[str, set] so that cache[folder]['resource'] is the refcount
    # associated with it
    cache = {rtype: dict() for rtype in _CLEANUP_FUNCS.keys()}
    exit_code = 0

    try:
        # loky: Windows support
        if sys.platform == "win32":
            fd = msvcrt.open_osfhandle(fd, os.O_RDONLY)
        # keep track of registered/unregistered resources
        with open(fd, 'rb') as f:
            for line in f:
                try:
                    cmd, rtype, name = _decode_message(line)
                    cleanup_func = _CLEANUP_FUNCS.get(rtype, None)
                    if cleanup_func is None:
                        raise ValueError(
                            f'Cannot register {name} for automatic cleanup: '
                            f'unknown resource type ({rtype})'
                            # loky: additional info for possible keys
                            '. Resource type should be one of the following: '
                            f'{list(_CLEANUP_FUNCS.keys())}'
                        )

                    if cmd == 'REGISTER':
                        # loky: refcount functionality
                        if name not in cache[rtype]:
                            cache[rtype][name] = 1
                        else:
                            cache[rtype][name] += 1

                        # loky: logging
                        if verbose:
                            util.debug(
                                '[ResourceTracker] incremented refcount of '
                                f'{rtype} {name} '
                                f'(current {cache[rtype][name]})'
                            )
                    elif cmd == 'UNREGISTER':
                        # loky: refcount functionality
                        del cache[rtype][name]
                        # loky: logging
                        if verbose:
                            util.debug(
                                f'[ResourceTracker] unregister {name} {rtype}: '
                                f'cache({len(cache)})'
                            )
                    elif cmd == 'PROBE':
                        pass
                    # loky: refcount functionality with logging
                    elif cmd == 'MAYBE_UNLINK':
                        cache[rtype][name] -= 1
                        if verbose:
                            util.debug(
                                '[ResourceTracker] decremented refcount of '
                                f'{rtype} {name} '
                                f'(current {cache[rtype][name]})'
                            )

                        if cache[rtype][name] == 0:
                            del cache[rtype][name]
                            try:
                                if verbose:
                                    util.debug(
                                        f'[ResourceTracker] unlink {name}'
                                    )
                                _CLEANUP_FUNCS[rtype](name)
                            except Exception as e:
                                warnings.warn(
                                    f"resource_tracker: {name}: {e!r}"
                                )

                    else:
                        raise RuntimeError('unrecognized command %r' % cmd)
                except Exception:
                    exit_code = 3
                    try:
                        sys.excepthook(*sys.exc_info())
                    except:
                        pass
    finally:
        # all processes have terminated; cleanup any remaining resources

        # loky: loky wants to clean ressources first and folder last because
        # there can be tracked resources inside tracked folders.
        # _unlink_resources is the stdlib code with some additional logging, it
        # is called for all resources except folders and then at the end for
        # all folders
        def _unlink_resources(rtype_cache, rtype):
            nonlocal exit_code
            if rtype_cache:
                try:
                    exit_code = 1
                    if rtype == 'dummy':
                        # The test 'dummy' resource is expected to leak.
                        # We skip the warning (and *only* the warning) for it.
                        pass
                    else:
                        warnings.warn(
                            f'resource_tracker: There appear to be '
                            f'{len(rtype_cache)} leaked {rtype} objects to '
                            f'clean up at shutdown: {rtype_cache}'
                        )
                except Exception:
                    pass
            for name in rtype_cache:
                # For some reason the process which created and registered this
                # resource has failed to unregister it. Presumably it has
                # died.  We therefore unlink it.
                try:
                    try:
                        _CLEANUP_FUNCS[rtype](name)
                        # loky: logging
                        if verbose:
                            util.debug(f'[ResourceTracker] unlink {name}')
                    except Exception as e:
                        exit_code = 2
                        # loky: tweaked formatting (%r instead of %s for
                        # exception and %s instead of %s for name) I guess you
                        # get the exact exception type with %r. %s instead of
                        # %r for name, may be because on Windows %r doubles the
                        # backslashes for path-like ressources
                        warnings.warn('resource_tracker: %s: %r' % (name, e))
                finally:
                    pass

        # loky: 2 stage cleaning process
        # The default cleanup routine for folders deletes everything inside
        # those folders recursively, which can include other resources tracked
        # by the resource tracker). To limit the risk of the resource tracker
        # attempting to delete twice a resource (once as part of a tracked
        # folder, and once as a resource), we delete the folders after all
        # other resource types.
        for rtype, rtype_cache in cache.items():
            if rtype == 'folder':
                continue
            _unlink_resources(rtype_cache, rtype)

        if 'folder' in cache:
            _unlink_resources(cache['folder'], 'folder')

    # loky: logging
    if verbose:
        util.debug("resource tracker shut down")

    # TODO not sure about exit_code management with 2-stage clean-up
    # (first non-folder resources then folder resources) but seems good enough
    # for now. We did not have any exit code management until
    # https://github.com/joblib/loky/pull/472
    sys.exit(exit_code)
# fmt: on


def spawnv_passfds(path, args, passfds):
    """Loky version multiprocessing.util.spawnv_passfds with added Windows support.

    Returns (pid, process_handle) because os.waitpid needs handle on Windows.
    On Linux process_handle is None.
    """
    if sys.platform != "win32":
        # loky: additional encoding needed here
        # TODO We should fix loky.backend.spawn.get_executable to return bytes
        # on POSIX so that this can be removed
        path = path.encode("utf-8")
        return util.spawnv_passfds(path, args, passfds), None
    else:
        # loky: Windows support
        passfds = sorted(passfds)
        cmd = " ".join(f'"{x}"' for x in args)
        hp, ht, pid, _ = _winapi.CreateProcess(
            path, cmd, None, None, True, 0, None, None, None
        )
        _winapi.CloseHandle(ht)
        # Keep the process handle for a safe wait during teardown.  Pids and
        # handles are different namespaces on Windows.
        return pid, hp
