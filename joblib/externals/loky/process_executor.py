###############################################################################
# Re-implementation of the ProcessPoolExecutor to robustify its fault tolerance
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from concurrent/futures/process_pool_executor.py (17/02/2017)
#  * Backport for python2.7/3.3,
#  * Add an extra management thread to detect queue_management_thread failures,
#  * Improve the shutdown process to avoid deadlocks,
#  * Add timeout for workers,
#  * More robust pickling process.
#
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""


__author__ = 'Thomas Moreau (thomas.moreau.2010@gmail.com)'


import os
import sys
import types
import weakref
import warnings
import itertools
import traceback
import threading
import multiprocessing as mp
from functools import partial

from . import _base
from .backend import get_context
from .backend.compat import queue
from .backend.compat import wait
from .backend.context import cpu_count
from .backend.queues import Queue, SimpleQueue, Full

try:
    from concurrent.futures.process import BrokenProcessPool as _BPPException
except ImportError:
    _BPPException = RuntimeError


# Compatibility for python2.7
if sys.version_info[0] == 2:
    ProcessLookupError = OSError


# Workers are created as daemon threads and processes. This is done to allow
# the interpreter to exit when there are still idle processes in a
# ProcessPoolExecutor's process pool (i.e. shutdown() was not called). However,
# allowing workers to die with the interpreter has two undesirable properties:
#   - The workers would still be running during interpreter shutdown,
#     meaning that they would fail in unpredictable ways.
#   - The workers could be killed while evaluating a work item, which could
#     be bad if the callable being evaluated has external side-effects e.g.
#     writing to a file.
#
# To work around this problem, an exit handler is installed which tells the
# workers to exit when their work queues are empty and then waits until the
# threads/processes finish.

_threads_wakeups = weakref.WeakKeyDictionary()
_global_shutdown = False

# Mechanism to prevent infinite process spawning. When a worker of a
# ProcessPoolExecutor nested in MAX_DEPTH Executor tries to create a new
# Executor, a LokyRecursionError is raised
MAX_DEPTH = int(os.environ.get("LOKY_MAX_DEPTH", 10))
_CURRENT_DEPTH = 0


class _ThreadWakeup:
    def __init__(self):
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        self._writer.close()
        self._reader.close()

    def wakeup(self):
        if sys.platform == "win32" and sys.version_info[:2] < (3, 4):
            # Compat for python2.7 on windows, where poll return false for
            # b"" messages. Use the slightly larger message b"0".
            self._writer.send_bytes(b"0")
        else:
            self._writer.send_bytes(b"")

    def clear(self):
        while self._reader.poll():
            self._reader.recv_bytes()


class _ExecutorFlags(object):
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by queue_management_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """
    def __init__(self):

        self.shutdown = False
        self.broken = None
        self.kill_workers = False
        self.shutdown_lock = threading.Lock()

    def flag_as_shutting_down(self, kill_workers=False):
        with self.shutdown_lock:
            self.shutdown = True
            self.kill_workers = kill_workers

    def flag_as_broken(self, broken):
        with self.shutdown_lock:
            self.shutdown = True
            self.broken = broken


def _python_exit():
    global _global_shutdown
    _global_shutdown = True
    items = list(_threads_wakeups.items())
    mp.util.debug("Interpreter shutting down. Waking up queue_manager_threads "
                  "{}".format(items))
    for thread, thread_wakeup in items:
        if thread.is_alive():
            thread_wakeup.wakeup()
    for thread, _ in items:
        thread.join()


# Module variable to register the at_exit call
process_pool_executor_at_exit = None

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1


class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback
    """
    def __init__(self, tb=None):
        self.tb = tb

    def __str__(self):
        return self.tb


class _ExceptionWithTraceback(BaseException):

    def __init__(self, exc, tb=None):
        if tb is None:
            _, _, tb = sys.exc_info()
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = '\n"""\n%s"""' % tb

    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)


def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc


class _WorkItem(object):

    __slots__ = ["future", "fn", "args", "kwargs"]

    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _ResultItem(object):

    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result


class _CallItem(object):

    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return "CallItem({}, {}, {}, {})".format(
            self.work_id, self.fn, self.args, self.kwargs)

    try:
        # If cloudpickle is present on the system, use it to pickle the
        # function. This permits to use interactive terminal for loky calls.
        # TODO: Add option to deactivate, as it increases pickling time.
        from .backend import LOKY_PICKLER
        assert LOKY_PICKLER is None or LOKY_PICKLER == ""

        import cloudpickle  # noqa: F401

        def __getstate__(self):
            from cloudpickle import dumps
            if isinstance(self.fn, (types.FunctionType,
                                    types.LambdaType,
                                    partial)):
                cp = True
                fn = dumps(self.fn)
            else:
                cp = False
                fn = self.fn
            return (self.work_id, self.args, self.kwargs, fn, cp)

        def __setstate__(self, state):
            self.work_id, self.args, self.kwargs, self.fn, cp = state
            if cp:
                from cloudpickle import loads
                self.fn = loads(self.fn)

    except (ImportError, AssertionError) as e:
        pass


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""
    def __init__(self, max_size=0, ctx=None, pending_work_items=None,
                 running_work_items=None, thread_wakeup=None, reducers=None):
        self.thread_wakeup = thread_wakeup
        self.pending_work_items = pending_work_items
        self.running_work_items = running_work_items
        super(_SafeQueue, self).__init__(max_size, reducers=reducers, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            # fromat traceback only on python3
            tb = traceback.format_exception(
                type(e), e, getattr(e, "__traceback__", None))
            e.__cause__ = _RemoteTraceback('\n"""\n{}"""'.format(''.join(tb)))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            self.running_work_items.remove(obj.work_id)
            # work_item can be None if another process terminated. In this
            # case, the queue_manager_thread fails all work_items with
            # BrokenProcessPool
            if work_item is not None:
                work_item.future.set_exception(e)
                del work_item
            self.thread_wakeup.wakeup()
        else:
            super()._on_queue_feeder_error(e, obj)


def _get_chunks(chunksize, *iterables):
    """ Iterates over zip()ed iterables in chunks. """
    if sys.version_info < (3, 3):
        it = itertools.izip(*iterables)
    else:
        it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(_ResultItem(work_id, result=result,
                                     exception=exception))
    except BaseException as e:
        exc = _ExceptionWithTraceback(e, getattr(e, "__traceback__", None))
        result_queue.put(_ResultItem(work_id, exception=exc))


def _process_worker(call_queue, result_queue, initializer, initargs,
                    processes_management_lock, timeout, worker_exit_lock,
                    current_depth):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
        process_management_lock: A ctx.Lock avoiding worker timeout while some
            workers are being spawned.
        timeout: maximum time to wait for a new item in the call_queue. If that
            time is expired, the worker will shutdown.
        worker_exit_lock: Lock to avoid flagging the executor as broken on
            workers timeout.
        current_depth: Nested parallelism level, to avoid infinite spawning.
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            _base.LOGGER.critical('Exception in initializer:', exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return

    # set the global _CURRENT_DEPTH mechanism to limit recursive call
    global _CURRENT_DEPTH
    _CURRENT_DEPTH = current_depth

    mp.util.debug('worker started with timeout=%s' % timeout)
    while True:
        try:
            call_item = call_queue.get(block=True, timeout=timeout)
            if call_item is None:
                mp.util.info("shutting down worker on sentinel")
        except queue.Empty:
            mp.util.info("shutting down worker after timeout %0.3fs"
                         % timeout)
            if processes_management_lock.acquire(block=False):
                processes_management_lock.release()
                call_item = None
            else:
                mp.util.info("Could not acquire processes_management_lock")
                continue
        except BaseException as e:
            traceback.print_exc()
            sys.exit(1)
        if call_item is None:
            # Notify queue management thread about clean worker shutdown
            result_queue.put(os.getpid())
            with worker_exit_lock:
                return
        try:
            r = call_item.fn(*call_item.args, **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, getattr(e, "__traceback__", None))
            result_queue.put(_ResultItem(call_item.work_id, exception=exc))
        else:
            _sendback_result(result_queue, call_item.work_id, result=r)

        # Liberate the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item


def _add_call_item_to_queue(pending_work_items,
                            running_work_items,
                            work_ids,
                            call_queue):
    """Fills call_queue with _WorkItems from pending_work_items.

    This function never blocks.

    Args:
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids: A queue.Queue of work ids e.g. Queue([5, 6, ...]). Work ids
            are consumed and the corresponding _WorkItems from
            pending_work_items are transformed into _CallItems and put in
            call_queue.
        call_queue: A ctx.Queue that will be filled with _CallItems
            derived from _WorkItems.
    """
    while True:
        if call_queue.full():
            return
        try:
            work_id = work_ids.get(block=False)
        except queue.Empty:
            return
        else:
            work_item = pending_work_items[work_id]

            if work_item.future.set_running_or_notify_cancel():
                running_work_items += [work_id]
                call_queue.put(_CallItem(work_id,
                                         work_item.fn,
                                         work_item.args,
                                         work_item.kwargs),
                               block=True)
            else:
                del pending_work_items[work_id]
                continue


def _queue_management_worker(executor_reference,
                             executor_flags,
                             processes,
                             pending_work_items,
                             running_work_items,
                             work_ids_queue,
                             call_queue,
                             result_queue,
                             thread_wakeup,
                             processes_management_lock):
    """Manages the communication between this process and the worker processes.

    This function is run in a local thread.

    Args:
        executor_reference: A weakref.ref to the ProcessPoolExecutor that owns
            this thread. Used to determine if the ProcessPoolExecutor has been
            garbage collected and that this function can exit.
        executor_flags: A ExecutorFlags holding internal states of the
            ProcessPoolExecutor. It permits to know if the executor is broken
            even the object has been gc.
        process: A list of the ctx.Process instances used as
            workers.
        pending_work_items: A dict mapping work ids to _WorkItems e.g.
            {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        work_ids_queue: A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        call_queue: A ctx.Queue that will be filled with _CallItems
            derived from _WorkItems for processing by the process workers.
        result_queue: A ctx.SimpleQueue of _ResultItems generated by the
            process workers.
        thread_wakeup: A _ThreadWakeup to allow waking up the
            queue_manager_thread from the main Thread and avoid deadlocks
            caused by permanently locked queues.
    """
    executor = None

    def is_shutting_down():
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that own this worker is not broken AND
        #        * The executor that owns this worker has been collected OR
        #        * The executor that owns this worker has been shutdown.
        # If the executor is broken, it should be detected in the next loop.
        return (_global_shutdown or
                ((executor is None or executor_flags.shutdown)
                 and not executor_flags.broken))

    def shutdown_all_workers():
        mp.util.debug("queue management thread shutting down")
        executor_flags.flag_as_shutting_down()
        # Create a list to avoid RuntimeError due to concurrent modification of
        # processes. nb_children_alive is thus an upper bound. Also release the
        # processes' safe_guard_locks to accelerate the shutdown procedure, as
        # there is no need for hand-shake here.
        with processes_management_lock:
            n_children_alive = 0
            for p in list(processes.values()):
                p._worker_exit_lock.release()
                n_children_alive += 1
        n_children_to_stop = n_children_alive
        n_sentinels_sent = 0
        # Send the right number of sentinels, to make sure all children are
        # properly terminated.
        while n_sentinels_sent < n_children_to_stop and n_children_alive > 0:
            for i in range(n_children_to_stop - n_sentinels_sent):
                try:
                    call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except Full:
                    break
            with processes_management_lock:
                n_children_alive = sum(
                    p.is_alive() for p in list(processes.values())
                )

        # Release the queue's resources as soon as possible. Flag the feeder
        # thread for clean exit to avoid having the crash detection thread flag
        # the Executor as broken during the shutdown. This is safe as either:
        #  * We don't need to communicate with the workers anymore
        #  * There is nothing left in the Queue buffer except None sentinels
        mp.util.debug("closing call_queue")
        call_queue.close()

        mp.util.debug("joining processes")
        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on Mac OS X.
        while processes:
            _, p = processes.popitem()
            p.join()
        mp.util.debug("queue management thread clean shutdown of worker "
                      "processes: {}".format(list(processes)))

    result_reader = result_queue._reader
    wakeup_reader = thread_wakeup._reader
    readers = [result_reader, wakeup_reader]

    while True:
        _add_call_item_to_queue(pending_work_items,
                                running_work_items,
                                work_ids_queue,
                                call_queue)
        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        worker_sentinels = [p.sentinel for p in processes.values()]
        ready = wait(readers + worker_sentinels)

        broken = ("A process in the executor was terminated abruptly", None)
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                broken = None
            except BaseException as e:
                tb = getattr(e, "__traceback__", None)
                if tb is None:
                    _, _, tb = sys.exc_info()
                broken = ("A result has failed to un-serialize",
                          traceback.format_exception(type(e), e, tb))
        elif wakeup_reader in ready:
            broken = None
            result_item = None
        thread_wakeup.clear()
        if broken:
            msg, cause = broken
            # Mark the process pool broken so that submits fail right now.
            executor_flags.flag_as_broken(
                msg + ", the pool is not usable anymore.")
            bpe = BrokenProcessPool(
                msg + " while the future was running or pending.")
            if cause is not None:
                bpe.__cause__ = _RemoteTraceback(
                    "\n'''\n{}'''".format(''.join(cause)))

            # All futures in flight must be marked failed
            for work_id, work_item in pending_work_items.items():
                work_item.future.set_exception(bpe)
                # Delete references to object. See issue16284
                del work_item
            pending_work_items.clear()

            # Terminate remaining workers forcibly: the queues or their
            # locks may be in a dirty state and block forever.
            while processes:
                _, p = processes.popitem()
                mp.util.debug('terminate process {}'.format(p.name))
                try:
                    p.terminate()
                    p.join()
                except ProcessLookupError:  # pragma: no cover
                    pass

            shutdown_all_workers()
            return
        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID, either on request
            # by the executor.shutdown method or by the timeout of the worker
            # itself: we should not mark the executor as broken.
            with processes_management_lock:
                p = processes.pop(result_item, None)

            # p can be None is the executor is concurrently shutting down.
            if p is not None:
                p._worker_exit_lock.release()
                p.join()
                del p

            # Make sure the executor have the right number of worker, even if a
            # worker timeout while some jobs were submitted. If some work is
            # pending or there is less processes than running items, we need to
            # start a new Process and raise a warning.
            n_pending = len(pending_work_items)
            n_running = len(running_work_items)
            if (n_pending - n_running > 0 or n_running > len(processes)):
                executor = executor_reference()
                if (executor is not None
                        and len(processes) < executor._max_workers):
                    warnings.warn(
                        "A worker timeout while some jobs were given to the "
                        "executor. You might want to use a longer timeout for "
                        "the executor.", UserWarning
                    )
                    executor._adjust_process_count()
                    executor = None

        elif result_item is not None:
            work_item = pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                # Delete references to object. See issue16284
                del work_item
                running_work_items.remove(result_item.work_id)
            # Delete reference to result_item
            del result_item

        # Check whether we should start shutting down.
        executor = executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this worker has been collected OR
        #   - The executor that owns this worker has been shutdown.
        if is_shutting_down():
            # bpo-33097: Make sure that the executor is flagged as shutting
            # down even if it is shutdown by the interpreter exiting.
            with executor_flags.shutdown_lock:
                executor_flags.shutdown = True
            if executor_flags.kill_workers:
                while pending_work_items:
                    _, work_item = pending_work_items.popitem()
                    work_item.future.set_exception(ShutdownExecutorError(
                        "The Executor was shutdown before this job could "
                        "complete."))
                    del work_item
                # Terminate remaining workers forcibly: the queues or their
                # locks may be in a dirty state and block forever.
                while processes:
                    _, p = processes.popitem()
                    p.terminate()
                    p.join()
                shutdown_all_workers()
                return
            # Since no new work items can be added, it is safe to shutdown
            # this thread if there are no pending work items.
            if not pending_work_items:
                shutdown_all_workers()
                return
        elif executor_flags.broken:
            return
        executor = None


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked:
        if _system_limited:
            raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # undetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = ("system provides too few semaphores (%d available, "
                       "256 necessary)" % nsems_max)
    raise NotImplementedError(_system_limited)


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


def _check_max_depth(context):
    # Limit the maxmal recursion level
    global _CURRENT_DEPTH
    if context.get_start_method() == "fork" and _CURRENT_DEPTH > 0:
        raise LokyRecursionError(
            "Could not spawn extra nested processes at depth superior to "
            "MAX_DEPTH=1. It is not possible to increase this limit when "
            "using the 'fork' start method.")

    if 0 < MAX_DEPTH and _CURRENT_DEPTH + 1 > MAX_DEPTH:
        raise LokyRecursionError(
            "Could not spawn extra nested processes at depth superior to "
            "MAX_DEPTH={}. If this is intendend, you can change this limit "
            "with the LOKY_MAX_DEPTH environment variable.".format(MAX_DEPTH))


class LokyRecursionError(RuntimeError):
    """Raised when a process try to spawn too many levels of nested processes.
    """


class BrokenProcessPool(_BPPException):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


# Alias for backward compat (for code written for loky 1.1.4 and earlier). Do
# not use in new code.
BrokenExecutor = BrokenProcessPool


class ShutdownExecutorError(RuntimeError):

    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """


class ProcessPoolExecutor(_base.Executor):

    _at_exit = None

    def __init__(self, max_workers=None, job_reducers=None,
                 result_reducers=None, timeout=None, context=None,
                 initializer=None, initargs=()):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: int, optional (default: cpu_count())
                The maximum number of processes that can be used to execute the
                given calls. If None or not given then as many worker processes
                will be created as the number of CPUs the current process
                can use.
            job_reducers, result_reducers: dict(type: reducer_func)
                Custom reducer for pickling the jobs and the results from the
                Executor. If only `job_reducers` is provided, `result_reducer`
                will use the same reducers
            timeout: int, optional (default: None)
                Idle workers exit after timeout seconds. If a new job is
                submitted after the timeout, the executor will start enough
                new Python processes to make sure the pool of workers is full.
            context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: An callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            self._max_workers = max_workers

        if context is None:
            context = get_context()
        self._context = context

        if initializer is not None and not callable(initializer):
            raise TypeError("initializer must be a callable")
        self._initializer = initializer
        self._initargs = initargs

        _check_max_depth(self._context)

        if result_reducers is None:
            result_reducers = job_reducers

        # Timeout
        self._timeout = timeout

        # Internal variables of the ProcessPoolExecutor
        self._processes = {}
        self._queue_count = 0
        self._pending_work_items = {}
        self._running_work_items = []
        self._work_ids = queue.Queue()
        self._processes_management_lock = self._context.Lock()
        self._queue_management_thread = None

        # _ThreadWakeup is a communication channel used to interrupt the wait
        # of the main loop of queue_manager_thread from another thread (e.g.
        # when calling executor.submit or executor.shutdown). We do not use the
        # _result_queue to send the wakeup signal to the queue_manager_thread
        # as it could result in a deadlock if a worker process dies with the
        # _result_queue write lock still acquired.
        self._queue_management_thread_wakeup = _ThreadWakeup()

        # Flag to hold the state of the Executor. This permits to introspect
        # the Executor state even once it has been garbage collected.
        self._flags = _ExecutorFlags()

        # Finally setup the queues for interprocess communication
        self._setup_queues(job_reducers, result_reducers)

        mp.util.debug('ProcessPoolExecutor is setup')

    def _setup_queues(self, job_reducers, result_reducers, queue_size=None):
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        if queue_size is None:
            queue_size = 2 * self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(
            max_size=queue_size, pending_work_items=self._pending_work_items,
            running_work_items=self._running_work_items,
            thread_wakeup=self._queue_management_thread_wakeup,
            reducers=job_reducers, ctx=self._context)
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True

        self._result_queue = SimpleQueue(reducers=result_reducers,
                                         ctx=self._context)

    def _start_queue_management_thread(self):
        if self._queue_management_thread is None:
            mp.util.debug('_start_queue_management_thread called')

            # When the executor gets garbarge collected, the weakref callback
            # will wake up the queue management thread so that it can terminate
            # if there is no pending work item.
            def weakref_cb(_,
                           thread_wakeup=self._queue_management_thread_wakeup):
                mp.util.debug('Executor collected: triggering callback for'
                              ' QueueManager wakeup')
                thread_wakeup.wakeup()

            # Start the processes so that their sentinels are known.
            self._queue_management_thread = threading.Thread(
                target=_queue_management_worker,
                args=(weakref.ref(self, weakref_cb),
                      self._flags,
                      self._processes,
                      self._pending_work_items,
                      self._running_work_items,
                      self._work_ids,
                      self._call_queue,
                      self._result_queue,
                      self._queue_management_thread_wakeup,
                      self._processes_management_lock),
                name="QueueManagerThread")
            self._queue_management_thread.daemon = True
            self._queue_management_thread.start()

            # register this executor in a mechanism that ensures it will wakeup
            # when the interpreter is exiting.
            _threads_wakeups[self._queue_management_thread] = \
                self._queue_management_thread_wakeup

            global process_pool_executor_at_exit
            if process_pool_executor_at_exit is None:
                # Ensure that the _python_exit function will be called before
                # the multiprocessing.Queue._close finalizers which have an
                # exitpriority of 10.
                process_pool_executor_at_exit = mp.util.Finalize(
                    None, _python_exit, exitpriority=20)

    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            worker_exit_lock = self._context.BoundedSemaphore(1)
            worker_exit_lock.acquire()
            p = self._context.Process(
                target=_process_worker,
                args=(self._call_queue,
                      self._result_queue,
                      self._initializer,
                      self._initargs,
                      self._processes_management_lock,
                      self._timeout,
                      worker_exit_lock,
                      _CURRENT_DEPTH + 1))
            p._worker_exit_lock = worker_exit_lock
            p.start()
            self._processes[p.pid] = p
        mp.util.debug('Adjust process count : {}'.format(self._processes))

    def _ensure_executor_running(self):
        """ensures all workers and management thread are running
        """
        with self._processes_management_lock:
            if len(self._processes) != self._max_workers:
                self._adjust_process_count()
            self._start_queue_management_thread()

    def submit(self, fn, *args, **kwargs):
        with self._flags.shutdown_lock:
            if self._flags.broken:
                raise BrokenProcessPool(self._flags.broken)
            if self._flags.shutdown:
                raise ShutdownExecutorError(
                    'cannot schedule new futures after shutdown')

            # Cannot submit a new calls once the interpreter is shutting down.
            # This check avoids spawning new processes at exit.
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after '
                                   'interpreter shutdown')

            f = _base.Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._queue_management_thread_wakeup.wakeup()

            self._ensure_executor_running()
            return f
    submit.__doc__ = _base.Executor.submit.__doc__

    def map(self, fn, *iterables, **kwargs):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        timeout = kwargs.get('timeout', None)
        chunksize = kwargs.get('chunksize', 1)
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super(ProcessPoolExecutor, self).map(
            partial(_process_chunk, fn), _get_chunks(chunksize, *iterables),
            timeout=timeout)
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True, kill_workers=False):
        mp.util.debug('shutting down executor %s' % self)

        self._flags.flag_as_shutting_down(kill_workers)
        qmt = self._queue_management_thread
        qmtw = self._queue_management_thread_wakeup
        if qmt:
            self._queue_management_thread = None
            if qmtw:
                self._queue_management_thread_wakeup = None
            # Wake up queue management thread
            if qmtw is not None:
                try:
                    qmtw.wakeup()
                except OSError:
                    # Can happen in case of concurrent calls to shutdown.
                    pass
            if wait:
                qmt.join()

        cq = self._call_queue
        if cq:
            self._call_queue = None
            cq.close()
            if wait:
                cq.join_thread()
        self._result_queue = None
        self._processes_management_lock = None

        if qmtw:
            try:
                qmtw.close()
            except OSError:
                # Can happen in case of concurrent calls to shutdown.
                pass
    shutdown.__doc__ = _base.Executor.shutdown.__doc__
