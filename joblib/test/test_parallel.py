"""
Test the parallel module.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2010-2011 Gael Varoquaux
# License: BSD Style, 3 clauses.

import time
import sys
import io
import os
from math import sqrt

from joblib import parallel

from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.testing import check_subprocess_call
from joblib._compat import PY3_OR_LATER
from multiprocessing import TimeoutError
from time import sleep

try:
    import cPickle as pickle
    PickleError = TypeError
except:
    import pickle
    PickleError = pickle.PicklingError


if PY3_OR_LATER:
    PickleError = pickle.PicklingError

try:
    # Python 2/Python 3 compat
    unicode('str')
except NameError:
    unicode = lambda s: s

try:
    from queue import Queue
except ImportError:
    # Backward compat
    from Queue import Queue

try:
    import posix
except ImportError:
    posix = None

from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import SafeFunction
from joblib._parallel_backends import WorkerInterrupt

from joblib.parallel import Parallel, delayed
from joblib.parallel import register_parallel_backend, parallel_backend

from joblib.parallel import mp, cpu_count, BACKENDS, effective_n_jobs
from joblib.my_exceptions import JoblibException

import nose
from nose.tools import assert_equal, assert_true, assert_false, assert_raises


ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())

if hasattr(mp, 'get_context'):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))


def division(x, y):
    return x / y


def square(x):
    return x ** 2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__
    """
    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    if x == 7:
        raise (MyExceptionWithFinickyInit('a', 'b', 'c', 'd')
               if custom_exception else ValueError)
    return x


def interrupt_raiser(x):
    time.sleep(.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x ** 2 + y + z


def _active_backend_type():
    return type(parallel.get_active_backend()[0])


###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


def test_effective_n_jobs():
    assert effective_n_jobs() > 0


###############################################################################
# Test parallel
def check_simple_parallel(backend):
    X = range(5)
    for n_jobs in (1, 2, -1, -2):
        assert_equal(
            [square(x) for x in X],
            Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(square)(x) for x in X))
    try:
        # To smoke-test verbosity, we capture stdout
        orig_stdout = sys.stdout
        orig_stderr = sys.stdout
        if PY3_OR_LATER:
            sys.stderr = io.StringIO()
            sys.stderr = io.StringIO()
        else:
            sys.stdout = io.BytesIO()
            sys.stderr = io.BytesIO()
        for verbose in (2, 11, 100):
            Parallel(n_jobs=-1, verbose=verbose, backend=backend)(
                delayed(square)(x) for x in X)
            Parallel(n_jobs=1, verbose=verbose, backend=backend)(
                delayed(square)(x) for x in X)
            Parallel(n_jobs=2, verbose=verbose, pre_dispatch=2,
                     backend=backend)(
                delayed(square)(x) for x in X)
            Parallel(n_jobs=2, verbose=verbose, backend=backend)(
                delayed(square)(x) for x in X)
    except Exception as e:
        my_stdout = sys.stdout
        my_stderr = sys.stderr
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        print(unicode(my_stdout.getvalue()))
        print(unicode(my_stderr.getvalue()))
        raise e
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def test_simple_parallel():
    for backend in ALL_VALID_BACKENDS:
        yield check_simple_parallel, backend


def nested_loop(backend):
    Parallel(n_jobs=2, backend=backend)(
        delayed(square)(.01) for _ in range(2))


def check_nested_loop(parent_backend, child_backend):
    Parallel(n_jobs=2, backend=parent_backend)(
        delayed(nested_loop)(child_backend) for _ in range(2))


def test_nested_loop():
    for parent_backend in BACKENDS:
        for child_backend in BACKENDS:
            yield check_nested_loop, parent_backend, child_backend


def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    q = Queue(maxsize=5)
    Parallel(n_jobs=2, backend="threading")(
        delayed(q.put, check_pickle=False)(1) for _ in range(5))
    assert_true(q.full())


def test_parallel_kwargs():
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    for n_jobs in (1, 4):
        yield (assert_equal,
               [f(x, y=1) for x in lst],
               Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst))


def check_parallel_as_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]
    with Parallel(n_jobs=4, backend=backend) as p:
        # Internally a pool instance has been eagerly created and is managed
        # via the context manager protocol
        managed_backend = p._backend
        if mp is not None:
            assert_true(managed_backend is not None)
            assert_true(managed_backend._pool is not None)

        # We make call with the managed parallel object several times inside
        # the managed block:
        assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))
        assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))

        # Those calls have all used the same pool instance:
        if mp is not None:
            assert_true(managed_backend._pool is p._backend._pool)

    # As soon as we exit the context manager block, the pool is terminated and
    # no longer referenced from the parallel object:
    if mp is not None:
        assert_true(p._backend._pool is None)

    # It's still possible to use the parallel instance in non-managed mode:
    assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))
    if mp is not None:
        assert_true(p._backend._pool is None)


def test_parallel_context_manager():
    for backend in ['multiprocessing', 'threading']:
        yield check_parallel_as_context_manager, backend


def test_parallel_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    def g(x):
        return x ** 2

    try:
        # pickling a local function always fail but the exception
        # raised is a PickleError for python <= 3.4 and AttributeError
        # for python >= 3.5
        pickle.dumps(g)
    except Exception as exc:
        exception_class = exc.__class__

    assert_raises(exception_class, Parallel(),
                  (delayed(g)(x) for x in range(10)))


def test_parallel_timeout_success():
    # Check that timeout isn't thrown when function is fast enough
    for backend in ['multiprocessing', 'threading']:
        nose.tools.assert_equal(
            10,
            len(Parallel(n_jobs=2, backend=backend, timeout=10)
                (delayed(sleep)(0.001) for x in range(10))))


@with_multiprocessing
def test_parallel_timeout_fail():
    # Check that timeout properly fails when function is too slow
    for backend in ['multiprocessing', 'threading']:
        nose.tools.assert_raises(
            TimeoutError,
            Parallel(n_jobs=2, backend=backend, timeout=0.01),
            (delayed(sleep)(10) for x in range(10))
        )


def test_error_capture():
    # Check that error are captured, and that correct exceptions
    # are raised.
    if mp is not None:
        # A JoblibException will be raised only if there is indeed
        # multiprocessing
        assert_raises(JoblibException, Parallel(n_jobs=2),
                      [delayed(division)(x, y)
                       for x, y in zip((0, 1), (1, 0))])
        assert_raises(WorkerInterrupt, Parallel(n_jobs=2),
                      [delayed(interrupt_raiser)(x) for x in (1, 0)])

        # Try again with the context manager API
        with Parallel(n_jobs=2) as parallel:
            assert_true(parallel._backend._pool is not None)
            original_pool = parallel._backend._pool

            assert_raises(JoblibException, parallel,
                          [delayed(division)(x, y)
                           for x, y in zip((0, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert_true(parallel._backend._pool is not None)

            # The pool should have been interrupted and restarted:
            assert_true(parallel._backend._pool is not original_pool)

            assert_equal([f(x, y=1) for x in range(10)],
                         parallel(delayed(f)(x, y=1) for x in range(10)))

            original_pool = parallel._backend._pool
            assert_raises(WorkerInterrupt, parallel,
                          [delayed(interrupt_raiser)(x) for x in (1, 0)])

            # The pool should still be available despite the exception
            assert_true(parallel._backend._pool is not None)

            # The pool should have been interrupted and restarted:
            assert_true(parallel._backend._pool is not original_pool)

            assert_equal([f(x, y=1) for x in range(10)],
                         parallel(delayed(f)(x, y=1) for x in range(10)))

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert_true(parallel._backend._pool is None)
    else:
        assert_raises(KeyboardInterrupt, Parallel(n_jobs=2),
                      [delayed(interrupt_raiser)(x) for x in (1, 0)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    assert_raises(ZeroDivisionError, Parallel(n_jobs=2),
                  [delayed(division)(x, y) for x, y in zip((0, 1), (1, 0))])

    assert_raises(
        MyExceptionWithFinickyInit,
        Parallel(n_jobs=2, verbose=0),
        (delayed(exception_raiser)(i, custom_exception=True)
         for i in range(30)))

    try:
        # JoblibException wrapping is disabled in sequential mode:
        ex = JoblibException()
        Parallel(n_jobs=1)(
            delayed(division)(x, y) for x, y in zip((0, 1), (1, 0)))
    except Exception as ex:
        assert_false(isinstance(ex, JoblibException))


class Counter(object):
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2

    def __call__(self, i):
        self.list1.append(i)
        assert_equal(len(self.list1), len(self.list2))


def consumer(queue, item):
    queue.append('Consumed %s' % item)


def check_dispatch_one_job(backend):
    """ Test that with only one job, Parallel does act as a iterator.
    """
    queue = list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    # disable batching
    Parallel(n_jobs=1, batch_size=1, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    assert_equal(queue, [
        'Produced 0', 'Consumed 0',
        'Produced 1', 'Consumed 1',
        'Produced 2', 'Consumed 2',
        'Produced 3', 'Consumed 3',
        'Produced 4', 'Consumed 4',
        'Produced 5', 'Consumed 5',
    ])
    assert_equal(len(queue), 12)

    # empty the queue for the next check
    queue[:] = []

    # enable batching
    Parallel(n_jobs=1, batch_size=4, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    assert_equal(queue, [
        # First batch
        'Produced 0', 'Produced 1', 'Produced 2', 'Produced 3',
        'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3',

        # Second batch
        'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5',
    ])
    assert_equal(len(queue), 12)


def test_dispatch_one_job():
    for backend in BACKENDS:
        yield check_dispatch_one_job, backend


def check_dispatch_multiprocessing(backend):
    """ Check that using pre_dispatch Parallel does indeed dispatch items
        lazily.
    """
    if mp is None:
        raise nose.SkipTest()
    manager = mp.Manager()
    queue = manager.list()

    def producer():
        for i in range(6):
            queue.append('Produced %i' % i)
            yield i

    Parallel(n_jobs=2, batch_size=1, pre_dispatch=3, backend=backend)(
        delayed(consumer)(queue, 'any') for _ in producer())

    # Only 3 tasks are dispatched out of 6. The 4th task is dispatched only
    # after any of the first 3 jobs have completed.
    first_four = list(queue)[:4]
    # The the first consumption event can sometimes happen before the end of
    # the dispatching, hence, pop it before introspecting the "Produced" events
    first_four.remove('Consumed any')
    assert_equal(first_four,
                 ['Produced 0', 'Produced 1', 'Produced 2'])
    assert_equal(len(queue), 12)


def test_dispatch_multiprocessing():
    for backend in BACKENDS:
        if backend != "sequential":
            yield check_dispatch_multiprocessing, backend


def test_batching_auto_threading():
    # batching='auto' with the threading backend leaves the effective batch
    # size to 1 (no batching) as it has been found to never be beneficial with
    # this low-overhead backend.

    with Parallel(n_jobs=2, batch_size='auto', backend='threading') as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks
        assert_equal(p._backend.compute_batch_size(), 1)


def test_batching_auto_multiprocessing():
    with Parallel(n_jobs=2, batch_size='auto', backend='multiprocessing') as p:
        p(delayed(id)(i) for i in range(5000))  # many very fast tasks

        # It should be strictly larger than 1 but as we don't want heisen
        # failures on clogged CI worker environment be safe and only check that
        # it's a strictly positive number.
        assert_true(p._backend.compute_batch_size() > 0)


def test_exception_dispatch():
    "Make sure that exception raised during dispatch are indeed captured"
    assert_raises(
        ValueError,
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0),
        (delayed(exception_raiser)(i) for i in range(30)))


def test_nested_exception_dispatch():
    # Ensure TransportableException objects for nested joblib cases gets
    # propagated.
    assert_raises(
        JoblibException,
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0),
        (delayed(SafeFunction(exception_raiser))(i) for i in range(30)))


def _reload_joblib():
    # Retrieve the path of the parallel module in a robust way
    joblib_path = Parallel.__module__.split(os.sep)
    joblib_path = joblib_path[:1]
    joblib_path.append('parallel.py')
    joblib_path = '/'.join(joblib_path)
    module = __import__(joblib_path)
    # Reload the module. This should trigger a fail
    reload(module)


def test_multiple_spawning():
    # Test that attempting to launch a new Python after spawned
    # subprocesses will raise an error, to avoid infinite loops on
    # systems that do not support fork
    if not int(os.environ.get('JOBLIB_MULTIPROCESSING', 1)):
        raise nose.SkipTest()
    assert_raises(ImportError, Parallel(n_jobs=2, pre_dispatch='all'),
                  [delayed(_reload_joblib)() for i in range(10)])


class FakeParallelBackend(SequentialBackend):
    """Pretends to run concurrently while running sequentially."""

    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.n_jobs = self.effective_n_jobs(n_jobs)
        self.parallel = parallel
        return n_jobs

    def effective_n_jobs(self, n_jobs=1):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        return n_jobs


def test_invalid_backend():
    assert_raises(ValueError, Parallel, backend='unit-testing')


def test_register_parallel_backend():
    try:
        register_parallel_backend("test_backend", FakeParallelBackend)
        assert_true("test_backend" in BACKENDS)
        assert_equal(BACKENDS["test_backend"], FakeParallelBackend)
    finally:
        del BACKENDS["test_backend"]


def test_overwrite_default_backend():
    assert_equal(_active_backend_type(), MultiprocessingBackend)
    try:
        register_parallel_backend("threading", BACKENDS["threading"],
                                  make_default=True)
        assert_equal(_active_backend_type(), ThreadingBackend)
    finally:
        # Restore the global default manually
        parallel.DEFAULT_BACKEND = 'multiprocessing'
    assert_equal(_active_backend_type(), MultiprocessingBackend)


def check_backend_context_manager(backend_name):
    with parallel_backend(backend_name, n_jobs=3):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert_equal(active_n_jobs, 3)
        assert_equal(effective_n_jobs(3), 3)
        p = Parallel()
        assert_equal(p.n_jobs, 3)
        if backend_name == 'multiprocessing':
            assert_equal(type(active_backend), MultiprocessingBackend)
            assert_equal(type(p._backend), MultiprocessingBackend)
        elif backend_name == 'threading':
            assert_equal(type(active_backend), ThreadingBackend)
            assert_equal(type(p._backend), ThreadingBackend)
        elif backend_name.startswith('test_'):
            assert_equal(type(active_backend), FakeParallelBackend)
            assert_equal(type(p._backend), FakeParallelBackend)


@with_multiprocessing
def test_backend_context_manager():
    all_test_backends = ['test_backend_%d' % i for i in range(3)]
    for test_backend in all_test_backends:
        register_parallel_backend(test_backend, FakeParallelBackend)
    all_backends = ['multiprocessing', 'threading'] + all_test_backends

    try:
        assert_equal(_active_backend_type(), MultiprocessingBackend)
        # check that this possible to switch parallel backends sequentially
        for test_backend in all_backends:
            yield check_backend_context_manager, test_backend

        # The default backend is retored
        assert_equal(_active_backend_type(), MultiprocessingBackend)

        # Check that context manager switching is thread safe:
        Parallel(n_jobs=2, backend='threading')(
            delayed(check_backend_context_manager)(b)
            for b in all_backends if not b)

        # The default backend is again retored
        assert_equal(_active_backend_type(), MultiprocessingBackend)
    finally:
        for backend_name in list(BACKENDS.keys()):
            if backend_name.startswith('test_'):
                del BACKENDS[backend_name]


class ParameterizedParallelBackend(SequentialBackend):
    """Pretends to run conncurrently while running sequentially."""

    def __init__(self, param=None):
        if param is None:
            raise ValueError('param should not be None')
        self.param = param


def test_parameterized_backend_context_manager():
    register_parallel_backend('param_backend', ParameterizedParallelBackend)
    try:
        assert_equal(_active_backend_type(), MultiprocessingBackend)

        with parallel_backend('param_backend', param=42, n_jobs=3):
            active_backend, active_n_jobs = parallel.get_active_backend()
            assert_equal(type(active_backend), ParameterizedParallelBackend)
            assert_equal(active_backend.param, 42)
            assert_equal(active_n_jobs, 3)
            p = Parallel()
            assert_equal(p.n_jobs, 3)
            assert_true(p._backend is active_backend)
            results = p(delayed(sqrt)(i) for i in range(5))
        assert_equal(results, [sqrt(i) for i in range(5)])

        # The default backend is again retored
        assert_equal(_active_backend_type(), MultiprocessingBackend)
    finally:
        del BACKENDS['param_backend']


def test_direct_parameterized_backend_context_manager():
    assert_equal(_active_backend_type(), MultiprocessingBackend)

    # Check that it's possible to pass a backend instance directly,
    # without registration
    with parallel_backend(ParameterizedParallelBackend(param=43), n_jobs=5):
        active_backend, active_n_jobs = parallel.get_active_backend()
        assert_equal(type(active_backend), ParameterizedParallelBackend)
        assert_equal(active_backend.param, 43)
        assert_equal(active_n_jobs, 5)
        p = Parallel()
        assert_equal(p.n_jobs, 5)
        assert_true(p._backend is active_backend)
        results = p(delayed(sqrt)(i) for i in range(5))
    assert_equal(results, [sqrt(i) for i in range(5)])

    # The default backend is again retored
    assert_equal(_active_backend_type(), MultiprocessingBackend)


###############################################################################
# Test helpers
def test_joblib_exception():
    # Smoke-test the custom exception
    e = JoblibException('foobar')
    # Test the repr
    repr(e)
    # Test the pickle
    pickle.dumps(e)


def test_safe_function():
    safe_division = SafeFunction(division)
    assert_raises(JoblibException, safe_division, 1, 0)


def test_invalid_batch_size():
    assert_raises(ValueError, Parallel, batch_size=0)
    assert_raises(ValueError, Parallel, batch_size=-1)
    assert_raises(ValueError, Parallel, batch_size=1.42)


def check_same_results(params):
    n_tasks = params.pop('n_tasks')
    expected = [square(i) for i in range(n_tasks)]
    results = Parallel(**params)(delayed(square)(i) for i in range(n_tasks))
    assert_equal(results, expected)


def test_dispatch_race_condition():
    # Check that using (async-)dispatch does not yield a race condition on the
    # iterable generator that is not thread-safe natively.
    # This is a non-regression test for the "Pool seems closed" class of error
    yield check_same_results, dict(n_tasks=2, n_jobs=2, pre_dispatch="all")
    yield check_same_results, dict(n_tasks=2, n_jobs=2, pre_dispatch="n_jobs")
    yield check_same_results, dict(n_tasks=10, n_jobs=2, pre_dispatch="n_jobs")
    yield check_same_results, dict(n_tasks=517, n_jobs=2,
                                   pre_dispatch="n_jobs")
    yield check_same_results, dict(n_tasks=10, n_jobs=2, pre_dispatch="n_jobs")
    yield check_same_results, dict(n_tasks=10, n_jobs=4, pre_dispatch="n_jobs")
    yield check_same_results, dict(n_tasks=25, n_jobs=4, batch_size=1)
    yield check_same_results, dict(n_tasks=25, n_jobs=4, batch_size=1,
                                   pre_dispatch="all")
    yield check_same_results, dict(n_tasks=25, n_jobs=4, batch_size=7)
    yield check_same_results, dict(n_tasks=10, n_jobs=4,
                                   pre_dispatch="2*n_jobs")


@with_multiprocessing
def test_default_mp_context():
    p = Parallel(n_jobs=2, backend='multiprocessing')
    context = p._backend_args.get('context')
    if sys.version_info >= (3, 4):
        start_method = context.get_start_method()
        # Under Python 3.4+ the multiprocessing context can be configured
        # by an environment variable
        env_method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
        if env_method is None:
            # Check the default behavior
            if sys.platform == 'win32':
                assert_equal(start_method, 'spawn')
            else:
                assert_equal(start_method, 'fork')
        else:
            assert_equal(start_method, env_method)
    else:
        assert_equal(context, None)


@with_multiprocessing
@with_numpy
def test_no_blas_crash_or_freeze_with_multiprocessing():
    if sys.version_info < (3, 4):
        raise nose.SkipTest('multiprocessing can cause BLAS freeze on'
                            ' old Python')

    # Use the spawn backend that is both robust and available on all platforms
    spawn_backend = mp.get_context('spawn')

    # Check that on recent Python version, the 'spawn' start method can make
    # it possible to use multiprocessing in conjunction of any BLAS
    # implementation that happens to be used by numpy with causing a freeze or
    # a crash
    rng = np.random.RandomState(42)

    # call BLAS DGEMM to force the initialization of the internal thread-pool
    # in the main process
    a = rng.randn(1000, 1000)
    np.dot(a, a.T)

    # check that the internal BLAS thread-pool is not in an inconsistent state
    # in the worker processes managed by multiprocessing
    Parallel(n_jobs=2, backend=spawn_backend)(
        delayed(np.dot)(a, a.T) for i in range(2))


def test_parallel_with_interactively_defined_functions():
    # When functions are defined interactively in a python/IPython
    # session, we want to be able to use them with joblib.Parallel
    if posix is None:
        # This test pass only when fork is the process start method
        raise nose.SkipTest('Not a POSIX platform')

    code = '\n\n'.join([
        'from joblib import Parallel, delayed',
        'def sqrt(x): return x**2',
        'print(Parallel(n_jobs=2)(delayed(sqrt)(i) for i in range(5)))'])

    check_subprocess_call([sys.executable, '-c', code],
                          stdout_regex=r'\[0, 1, 4, 9, 16\]')


def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert_equal(Parallel(n_jobs=2)(exhausted_iterator), [])


def check_memmap(a):
    if not isinstance(a, np.memmap):
        raise TypeError('Expected np.memmap instance, got %r',
                        type(a))
    return a.copy()  # return a regular array instead of a memmap


@with_numpy
@with_multiprocessing
def test_auto_memmap_on_arrays_from_generator():
    # Non-regression test for a problem with a bad interaction between the
    # GC collecting arrays recently created during iteration inside the
    # parallel dispatch loop and the auto-memmap feature of Parallel.
    # See: https://github.com/joblib/joblib/pull/294
    def generate_arrays(n):
        for i in range(n):
            yield np.ones(10, dtype=np.float32) * i
    # Use max_nbytes=1 to force the use of memory-mapping even for small
    # arrays
    results = Parallel(n_jobs=2, max_nbytes=1)(
        delayed(check_memmap)(a) for a in generate_arrays(100))
    for result, expected in zip(results, generate_arrays(len(results))):
        np.testing.assert_array_equal(expected, result)


@with_multiprocessing
def test_nested_parallel_warnings():
    # The warnings happen in child processes so
    # warnings.catch_warnings can not be used for this tests that's
    # why we use check_subprocess_call instead
    if posix is None:
        # This test pass only when fork is the process start method
        raise nose.SkipTest('Not a POSIX platform')

    template_code = """
import sys

from joblib import Parallel, delayed


def func():
    return 42


def parallel_func():
    res =  Parallel(n_jobs={inner_n_jobs})(delayed(func)() for _ in range(3))
    return res

Parallel(n_jobs={outer_n_jobs})(delayed(parallel_func)() for _ in range(5))
    """
    # no warnings if inner_n_jobs=1
    code = template_code.format(inner_n_jobs=1, outer_n_jobs=2)
    check_subprocess_call([sys.executable, '-c', code],
                          stderr_regex='^$')

    #  warnings if inner_n_jobs != 1
    regex = ('Multiprocessing-backed parallel loops cannot '
             'be nested')
    code = template_code.format(inner_n_jobs=2, outer_n_jobs=2)
    check_subprocess_call([sys.executable, '-c', code],
                          stderr_regex=regex)
