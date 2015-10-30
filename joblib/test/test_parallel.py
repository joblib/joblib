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
from joblib.test.common import np, with_numpy

try:
    import cPickle as pickle
    PickleError = TypeError
except:
    import pickle
    PickleError = pickle.PicklingError


if sys.version_info[0] == 3:
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


from joblib.parallel import Parallel, delayed, SafeFunction, WorkerInterrupt
from joblib.parallel import mp, cpu_count, VALID_BACKENDS
from joblib.my_exceptions import JoblibException

import nose
from nose.tools import assert_equal, assert_true, assert_false, assert_raises


ALL_VALID_BACKENDS = [None] + VALID_BACKENDS

if hasattr(mp, 'get_context'):
    # Custom multiprocessing context in Python 3.4+
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))


def division(x, y):
    return x / y


def square(x):
    return x ** 2


def exception_raiser(x):
    if x == 7:
        raise ValueError
    return x


def interrupt_raiser(x):
    time.sleep(.05)
    raise KeyboardInterrupt


def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x ** 2 + y + z


###############################################################################
def test_cpu_count():
    assert cpu_count() > 0


###############################################################################
# Test parallel
def check_simple_parallel(backend):
    X = range(5)
    for n_jobs in (1, 2, -1, -2):
        nose.tools.assert_equal(
            [square(x) for x in X],
            Parallel(n_jobs=n_jobs)(delayed(square)(x) for x in X))
    try:
        # To smoke-test verbosity, we capture stdout
        orig_stdout = sys.stdout
        orig_stderr = sys.stdout
        if sys.version_info[0] == 3:
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
    for parent_backend in VALID_BACKENDS:
        for child_backend in VALID_BACKENDS:
            yield check_nested_loop, parent_backend, child_backend


def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    q = Queue(maxsize=5)
    Parallel(n_jobs=2, backend="threading")(
        delayed(q.put, check_pickle=False)(1) for _ in range(5))
    nose.tools.assert_true(q.full())


def test_parallel_kwargs():
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    for n_jobs in (1, 4):
        yield (assert_equal,
               [f(x, y=1) for x in lst],
               Parallel(n_jobs=n_jobs)(delayed(f)(x, y=1) for x in lst))


def check_parallel_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]
    with Parallel(n_jobs=4, backend=backend) as p:
        # Internally a pool instance has been eagerly created and is managed
        # via the context manager protocol
        managed_pool = p._pool
        if mp is not None:
            assert_true(managed_pool is not None)

        # We make call with the managed parallel object several times inside
        # the managed block:
        assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))
        assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))

        # Those calls have all used the same pool instance:
        if mp is not None:
            assert_true(managed_pool is p._pool)

    # As soon as we exit the context manager block, the pool is terminated and
    # no longer referenced from the parallel object:
    assert_true(p._pool is None)

    # It's still possible to use the parallel instance in non-managed mode:
    assert_equal(expected, p(delayed(f)(x, y=1) for x in lst))
    assert_true(p._pool is None)


def test_parallel_context_manager():
    for backend in ['multiprocessing', 'threading']:
        yield check_parallel_context_manager, backend


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
            assert_true(parallel._pool is not None)

            assert_raises(JoblibException, parallel,
                          [delayed(division)(x, y)
                           for x, y in zip((0, 1), (1, 0))])

            # The managed pool should still be available and be in a working
            # state despite the previously raised (and caught) exception
            assert_true(parallel._pool is not None)
            assert_equal([f(x, y=1) for x in range(10)],
                         parallel(delayed(f)(x, y=1) for x in range(10)))

            assert_raises(WorkerInterrupt, parallel,
                          [delayed(interrupt_raiser)(x) for x in (1, 0)])

            # The pool should still be available despite the exception
            assert_true(parallel._pool is not None)
            assert_equal([f(x, y=1) for x in range(10)],
                         parallel(delayed(f)(x, y=1) for x in range(10)))

        # Check that the inner pool has been terminated when exiting the
        # context manager
        assert_true(parallel._pool is None)
    else:
        assert_raises(KeyboardInterrupt, Parallel(n_jobs=2),
                      [delayed(interrupt_raiser)(x) for x in (1, 0)])

    # wrapped exceptions should inherit from the class of the original
    # exception to make it easy to catch them
    assert_raises(ZeroDivisionError, Parallel(n_jobs=2),
                  [delayed(division)(x, y) for x, y in zip((0, 1), (1, 0))])
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
        nose.tools.assert_equal(len(self.list1), len(self.list2))


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
    nose.tools.assert_equal(queue, [
        'Produced 0', 'Consumed 0',
        'Produced 1', 'Consumed 1',
        'Produced 2', 'Consumed 2',
        'Produced 3', 'Consumed 3',
        'Produced 4', 'Consumed 4',
        'Produced 5', 'Consumed 5',
    ])
    nose.tools.assert_equal(len(queue), 12)

    # empty the queue for the next check
    queue[:] = []

    # enable batching
    Parallel(n_jobs=1, batch_size=4, backend=backend)(
        delayed(consumer)(queue, x) for x in producer())
    nose.tools.assert_equal(queue, [
        # First batch
        'Produced 0', 'Produced 1', 'Produced 2', 'Produced 3',
        'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3',

        # Second batch
        'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5',
    ])
    nose.tools.assert_equal(len(queue), 12)


def test_dispatch_one_job():
    for backend in VALID_BACKENDS:
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
    nose.tools.assert_equal(first_four,
                            ['Produced 0', 'Produced 1', 'Produced 2'])
    nose.tools.assert_equal(len(queue), 12)


def test_dispatch_multiprocessing():
    for backend in VALID_BACKENDS:
        yield check_dispatch_multiprocessing, backend


def test_batching_auto_threading():
    # batching='auto' with the threading backend leaves the effective batch
    # size to 1 (no batching) as it has been found to never be beneficial with
    # this low-overhead backend.
    p = Parallel(n_jobs=2, batch_size='auto', backend='threading')
    p(delayed(id)(i) for i in range(5000))  # many very fast tasks
    assert_equal(p._effective_batch_size, 1)


def test_batching_auto_multiprocessing():
    p = Parallel(n_jobs=2, batch_size='auto', backend='multiprocessing')
    p(delayed(id)(i) for i in range(5000))  # many very fast tasks

    # When the auto-tuning of the batch size is enabled
    # size kicks in the following attribute gets updated.
    assert_true(hasattr(p, '_effective_batch_size'))

    # It should be strictly larger than 1 but as we don't want heisen failures
    # on clogged CI worker environment be safe and only check that it's a
    # strictly positive number.
    assert_true(p._effective_batch_size > 0)


def test_exception_dispatch():
    "Make sure that exception raised during dispatch are indeed captured"
    nose.tools.assert_raises(
            ValueError,
            Parallel(n_jobs=2, pre_dispatch=16, verbose=0),
                    (delayed(exception_raiser)(i) for i in range(30)),
            )


def test_nested_exception_dispatch():
    # Ensure TransportableException objects for nested joblib cases gets
    # propagated.
    nose.tools.assert_raises(
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
    nose.tools.assert_raises(ImportError, Parallel(n_jobs=2, pre_dispatch='all'),
                             [delayed(_reload_joblib)() for i in range(10)])


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
    nose.tools.assert_raises(JoblibException, safe_division, 1, 0)


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


def test_default_mp_context():
    p = Parallel(n_jobs=2, backend='multiprocessing')
    if sys.version_info >= (3, 4):
        # Under Python 3.4+ the multiprocessing context can be configured
        # by an environment variable
        env_method = os.environ.get('JOBLIB_START_METHOD', '').strip() or None
        if env_method is None:
            # Check the default behavior
            if sys.platform == 'win32':
                assert_equal(p._mp_context.get_start_method(), 'spawn')
            else:
                assert_equal(p._mp_context.get_start_method(), 'fork')
        else:
            assert_equal(p._mp_context.get_start_method(), env_method)
    else:
        assert_equal(p._mp_context, None)


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
