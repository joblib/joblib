"""
=================================================
Using tqdm to track progress with joblib.Parallel
=================================================

This example demonstrates how to use a ``tqdm`` progress bar together with
:class:`joblib.Parallel`.

We present three main approaches:

1.  :ref:`solution_generator`, via ``return_as='generator'`` to update the
    progress bar as tasks finish. This works well for homogeneous tasks but
    has some limitations for heterogeneous ones.
2.  :ref:`solution_unordered_generator`, which works for heterogeneous tasks
    but does not preserve the tasks' order.
3.  :ref:`solution_subclass`, which requires more code but allows accurate
    progress reporting independently of the ``return_as`` option.

Note that ``return_as='generator'`` is not available for the ``'multiprocessing'``
backend.
"""

##############################################################################
# Task definition
# ===============
#
# We first define a task that sleeps for a given amount of time and returns
# that value. We also define a list of sleep durations that will be used.
# Note that the task durations are heterogeneous, simulating a situation
# where not all tasks take the same amount of time to complete.

import time


def task(t):
    time.sleep(0.1 * t)
    return t


times = [7, 2, 3, 5, 6, 4, 1]


##############################################################################
# A first naive solution
# ======================
#
# .. warning::
#
#     This solution reports the number of *dispatched* tasks rather than the
#     number of *completed* tasks. It is presented for pedagogical purposes
#     and should not be used in practice.
#
# A ``tqdm`` progress bar takes an iterable as input and returns an iterable.
# A :class:`~joblib.Parallel` call also consumes an iterable.
#
# A straightforward solution is therefore to wrap the input iterable with
# ``tqdm`` before passing it to :class:`~joblib.Parallel`. The progress bar
# then reflects the number of tasks dispatched to the workers.

from tqdm import tqdm

from joblib import Parallel, delayed

p = Parallel(n_jobs=2, pre_dispatch=4, batch_size=1)
out = p(delayed(task)(t) for t in tqdm(times))
print(*out)

##############################################################################
# As can be observed, the progress bar initially jumps from 0 to 4 tasks.
# This is due to the ``pre_dispatch=4`` argument, which instructs
# :class:`~joblib.Parallel` to dispatch 4 tasks at the start.
# Afterwards, tasks are dispatched in chunks of ``n_jobs * batch_size``.
# In this example, the progress bar advances in steps of 2 because
# ``n_jobs=2`` and ``batch_size=1``.
#
# For large ``batch_size`` values, this progress bar becomes less precise
# in reflecting how many tasks have started. Moreover, this solution reports
# the number of dispatched tasks, whereas one might instead be interested in
# the number of completed tasks. The following examples show how to track
# completed tasks instead.

##############################################################################
#
# .. _solution_generator:
#
# Using a generator
# =================
#
# .. note::
#
#     This solution works for homogeneous tasks but updates progress based on
#     task order. It fails to report progress when a later task completes
#     while an earlier one is still running.
#
# Since a :class:`~joblib.Parallel` call returns an iterable over task outputs
# (by default, a list), one might try to wrap the returned iterable with
# ``tqdm``.
#
# However, since the default return type is a ``list``, all tasks must
# complete before the list becomes available. To observe progress as tasks
# finish, we must use ``return_as='generator'``, which yields outputs as they
# become available.

p = Parallel(n_jobs=2, return_as="generator")
out = p(delayed(task)(t) for t in times)
print(*tqdm(out, total=len(times)))

##############################################################################
# The progress bar now updates as results are produced. However, we still
# observe a jump from 1 task directly to 4 tasks.
#
# This happens because when using ``return_as='generator'``, outputs are
# yielded in the same order as the input tasks. The first yielded value
# therefore corresponds to the task that takes 0.7 seconds to complete
# (recall ``times = [7, 2, 3, 5, 6, 4, 1]``). When this first task finishes,
# the second and third tasks have already completed as well, since
# 2 + 3 < 7. This explains the jump from 1/7 to 4/7.
#
# Note that we must explicitly pass ``total=len(times)`` to ``tqdm`` because
# the returned generator does not define a length, unlike a ``list``.

##############################################################################
#
# .. _solution_unordered_generator:
#
# Using an unordered generator
# ============================
#
# .. note::
#
#     This solution does not preserve the tasks' input order in the results
#
# If you do not care about the order of the outputs and want smoother
# progress reporting, you can use ``return_as='generator_unordered'``.
#
# In this case, results are yielded as soon as tasks complete, allowing the
# progress bar to advance one step at a time.

p = Parallel(n_jobs=2, return_as="generator_unordered")
out = p(delayed(task)(t) for t in times)
print(*tqdm(out, total=len(times)))

##############################################################################
# The progress bar now advances smoothly. Note how the order of the outputs
# differs from the original input order.
# If you need to preserve the original ordering, you could modify the task
# function to also return the index of the task.

##############################################################################
#
# .. _solution_subclass:
#
# Subclassing :class:`joblib.Parallel`
# ====================================
#
# The :class:`~joblib.Parallel` class provides a method
# :meth:`~joblib.Parallel.print_progress`, which is called whenever a task
# completes. By default, this method prints varying amounts of information
# depending on the value of the ``verbose`` parameter.
#
# Here, we override this method in a custom subclass in order to update a
# ``tqdm`` progress bar instead.


class ParallelTqdm(Parallel):
    def __call__(self, iterable, n_tasks):
        self.tqdm = tqdm(total=n_tasks)
        return super().__call__(iterable)

    def print_progress(self):
        self.tqdm.update(self.n_completed_tasks - self.tqdm.n)
        if self._is_completed():
            self.tqdm.close()


p = ParallelTqdm(n_jobs=2)
out = p((delayed(task)(t) for t in times), len(times))
print(*out)

##############################################################################
# This approach provides accurate progress reporting for heterogeneous tasks
# without changing the result ordering. However, it requires a more complex
# interface, so the previous solutions may be preferable for simpler use
# cases.
