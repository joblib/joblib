"""
===========================================================
Running a self-sustained flow of parallel tasks with joblib
===========================================================

This example introduces a recipe for implementing a feedback loop where the
output generator of joblib.Parallel is used to fuel the input generator with
new tasks, based on the value that are returned by the tasks that have been
already completed, until some stopping criterion is met.

The toy example given here starts by scheduling a few tasks that consists in
incrementing an integer and returning it, then the feedback loop creates new
tasks that increments again the returned integer, and so on, for as long as the
integers remain below some threshold.

The toy example has little value in itself, but introduces the skeleton of the
implementation of this particular computational model, and it could be extended
 to fulfill real world needs.

For instance one such application is bayesian hyper-parameter search where the
grid of parameters is explored based on a bayesian strategy, in which the
cross-validated performance of hyper-parameter combinations that have been
previously evaluated are used to choose the hyper-parameters that will be
explored in the next rounds.

"""


##############################################################################
# The following class implements the computational model. It can be easily
# adapted to various usecases, since the functions that define the initial
# tasks and the feedback loop are abstracted

import joblib
from queue import SimpleQueue


class SelfSustainingParallelWork:

    def __init__(self, parallel_n_jobs, parallel_batch_size=1):
        self.parallel = joblib.Parallel(
            n_jobs=parallel_n_jobs,
            return_as="generator_unordered",
            batch_size=parallel_batch_size,
        )

        if self.parallel.batch_size == "auto":
            # Adaptive batch sizing might set batch size to a value higher than
            # `n_jobs` during compute, which would cause a deadlock, since
            # then `joblib.Parallel` would risk having to wait for more jobs
            # than the feedback loop can provide.
            raise ValueError(
                "Setting joblib.Parallel `batch_size` parameter to 'auto' is "
                "not allowed in SelfSustainingParallelWork"
            )

    def __call__(
            self,
            starting_tasks,
            task_from_output_fn,
            task_completion_callback
    ):
        """Main function that starts the compute and the feedback loop.

        Parameters
        ----------
        starting_tasks: iterable of functions
            An iterable of functions that take no arguments. The compute will
            start by dispatching those tasks to the workers.

        task_from_output_fn: function
            A function that the feedback loop will used to create new tasks
            from previous results. The function must expect one argument, which
            will be passed the output returned by the latest completed task. It
            must return either a joblib.delayed-like function, either None, in
            which case no new task will be dispatched.

        task_completion_callback: function
            A function that will be called in the calling thread each time an
            output is retrieved, taking said output as argument. It can be used
            for recording, plotting, logging,...
        """
        # It's required to use a multi-threaded queue here, so that the
        # callback threads in joblib.Parallel internal mechanisms can also
        # get the new tasks as soon as it's available and dispatch it.
        task_feedback_queue = SimpleQueue()

        # Start the compute...
        input_generator = self._input_generator(
            starting_tasks, task_from_output_fn, task_feedback_queue
        )
        _output_generator = self.parallel(input_generator)

        # ... start retrieving outputs...
        for output in _output_generator:
            task_completion_callback(output)

            # ...create a new task based on the outputs...
            try:
                new_task = task_from_output_fn(output)
            except BaseException:
                # ...but make sure that the input generator stops properly
                # in case the feedback loop raises an exception...
                task_feedback_queue.put(None)
                raise

            # ...make the new task available to workers...
            task_feedback_queue.put(new_task)

            # ...until the stopping criterion is met...
            if new_task is None:
                break

        # ...then, retrieve the remaining outputs.
        for output in _output_generator:
            task_completion_callback(output)

    def _input_generator(
            self,
            starting_tasks,
            task_from_output_fn,
            task_feedback_queue
    ):
        '''The input generator, that forwards new tasks created in the calling
        thread to the workers.
        '''
        nb_pending_results = 0

        # First, dispatch the starting tasks.
        for task in starting_tasks:
            nb_pending_results += 1
            yield task

        # Need enough starting tasks start the dispatch with respect to the
        # batch_size and pre_dispatch parameters.
        if nb_pending_results < (
                minimum_starting_tasks_nb := (
                    self.parallel.batch_size *
                    self.parallel._pre_dispatch_amount
            )
        ):
            raise ValueError(
                "Expected at least batch_size * parallel._pre_dispatch_amount "
                f"= {minimum_starting_tasks_nb} starting tasks, but only got "
                f"{nb_pending_results} starting tasks."
            )

        while True:
            # Wait for a next task to be available...
            new_task = task_feedback_queue.get()

            # ...if the feedback loop does not return a new task this time,
            # it means the compute stops after the remaining pending tasks
            # are done.
            if new_task is None:
                return

            yield new_task


##############################################################################
# Now let's define our starting tasks, the feedback loop, and a counter
# callback.

def add_one(i):
    return i + 1


starting_tasks = (joblib.delayed(add_one)(i) for i in range(10))


def task_from_output_fn(output):
    # Stopping criterion: after the first time an integer went above 100, stop
    # issuing new tasks.
    if output >= 100:
        return

    return joblib.delayed(add_one)(output)


class CounterCallback:
    def __init__(self):
        self.counter = 0

    def __call__(self, output):
        self.counter += 1

    def get_count(self):
        return self.counter


##############################################################################
# Start the compute and check the counter.

callback = CounterCallback()
SelfSustainingParallelWork(parallel_n_jobs=2)(
    starting_tasks,
    task_from_output_fn,
    callback
)

# NB: in our example, the number of tasks done can change between runs since
# it depends on concurrency between workers until one of them meets the
# stopping criterion first.
print(f"{callback.get_count()} tasks done.")