from concurrent.futures import ThreadPoolExecutor
from time import time_ns

from joblib import Parallel, delayed


def noop(_x=None):
    pass


def work_10us(_x=None):
    start = time_ns()
    while time_ns() - start < 10_000:
        pass


def joblib_single_delayed(n_jobs, n_tasks, func_to_run):
    result = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(func_to_run)() for _ in range(n_tasks)
    )
    assert len(list(result)) == n_tasks


def cfexecutor(n_jobs, n_tasks, func_to_run):
    with ThreadPoolExecutor(n_jobs) as executor:
        result = executor.map(func_to_run, range(n_tasks))
    assert len(list(result)) == n_tasks


def go(n_jobs, num_tasks, func_to_run, runner):
    start = time_ns()
    runner(n_jobs, num_tasks, func_to_run)
    elapsed = time_ns() - start
    print(
        f"{func.__name__}(): {n_jobs} workers, "
        + f"{num_tasks:5} tasks, {elapsed // 1000:8} µs"
    )


if __name__ == "__main__":
    print("== Threaded/sequential backend ==")
    for func in [noop, work_10us]:
        for n_jobs in [4, 1]:
            for num_tasks in [10, 100, 1000, 10_000]:
                go(n_jobs, num_tasks, func, joblib_single_delayed)
                go(n_jobs, num_tasks, func, joblib_single_delayed)

    print()
    print("== ThreadPoolExecutor ==")
    for func in [noop, work_10us]:
        for num_tasks in [10, 100, 1000, 10_000]:
            go(4, num_tasks, func, cfexecutor)
            go(4, num_tasks, func, cfexecutor)
