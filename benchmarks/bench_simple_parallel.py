from time import time_ns

from joblib import Parallel, delayed


def noop():
    pass


def joblib_single_delayed(n_jobs, n_tasks):
    result = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(noop)() for _ in range(n_tasks)
    )
    assert len(list(result)) == n_tasks


def go(n_jobs, num_tasks):
    start = time_ns()
    joblib_single_delayed(n_jobs, num_tasks)
    elapsed = time_ns() - start
    print(f"{n_jobs} workers, {num_tasks:5} tasks, {elapsed // 1000:8} µs")


if __name__ == "__main__":
    print("== Threaded/sequential backend ==")
    for n_jobs in [4, 1]:
        for num_tasks in [10, 100, 1000, 10_000]:
            go(n_jobs, num_tasks)
            go(n_jobs, num_tasks)
