"""Benchmark a small scale scikit-learn GridSearch and the scaling with n_jobs.

This benchmark requires ``scikit-learn`` to be installed.

The goal of this script is to make sure the scaling does not worsen with time.
In particular, it can be used to compare 2 joblib versions by first running
the benchmark with the option `-n name1`, then changing the joblib version and
running the script with option `-c name1`. This option can be used multiple
times to build a comparison with more than 2 version.
"""
from time import time

import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV


def get_file_name(name):
    return f"bench_gs_scaling_{name}.npy"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument(
        "--n-rep", "-r", type=int, default=5,
        help="Number of repetition to average on."
    )
    parser.add_argument(
        "--name", "-n", type=str, default="",
        help="Name to save the results with. This can be used to compare "
        "different branches with '-c'."
    )
    parser.add_argument(
        "--compare", "-c", action="append",
        help="Loads the results from a benchmark saved previously with a name "
        "given as the present argument value. This allows comparing the "
        "results across different versions of joblib."
    )
    args = parser.parse_args()

    # Generate a synthetic dataset for classification.
    rng = np.random.RandomState(0)
    X, y = datasets.make_classification(n_samples=1000, random_state=rng)

    #
    gammas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    Cs = [1, 10, 100, 1e3, 1e4, 1e5]
    param_grid = {"gamma": gammas, "C": Cs}

    clf = SVC(random_state=rng)

    # Warm up run to avoid the first run overhead of starting the executor.
    GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1).fit(X, y)

    # We run the n_jobs in decreasing order to avoid the issue joblib/loky#396
    # that make the queue size too small when increasing an executor size.
    res = []
    for n_jobs in range(joblib.cpu_count(), 0, -2):
        T = []
        for _ in range(args.n_rep):
            tic = time()
            gs = GridSearchCV(
                estimator=clf, param_grid=param_grid, n_jobs=n_jobs
            )
            gs.fit(X, y)
            T += [time() - tic]
        res += [(n_jobs, *np.quantile(T, [0.5, 0.2, 0.8]))]
    res = np.array(res).T

    if args.name:
        fname = get_file_name(args.name)
        np.save(fname, res)

    label = args.name or "current"
    plt.fill_between(res[0], res[2], res[3], alpha=0.3, color="C0")
    plt.plot(res[0], res[1], c="C0", lw=2, label=label)

    if args.compare:
        for i, name_c in enumerate(args.compare):
            fname_compare = get_file_name(name_c)
            res_c = np.load(fname_compare)
            plt.fill_between(
                res_c[0], res_c[2], res_c[3], alpha=0.3, color=f"C{i+1}"
            )
            plt.plot(res_c[0], res_c[1], c=f"C{i+1}", lw=2, label=name_c)

    plt.xlabel("n_jobs")
    plt.ylabel("Time [s]")
    plt.ylim(0, None)
    plt.legend()
    plt.show()
