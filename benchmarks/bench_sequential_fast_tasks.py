"""Benchmark n_jobs=1 on high number of fast tasks

The goal of this script is to study the overhead incurred when calling small
tasks with `n_jobs=1` compared to just running a simple list comprehension.

"""
# Author: Thomas Moreau
# License: BSD 3 clause

import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable


from joblib import Parallel, delayed


# Style for plots
LINE_STYLES = {'iter': '--', 'parallel': '-', 'loop': ':'}
COLORS = {'none': 'indianred'}
CMAP = plt.colormaps['viridis']

# Generate functions that are more and more complex, to see
# the relative impact depending on the task complexity
funcs = [("none", lambda x: None, None)]
n_size = 3
for i, n in enumerate(np.logspace(0, 2, n_size, dtype=int)):
    n = max(1, n)
    label = f'mat({n:3d}, {n:3d})'
    A = np.random.randn(n, n)
    funcs.append((label, lambda A: A @ A, A))
    COLORS[label] = CMAP(i / (n_size - 1))

# For each function and for different number of repetition,
# time the Parallel call.
results = []
for f_name, func, arg in funcs:
    print('Benchmarking:', f_name)
    f_delayed = delayed(func)
    for N in np.logspace(1, 4, 4, dtype=int):
        print('# tasks:', N)
        for _ in range(10):

            t_start = time.perf_counter()
            list(func(arg) for _ in range(N))
            runtime = time.perf_counter() - t_start
            results.append(dict(
                method="iter", N=N, func=f_name, runtime=runtime / N
            ))

            t_start = time.perf_counter()
            Parallel(n_jobs=1)(f_delayed(arg) for _ in range(N))
            runtime = time.perf_counter() - t_start
            results.append(dict(
                method="parallel", N=N, func=f_name, runtime=runtime / N
            ))

# Use a DataFrame to manipulate the results.
df = pd.DataFrame(results)

# Compute median runtime for each set of parameters
curve = df.groupby(["method", "N", "func"])["runtime"].median().reset_index()

# Print the overhead incurred for each task (estimated as the median of the
# time difference per task).
for k, grp in curve.groupby("func"):

    c_iter = grp.query('method == "iter"').set_index("N")
    c_parallel = grp.query('method == "parallel"').set_index("N")
    overhead_percent = (c_parallel["runtime"] / c_iter["runtime"]).median() - 1
    overhead_time = (c_parallel["runtime"] - c_iter["runtime"]) / c_iter.index
    print(
        f"For func {k}, overhead_time is {overhead_time.median()/1e-6:.2f}us, "
        f"increasing runtime by {overhead_percent * 100:.2f}%"
    )

# Plot the scaling curves.
fig, ax = plt.subplots()
for key, grp in curve.groupby(["method", "func"]):
    ax.loglog(
        grp["N"],
        grp["runtime"],
        label=key,
        ls=LINE_STYLES[key[0]],
        color=COLORS[key[1]],
    )

ax.set_xlabel("# Tasks")
ax.set_ylabel("Runtime per task")
ax.legend(
    (plt.Line2D([], [], ls=ls, c="k") for ls in LINE_STYLES.values()),
    LINE_STYLES,
    bbox_to_anchor=(0.3, 1, 0.3, 0.1),
    loc='center',
    ncol=3,
)
plt.colorbar(
    ScalarMappable(norm=LogNorm(1, 200), cmap=plt.cm.viridis),
    label="Matrix size"
)

plt.show()
