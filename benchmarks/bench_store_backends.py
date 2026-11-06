import os
import sqlite3
from time import time

import matplotlib.pyplot as plt
import numpy as np

from joblib import Memory, Parallel, delayed


def fun(d):
    ans = 0
    for k, x in d.items():
        ans += k * x.sum()
    return ans


def gen_value(size_dict, size_array):
    d = dict()
    for k in np.random.choice(10**4, size_dict, replace=False):
        d[k] = np.random.randint(0, 10**6, size_array)
    return d


def get_dir_size(path):
    if os.path.isfile(path):
        con = sqlite3.connect(path)
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        return os.stat(path).st_blocks * 512
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        total_size += os.stat(dirpath).st_blocks * 512
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.stat(fp).st_blocks * 512
    return total_size


fs = Memory(location="cache_fs", backend="local", verbose=0)
sq = Memory(location="cache_sql", backend="sqlite", verbose=0)
fun_fs = delayed(fs.cache(fun))
fun_sq = delayed(sq.cache(fun))

fig, (ax_time, ax_mem) = plt.subplots(1, 2, figsize=(10, 4))

for num_objects in [100, 1000, 10_000]:
    true_sizes = []
    sq_data = [], []
    fs_data = [], []

    indices = np.repeat(np.arange(num_objects), 5)
    np.random.shuffle(indices)

    for tot_size in [10, 100, 1000]:  # in Mb
        size = 10**6 * tot_size // (8 * num_objects)  # in bytes
        size_dict = int(size**0.33)
        size_array = size // size_dict
        true_size = 8 * num_objects * size_dict * size_array
        values = [gen_value(size_dict, size_array) for _ in range(num_objects)]
        values = [values[i] for i in indices]

        fs.clear(warn=False)
        sq.clear(warn=False)

        true_sizes.append(true_size)
        for data, fun in [(sq_data, fun_sq), (fs_data, fun_fs)]:
            t0 = time()
            list(Parallel(n_jobs=-1)(fun(d) for d in values))
            t1 = time()
            data[0].append(t1 - t0)
            data[1].append(get_dir_size(fun.store_backend.location))

    col = ax_time.loglog(true_sizes, sq_data[0])[0].get_color()
    ax_mem.loglog(true_sizes, sq_data[1], label=f"SQL {num_objects} objs")
    ax_time.loglog(true_sizes, fs_data[0], linestyle="--", color=col)
    ax_mem.loglog(
        true_sizes,
        fs_data[1],
        linestyle="--",
        color=col,
        label=f"FileSystem {num_objects} objs",
    )

ax_time.set_xlabel("total size of inputs (bytes)")
ax_time.set_ylabel("time (sec)")
ax_mem.set_xlabel("total size of inputs (bytes)")
ax_mem.set_ylabel("cache size (bytes)")
ax_mem.legend()

plt.tight_layout()
plt.show()
