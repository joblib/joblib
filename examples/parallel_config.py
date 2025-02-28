"""
==============================================
Sharing configuration between parallel workers
==============================================

This example illustrates how to share a configuration between the parallel driver
and the parallel workers.

First, we show that `joblib` exposes some tools to manage some configuration that
can be used by any third party library. Then, we give two examples of how to share
such configuration between the main process and the parallel workers.
"""

# %%
# Configuration utilities in `joblib`
# -----------------------------------
#
# `joblib` exposes three functions, :func:`~joblib.get_config`,
# :func:`~joblib.set_config` and :func:`~joblib.config_context`, to get, set and
# temporarily set configuration.
from joblib import config_context, get_config, set_config

# %%
# By default, the configuration is empty.
get_config()

# %%
# We can set some configuration using :func:`~joblib.set_config`.
set_config(parameter="value")

# %%
# This string will be stored in the joblib configuration.
get_config()

# %%
# It is important to note that this configuration does not any effect on a particular
# processing in `joblib`. It is just an handy way to share some configuration between
# the main process and the parallel workers available in `joblib`.
#
# :func:`~joblib.config_context` is a context manager that will restore the previous
# configuration when the context is exited.
with config_context(parameter="another value"):
    print(f"The parameter in the context is {get_config()['parameter']}")
print(f"The parameter outside the context is {get_config()['parameter']}")

# %%
# Share configuration between parallel workers
# --------------------------------------------
#

# %%
from joblib import list_call_context_names

list_call_context_names()


# %%
def get_joblib_config_in_worker():
    return get_config()


# %%
from joblib import Parallel, delayed

with config_context(parameter="another value"):
    result = Parallel(n_jobs=2)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
result = Parallel(n_jobs=2)(delayed(get_joblib_config_in_worker)() for _ in range(2))
result

# %%
from joblib import unregister_call_context

unregister_call_context("joblib")

# %%
result = Parallel(n_jobs=2)(delayed(get_joblib_config_in_worker)() for _ in range(2))
result

# %%
with config_context(parameter="another value"):
    result = Parallel(n_jobs=2)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
call_context = [(config_context, get_config)]
result = Parallel(n_jobs=2, call_context=call_context)(
    delayed(get_joblib_config_in_worker)() for _ in range(2)
)
result


# %%
with config_context(parameter="another value"):
    result = Parallel(n_jobs=2, call_context=call_context)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
import warnings


def func_that_warns(value):
    if value > 10:
        warnings.warn("This function is warning", RuntimeWarning)
    return value


# %%
result = Parallel(n_jobs=2)(delayed(func_that_warns)(i) for i in range(8, 12))

# %%
from contextlib import contextmanager


@contextmanager
def silence_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield
        warnings.resetwarnings()


# %%
call_context = [(silence_warnings, lambda: {})]
result = Parallel(n_jobs=2, call_context=call_context)(
    delayed(func_that_warns)(i) for i in range(8, 12)
)
result


# %%
