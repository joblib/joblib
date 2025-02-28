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
set_config(backend="loky")

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
with config_context(backend="threading"):
    print(f"The backend in the context is {get_config()['backend']}")
print(f"The backend outside the context is {get_config()['backend']}")

# %%
# Share configuration between parallel workers
# --------------------------------------------
#


# %%
def get_backend_in_worker():
    return get_config()


# %%
from joblib import Parallel, delayed

with config_context(backend="threading"):
    result = Parallel(n_jobs=2)(delayed(get_backend_in_worker)() for _ in range(2))
result

# %%
with config_context(backend="threading"):
    call_context = [(config_context, get_config)]
    result = Parallel(n_jobs=2, call_context=call_context)(
        delayed(get_backend_in_worker)() for _ in range(2)
    )
print(f"backend in the main process: {get_config()['backend']}")
print(f"backend in the parallel workers: {result}")

# %%
from joblib import register_call_context

register_call_context(
    context_name="joblib_config",
    context_manager=config_context,
    state_retriever=get_config,
)

# %%
with config_context(backend="threading"):
    result = Parallel(n_jobs=2)(delayed(get_backend_in_worker)() for _ in range(2))
result

# %%
from joblib import unregister_call_context

unregister_call_context("joblib_config")

# %%
with config_context(backend="threading"):
    result = Parallel(n_jobs=2)(delayed(get_backend_in_worker)() for _ in range(2))
result

# %%
