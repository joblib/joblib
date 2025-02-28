"""
==============================================
Sharing configuration between parallel workers
==============================================

This example shows how to share configuration between the parallel driver
and parallel workers.

First, we demonstrate joblib's tools for managing configuration that any third-party
library can use. Then, we provide two examples of sharing such configuration between
the main process and parallel workers.
"""

# %%
# Configuration utilities in `joblib`
# -----------------------------------
#
# `joblib` provides three functions to manage configuration:
# - :func:`~joblib.get_config`: gets current configuration
# - :func:`~joblib.set_config`: sets configuration
# - :func:`~joblib.config_context`: temporarily sets configuration
from joblib import config_context, get_config, set_config

# %%
# The configuration starts empty by default.
get_config()

# %%
# Use :func:`~joblib.set_config` to set configuration values.
set_config(parameter="value")

# %%
# joblib stores this string in its configuration.
get_config()

# %%
# Note that this configuration doesn't affect joblib's processing directly. It simply
# provides a convenient way to share configuration between the main process and
# parallel workers.
#
# :func:`~joblib.config_context` lets you temporarily change the configuration and
# restores the previous values when exiting the context.
with config_context(parameter="another value"):
    print(f"The parameter in the context is {get_config()['parameter']}")
print(f"The parameter outside the context is {get_config()['parameter']}")

# %%
# Register configuration and share it between parallel workers
# ------------------------------------------------------------
#
# When using parallel processing, you might want to share configuration between
# the main process (driver) and the workers (sub-processes or threads).
# joblib supports context managers that apply to each function call in the workers.
#
# You can pass these call contexts in two ways:
# 1. Register them in a registry
# 2. Pass them as a list of tuples to the :class:`~joblib.Parallel` class
#
# Let's first look at the registry. We can list the default call contexts
# that joblib provides:
from joblib import list_call_context_names

list_call_context_names()

# %%
# The default 'joblib' call context uses the :func:`~joblib.config_context` context
# manager we saw earlier. Let's see how this works with parallel processing.
#
# We define a function that returns the joblib configuration. When we pass this
# function to :class:`~joblib.Parallel`, each worker executes it, showing us
# the configuration in each worker.
from joblib import Parallel, delayed


def get_joblib_config_in_worker():
    return get_config()


result = Parallel(n_jobs=2)(delayed(get_joblib_config_in_worker)() for _ in range(2))
result

# %%
# We see that the configuration that we set earlier is passed to each worker as one
# would expect. Now, let's use an external context manager.
with config_context(parameter="another value"):
    result = Parallel(n_jobs=2)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
# We observe that the configuration that we set in the main process is thus propagated
# to each worker. Let's now unregister the default call context and repeat the same
# experiment.
from joblib import unregister_call_context

unregister_call_context("joblib")

# %%
result = Parallel(n_jobs=2)(delayed(get_joblib_config_in_worker)() for _ in range(2))
result

# %%
# We see that the configuration of the main process is not propagated to the workers.
# Let's try wrap the parallel processing in a context manager to see if anything
# changes.
with config_context(parameter="another value"):
    result = Parallel(n_jobs=2)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
# No difference. So we conclude that setting a configuration in the main process is not
# automatically propagated to the workers. We therefore can use the default joblib
# configuration and the fact that it is registered by default to share configuration
# between the main process and the workers.
#
# Now, let see how we can pass the context manager to the :class:`~joblib.Parallel`
# class.
call_context = [(config_context, get_config)]
result = Parallel(n_jobs=2, call_context=call_context)(
    delayed(get_joblib_config_in_worker)() for _ in range(2)
)
result

# %%
# We see that the configuration is now propagated to the workers. We can also make sure
# that an external context manager is also working.
with config_context(parameter="another value"):
    result = Parallel(n_jobs=2, call_context=call_context)(
        delayed(get_joblib_config_in_worker)() for _ in range(2)
    )
result

# %%
# Let's see how we could register a given call context instead of passing it as a list
# of tuples to the :class:`~joblib.Parallel` class.
from joblib import register_call_context

register_call_context("joblib", config_context, get_config)

# %%
list_call_context_names()

# %%
# So to register a call context, one needs to provide a name, a context manager factory
# and a function to retrieve the state to set the context. So we are back to the initial
# situation.
#
# Now we provide an example on how you can create a custom context manager and use it.
# Here, we have a function that raises warnings when the value is greater than 10.
import warnings


def func_that_warns(value):
    if value > 10:
        warnings.warn("This function is warning", RuntimeWarning)
    return value


# %%
# Calling the function in parallel will raise warnings.
result = Parallel(n_jobs=2)(delayed(func_that_warns)(i) for i in range(8, 12))

# %%
# Let's create a context manager to silence the warnings.
# ``contextlib.contextmanager`` helps create a context manager from a function.
# The function runs until the ``yield`` statement when entering the context.
# The worker then continues. After completion, it runs the teardown code
# (after the ``yield``).
from contextlib import contextmanager


@contextmanager
def silence_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        yield
        warnings.resetwarnings()


# %%
# We can now pass this context manager to the :class:`~joblib.Parallel` class. There is
# no function to retrieve the state to set the context. We therefore pass a lambda that
# does nothing.
call_context = [(silence_warnings, lambda: {})]
result = Parallel(n_jobs=2, call_context=call_context)(
    delayed(func_that_warns)(i) for i in range(8, 12)
)
result


# %%
# We see that the warnings are now silenced.
