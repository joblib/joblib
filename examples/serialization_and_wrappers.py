# -*- coding: utf-8 -*-
"""
Serialization of un-picklable objects
=====================================

This example highlights the options for tempering with joblib serialization
process.

"""

# Code source: Thomas Moreau
# License: BSD 3 clause

import sys
import time
import traceback
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_config
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects


###############################################################################
# First, define functions which cannot be pickled with the standard ``pickle``
# protocol. They cannot be serialized with ``pickle`` because they are defined
# in the ``__main__`` module. They can however be serialized with
# ``cloudpickle``. With the default behavior, ``loky`` is to use
# ``cloudpickle`` to serialize the objects that are sent to the workers.
#

def func_async(i, *args):
    return 2 * i


print(Parallel(n_jobs=2)(delayed(func_async)(21) for _ in range(1))[0])


###############################################################################
# For most use-cases, using ``cloudpickle`` is efficient enough. However, this
# solution can be very slow to serialize large python objects, such as dict or
# list, compared to the standard ``pickle`` serialization.
#

def func_async(i, *args):
    return 2 * i


# We have to pass an extra argument with a large list (or another large python
# object).
large_list = list(range(1000000))

t_start = time.time()
Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))
print("With loky backend and cloudpickle serialization: {:.3f}s"
      .format(time.time() - t_start))


###############################################################################
# If you are on a UNIX system, it is possible to fallback to the old
# ``multiprocessing`` backend, which can pickle interactively defined functions
# with the default pickle module, which is faster for such large objects.
#

import multiprocessing as mp
if mp.get_start_method() != "spawn":
    def func_async(i, *args):
        return 2 * i

    with parallel_config('multiprocessing'):
        t_start = time.time()
        Parallel(n_jobs=2)(
            delayed(func_async)(21, large_list) for _ in range(1))
        print("With multiprocessing backend and pickle serialization: {:.3f}s"
              .format(time.time() - t_start))


###############################################################################
# However, using ``fork`` to start new processes can cause violation of the
# POSIX specification and can have bad interaction with compiled extensions
# that use ``openmp``. Also, it is not possible to start processes with
# ``fork`` on windows where only ``spawn`` is available. The ``loky`` backend
# has been developed to mitigate these issues.
#
# To have fast pickling with ``loky``, it is possible to rely on ``pickle`` to
# serialize all communications between the main process and the workers with
# the ``loky`` backend. This can be done by setting the environment variable
# ``LOKY_PICKLER=pickle`` before the script is launched. Here we use an
# internal programmatic switch ``loky.set_loky_pickler`` for demonstration
# purposes but it has the same effect as setting ``LOKY_PICKLER``. Note that
# this switch should not be used as it has some side effects with the workers.
#

# Now set the `loky_pickler` to use the pickle serialization from stdlib. Here,
# we do not pass the desired function ``func_async`` as it is not picklable
# but it is replaced by ``id`` for demonstration purposes.

set_loky_pickler('pickle')
t_start = time.time()
Parallel(n_jobs=2)(delayed(id)(large_list) for _ in range(1))
print("With pickle serialization: {:.3f}s".format(time.time() - t_start))


###############################################################################
# However, the function and objects defined in ``__main__`` are not
# serializable anymore using ``pickle`` and it is not possible to call
# ``func_async`` using this pickler.
#

def func_async(i, *args):
    return 2 * i


try:
    Parallel(n_jobs=2)(delayed(func_async)(21, large_list) for _ in range(1))
except Exception:
    traceback.print_exc(file=sys.stdout)


###############################################################################
# To have both fast pickling, safe process creation and serialization of
# interactive functions, ``joblib`` provides a wrapper function
# :func:`~joblib.wrap_non_picklable_objects` to wrap the non-picklable function
# and indicate to the serialization process that this specific function should
# be serialized using ``cloudpickle``. This changes the serialization behavior
# only for this function and keeps using ``pickle`` for all other objects. The
# drawback of this solution is that it modifies the object. This should not
# cause many issues with functions but can have side effects with object
# instances.
#

@delayed
@wrap_non_picklable_objects
def func_async_wrapped(i, *args):
    return 2 * i


t_start = time.time()
Parallel(n_jobs=2)(func_async_wrapped(21, large_list) for _ in range(1))
print("With pickle from stdlib and wrapper: {:.3f}s"
      .format(time.time() - t_start))


###############################################################################
# The same wrapper can also be used for non-picklable classes. Note that the
# side effects of ``wrap_non_picklable_objects`` on objects can break magic
# methods such as ``__add__`` and can mess up the ``isinstance`` and
# ``issubclass`` functions. Some improvements will be considered if use-cases
# are reported.
#

# Reset the loky_pickler to avoid border effects with other examples in
# sphinx-gallery.
set_loky_pickler()
