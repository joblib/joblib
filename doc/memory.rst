..
    For doctests:

    >>> from joblib.testing import warnings_to_stdout
    >>> warnings_to_stdout()

.. _memory:

===========================================
On demand recomputing: the `Memory` class
===========================================

.. currentmodule:: joblib.memory

Use case
--------

The :class:`~joblib.Memory` class defines a context for lazy evaluation of
function, by putting the results in a store, by default using a disk, and not
re-running the function twice for the same arguments.

It works by explicitly saving the output to a file and it is designed to
work with non-hashable and potentially large input and output data types
such as numpy arrays.

A simple example:
~~~~~~~~~~~~~~~~~

  First, define the cache directory::

    >>> cachedir = 'your_cache_location_directory'

  Then, instantiate a memory context that uses this cache directory::

    >>> from joblib import Memory
    >>> memory = Memory(cachedir, verbose=0)

  After these initial steps, just decorate a function to cache its output in
  this context::

    >>> @memory.cache
    ... def f(x):
    ...     print('Running f(%s)' % x)
    ...     return x

  Calling this function twice with the same argument does not execute it the
  second time, the output is just reloaded from a pickle file in the cache
  directory::

    >>> print(f(1))
    Running f(1)
    1
    >>> print(f(1))
    1

  However, calling the function with a different parameter executes it and
  recomputes the output::

    >>> print(f(2))
    Running f(2)
    2

Comparison with `memoize`
~~~~~~~~~~~~~~~~~~~~~~~~~

The `memoize` decorator (https://code.activestate.com/recipes/52201/)
caches in memory all the inputs and outputs of a function call. It can
thus avoid running twice the same function, with a very small
overhead. However, it compares input objects with those in cache on each
call. As a result, for big objects there is a huge overhead. Moreover
this approach does not work with numpy arrays, or other objects subject
to non-significant fluctuations. Finally, using `memoize` with large
objects will consume all the memory, where with `Memory`, objects are
persisted to disk, using a persister optimized for speed and memory
usage (:func:`joblib.dump`).

In short, `memoize` is best suited for functions with "small" input and
output objects, whereas `Memory` is best suited for functions with complex
input and output objects, and aggressive persistence to disk.


Using with `numpy`
------------------

The original motivation behind the `Memory` context was to have a
memoize-like pattern on numpy arrays. `Memory` uses fast cryptographic
hashing of the input arguments to check if they have been computed.

An example
~~~~~~~~~~

  Define two functions: the first with a number as an argument,
  outputting an array, used by the second one. Both functions are decorated
  with :meth:`Memory.cache <joblib.Memory.cache>`::

    >>> import numpy as np

    >>> @memory.cache
    ... def g(x):
    ...     print('A long-running calculation, with parameter %s' % x)
    ...     return np.hamming(x)

    >>> @memory.cache
    ... def h(x):
    ...     print('A second long-running calculation, using g(x)')
    ...     return np.vander(x)

  If the function `h` is called with the array created by the same call to `g`,
  `h` is not re-run::

    >>> a = g(3)
    A long-running calculation, with parameter 3
    >>> a
    array([0.08, 1.  , 0.08])
    >>> g(3)
    array([0.08, 1.  , 0.08])
    >>> b = h(a)
    A second long-running calculation, using g(x)
    >>> b2 = h(a)
    >>> b2
    array([[0.0064, 0.08  , 1.    ],
           [1.    , 1.    , 1.    ],
           [0.0064, 0.08  , 1.    ]])
    >>> np.allclose(b, b2)
    True


Using memmapping
~~~~~~~~~~~~~~~~

Memmapping (memory mapping) speeds up cache looking when reloading large numpy
arrays::

    >>> cachedir2 = 'your_cachedir2_location'
    >>> memory2 = Memory(cachedir2, mmap_mode='r')
    >>> square = memory2.cache(np.square)
    >>> a = np.vander(np.arange(3)).astype(float)
    >>> square(a)
    ________________________________________________________________________________
    [Memory] Calling square...
    square(array([[0., 0., 1.],
           [1., 1., 1.],
           [4., 2., 1.]]))
    ___________________________________________________________square - ...min
    memmap([[ 0.,  0.,  1.],
            [ 1.,  1.,  1.],
            [16.,  4.,  1.]])

.. note::

    Notice the debug mode used in the above example. It is useful for
    tracing of what is being reexecuted, and where the time is spent.

If the `square` function is called with the same input argument, its
return value is loaded from the disk using memmapping::

    >>> res = square(a)
    >>> print(repr(res))
    memmap([[ 0.,  0.,  1.],
            [ 1.,  1.,  1.],
            [16.,  4.,  1.]])

..

 The memmap file must be closed to avoid file locking on Windows; closing
 numpy.memmap objects is done with del, which flushes changes to the disk

    >>> del res

.. note::

   If the memory mapping mode used was 'r', as in the above example, the
   array will be read only, and will be impossible to modified in place.

   On the other hand, using 'r+' or 'w+' will enable modification of the
   array, but will propagate these modification to the disk, which will
   corrupt the cache. If you want modification of the array in memory, we
   suggest you use the 'c' mode: copy on write.


Shelving: using references to cached values
-------------------------------------------

In some cases, it can be useful to get a reference to the cached
result, instead of having the result itself. A typical example of this
is when a lot of large numpy arrays must be dispatched across several
workers: instead of sending the data themselves over the network, send
a reference to the joblib cache, and let the workers read the data
from a network filesystem, potentially taking advantage of some
system-level caching too.

Getting a reference to the cache can be done using the
`call_and_shelve` method on the wrapped function::

    >>> result = g.call_and_shelve(4)
    A long-running calculation, with parameter 4
    >>> result  #doctest: +ELLIPSIS
    MemorizedResult(location="...", func="...g...", args_id="...")

Once computed, the output of `g` is stored on disk, and deleted from
memory. Reading the associated value can then be performed with the
`get` method::

    >>> result.get()
    array([0.08, 0.77, 0.77, 0.08])

The cache for this particular value can be cleared using the `clear`
method. Its invocation causes the stored value to be erased from disk.
Any subsequent call to `get` will cause a `KeyError` exception to be
raised::

    >>> result.clear()
    >>> result.get()  #doctest: +SKIP
    Traceback (most recent call last):
    ...
    KeyError: 'Non-existing cache value (may have been cleared).\nFile ... does not exist'

A `MemorizedResult` instance contains all that is necessary to read
the cached value. It can be pickled for transmission or storage, and
the printed representation can even be copy-pasted to a different
python interpreter.

.. topic:: Shelving when cache is disabled

    In the case where caching is disabled (e.g.
    `Memory(None)`), the `call_and_shelve` method returns a
    `NotMemorizedResult` instance, that stores the full function
    output, instead of just a reference (since there is nothing to
    point to). All the above remains valid though, except for the
    copy-pasting feature.


Gotchas
-------

* **Across sessions, function cache is identified by the function's name**.
  Thus assigning the same name to different functions, their cache will
  override each-others (e.g. there are 'name collisions'), and unwanted re-run
  will happen::

    >>> @memory.cache
    ... def func(x):
    ...     print('Running func(%s)' % x)

    >>> func2 = func

    >>> @memory.cache
    ... def func(x):
    ...     print('Running a different func(%s)' % x)

  As long as the same session is used, there are no collisions (in joblib
  0.8 and above), although joblib does warn you that you are doing something
  dangerous::

    >>> func(1)
    Running a different func(1)

    >>> # FIXME: The next line should create a JolibCollisionWarning but does not
    >>> # memory.rst:0: JobLibCollisionWarning: Possible name collisions between functions 'func' (<doctest memory.rst>:...) and 'func' (<doctest memory.rst>:...)
    >>> func2(1)  #doctest: +ELLIPSIS
    Running func(1)

    >>> func(1) # No recomputation so far
    >>> func2(1) # No recomputation so far

  ..
     Empty the in-memory cache to simulate exiting and reloading the
     interpreter

     >>> import joblib.memory
     >>> joblib.memory._FUNCTION_HASHES.clear()

  But suppose the interpreter is exited and then restarted, the cache will not
  be identified properly, and the functions will be rerun::

    >>> # FIXME: The next line will should create a JoblibCollisionWarning but does not. Also it is skipped because it does not produce any output
    >>> # memory.rst:0: JobLibCollisionWarning: Possible name collisions between functions 'func' (<doctest memory.rst>:...) and 'func' (<doctest memory.rst>:...)
    >>> func(1) #doctest: +ELLIPSIS +SKIP
    Running a different func(1)
    >>> func2(1)  #doctest: +ELLIPSIS +SKIP
    Running func(1)

  As long as the same session is used, there are no needless
  recomputation::

    >>> func(1) # No recomputation now
    >>> func2(1) # No recomputation now

* **lambda functions**

  Beware that with Python 2.7 lambda functions cannot be separated out::

    >>> def my_print(x):
    ...     print(x)

    >>> f = memory.cache(lambda : my_print(1))
    >>> g = memory.cache(lambda : my_print(2))

    >>> f()
    1
    >>> f()
    >>> g() # doctest: +SKIP
    memory.rst:0: JobLibCollisionWarning: Cannot detect name collisions for function '<lambda>'
    2
    >>> g() # doctest: +SKIP
    >>> f() # doctest: +SKIP
    1

* **memory cannot be used on some complex objects**, e.g. a callable
  object with a `__call__` method.

  However, it works on numpy ufuncs::

    >>> sin = memory.cache(np.sin)
    >>> print(sin(0))
    0.0

* **caching methods: memory is designed for pure functions and it is
  not recommended to use it for methods**. If one wants to use cache
  inside a class the recommended pattern is to cache a pure function
  and use the cached function inside your class, i.e. something like
  this::

    @memory.cache
    def compute_func(arg1, arg2, arg3):
        # long computation
        return result


    class Foo(object):
        def __init__(self, args):
            self.data = None

        def compute(self):
            self.data = compute_func(self.arg1, self.arg2, 40)


  Using ``Memory`` for methods is not recommended and has some caveats
  that make it very fragile from a maintenance point of view because
  it is very easy to forget about these caveats when a software
  evolves. If this cannot be avoided (we would be interested about
  your use case by the way), here are a few known caveats:

  1. a method cannot be decorated at class definition,
     because when the class is instantiated, the first argument (self) is
     *bound*, and no longer accessible to the `Memory` object. The
     following code won't work::

       class Foo(object):

           @memory.cache  # WRONG
           def method(self, args):
               pass

     The right way to do this is to decorate at instantiation time::

       class Foo(object):

           def __init__(self, args):
               self.method = memory.cache(self.method)

           def method(self, ...):
               pass

  2. The cached method will have ``self`` as one of its
     arguments. That means that the result will be recomputed if
     anything with ``self`` changes. For example if ``self.attr`` has
     changed calling ``self.method`` will recompute the result even if
     ``self.method`` does not use ``self.attr`` in its body. Another
     example is changing ``self`` inside the body of
     ``self.method``. The consequence is that ``self.method`` will
     create cache that will not be reused in subsequent calls. To
     alleviate these problems and if you *know* that the result of
     ``self.method`` does not depend on ``self`` you can use
     ``self.method = memory.cache(self.method, ignore=['self'])``.

* **joblib cache entries may be invalidated after environment updates**.
  Values returned by :func:`joblib.hash` are not guaranteed to stay
  constant across ``joblib`` versions. This means that **all** entries of a
  :class:`Memory` cache can get invalidated when upgrading ``joblib``.
  Invalidation can also happen when upgrading a third party library (such as
  ``numpy``): in such a case, only the cached function calls with parameters
  that are constructs (or contain references to constructs) defined in the
  upgraded library should potentially be invalidated after the upgrade.


Ignoring some arguments
-----------------------

It may be useful not to recalculate a function when certain arguments
change, for instance a debug flag. :class:`Memory` provides the ``ignore``
list::

    >>> @memory.cache(ignore=['debug'])
    ... def my_func(x, debug=True):
    ...	    print('Called with x = %s' % x)
    >>> my_func(0)
    Called with x = 0
    >>> my_func(0, debug=False)
    >>> my_func(0, debug=True)
    >>> # my_func was not reevaluated


Custom cache validation
-----------------------

In some cases, external factors can invalidate the cached results and
one wants to have more control on whether to reuse a result or not.

This is for instance the case if the results depends on database records
that change over time: a small delay in the updates might be tolerable
but after a while, the results might be invalid.

One can have a finer control on the cache validity specifying a function
via ``cache_validation_callback`` in :meth:`~joblib.Memory.cache`. For
instance, one can only cache results that take more than 1s to be computed.

    >>> import time
    >>> def cache_validation_cb(metadata):
    ...     # Only retrieve cached results for calls that take more than 1s
    ...     return metadata['duration'] > 1

    >>> @memory.cache(cache_validation_callback=cache_validation_cb)
    ... def my_func(delay=0):
    ...     time.sleep(delay)
    ...	    print(f'Called with {delay}s delay')

    >>> my_func()
    Called with 0s delay
    >>> my_func(1.1)
    Called with 1.1s delay
    >>> my_func(1.1)  # This result is retrieved from cache
    >>> my_func()  # This one is not and the call is repeated
    Called with 0s delay

``cache_validation_cb`` will be called with a single argument containing
the metadata of the cached call as a dictionary containing the following
keys:

  - ``duration``: the duration of the function call,
  - ``time``: the timestamp when the cache called has been recorded
  - ``input_args``: a dictionary of keywords arguments for the cached function call.

Note a validity duration for cached results can be defined via
:func:`joblib.expires_after` by providing similar with arguments similar to the
ones of a ``datetime.timedelta``:

    >>> from joblib import expires_after
    >>> @memory.cache(cache_validation_callback=expires_after(seconds=0.5))
    ... def my_func():
    ...	    print(f'Function run')
    >>> my_func()
    Function run
    >>> my_func()
    >>> time.sleep(0.5)
    >>> my_func()
    Function run


.. _memory_reference:

Reference documentation of the :class:`~joblib.Memory` class
------------------------------------------------------------

.. autoclass:: joblib.Memory
    :members: __init__, cache, eval, clear, reduce_size, format
    :no-inherited-members:
    :noindex:

Useful methods of decorated functions
-------------------------------------

Functions decorated by :meth:`Memory.cache <joblib.Memory.cache>` are
:class:`MemorizedFunc`
objects that, in addition of behaving like normal functions, expose
methods useful for cache exploration and management. For example, you can
use :meth:`func.check_call_in_cache <MemorizedFunc.check_call_in_cache>` to
check if a cache hit will occur for a decorated ``func`` given a set of inputs
without actually needing to call the function itself::

    >>> @memory.cache
    ... def func(x):
    ...     print('Running func(%s)' % x)
    ...     return x
    >>> type(func)
    <class 'joblib.memory.MemorizedFunc'>
    >>> func(1)
    Running func(1)
    1
    >>> func.check_call_in_cache(1)  # cache hit
    True
    >>> func.check_call_in_cache(2)  # cache miss
    False

.. autoclass:: MemorizedFunc
    :members: __init__, call, clear, check_call_in_cache


..
 Let us not forget to clean our cache dir once we are finished::

    >>> import shutil
    >>> try:
    ...     shutil.rmtree(cachedir)
    ...     shutil.rmtree(cachedir2)
    ... except OSError:
    ...     pass  # this can sometimes fail under Windows


Helper Reference
~~~~~~~~~~~~~~~~

.. autofunction:: joblib.expires_after
