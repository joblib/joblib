..
    For doctests:

    >>> from joblib.testing import warnings_to_stdout
    >>> warnings_to_stdout()

.. _memory:

===========================================
On demand recomputing: the `Memory` class
===========================================

.. currentmodule:: joblib.memory

Usecase
--------

The `Memory` class defines a context for lazy evaluation of function, by
storing the results to the disk, and not rerunning the function twice for
the same arguments.

..
 Commented out in favor of briefness

    You can use it as a context, with its `eval` method:

    .. automethod:: Memory.eval

    or decorate functions with the `cache` method:

    .. automethod:: Memory.cache

It works by explicitly saving the output to a file and it is designed to
work with non-hashable and potentially large input and output data types
such as numpy arrays.

A simple example:
~~~~~~~~~~~~~~~~~

  First we create a temporary directory, for the cache::

    >>> from tempfile import mkdtemp
    >>> cachedir = mkdtemp()

  We can instantiate a memory context, using this cache directory::

    >>> from joblib import Memory
    >>> memory = Memory(cachedir=cachedir, verbose=0)

  Then we can decorate a function to be cached in this context::

    >>> @memory.cache
    ... def f(x):
    ...     print('Running f(%s)' % x)
    ...     return x

  When we call this function twice with the same argument, it does not
  get executed the second time, and the output gets loaded from the pickle
  file::

    >>> print(f(1))
    Running f(1)
    1
    >>> print(f(1))
    1

  However, when we call it a third time, with a different argument, the
  output gets recomputed::

    >>> print(f(2))
    Running f(2)
    2

Comparison with `memoize`
~~~~~~~~~~~~~~~~~~~~~~~~~

The `memoize` decorator (http://code.activestate.com/recipes/52201/)
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
-------------------

The original motivation behind the `Memory` context was to be able to a
memoize-like pattern on numpy arrays. `Memory` uses fast cryptographic
hashing of the input arguments to check if they have been computed;

An example
~~~~~~~~~~~

  We define two functions, the first with a number as an argument,
  outputting an array, used by the second one. We decorate both
  functions with `Memory.cache`::

    >>> import numpy as np

    >>> @memory.cache
    ... def g(x):
    ...     print('A long-running calculation, with parameter %s' % x)
    ...     return np.hamming(x)

    >>> @memory.cache
    ... def h(x):
    ...     print('A second long-running calculation, using g(x)')
    ...     return np.vander(x)

  If we call the function h with the array created by the same call to g,
  h is not re-run::

    >>> a = g(3)
    A long-running calculation, with parameter 3
    >>> a
    array([ 0.08,  1.  ,  0.08])
    >>> g(3)
    array([ 0.08,  1.  ,  0.08])
    >>> b = h(a)
    A second long-running calculation, using g(x)
    >>> b2 = h(a)
    >>> b2
    array([[ 0.0064,  0.08  ,  1.    ],
           [ 1.    ,  1.    ,  1.    ],
           [ 0.0064,  0.08  ,  1.    ]])
    >>> np.allclose(b, b2)
    True


Using memmapping
~~~~~~~~~~~~~~~~

To speed up cache looking of large numpy arrays, you can load them
using memmapping (memory mapping)::

    >>> cachedir2 = mkdtemp()
    >>> memory2 = Memory(cachedir=cachedir2, mmap_mode='r')
    >>> square = memory2.cache(np.square)
    >>> a = np.vander(np.arange(3)).astype(np.float)
    >>> square(a)
    ________________________________________________________________________________
    [Memory] Calling square...
    square(array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]]))
    ___________________________________________________________square - 0.0s, 0.0min
    memmap([[  0.,   0.,   1.],
           [  1.,   1.,   1.],
           [ 16.,   4.,   1.]])

.. note::

    Notice the debug mode used in the above example. It is useful for
    tracing of what is being reexecuted, and where the time is spent.

If the `square` function is called with the same input argument, its
return value is loaded from the disk using memmapping::

    >>> res = square(a)
    >>> print(repr(res))
    memmap([[  0.,   0.,   1.],
           [  1.,   1.,   1.],
           [ 16.,   4.,   1.]])

..

 We need to close the memmap file to avoid file locking on Windows; closing
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
is when a lot of large numpy arrays must be dispatched accross several
workers: instead of sending the data themselves over the network, send
a reference to the joblib cache, and let the workers read the data
from a network filesystem, potentially taking advantage of some
system-level caching too.

Getting a reference to the cache can be done using the
`call_and_shelve` method on the wrapped function::

    >>> result = g.call_and_shelve(4)
    A long-running calculation, with parameter 4
    >>> result  #doctest: +ELLIPSIS 
    MemorizedResult(cachedir="...", func="g...", argument_hash="...")

Once computed, the output of `g` is stored on disk, and deleted from
memory. Reading the associated value can then be performed with the
`get` method::

    >>> result.get()
    array([ 0.08,  0.77,  0.77,  0.08])

The cache for this particular value can be cleared using the `clear`
method. Its invocation causes the stored value to be erased from disk.
Any subsequent call to `get` will cause a `KeyError` exception to be
raised::

    >>> result.clear()
    >>> result.get()  #doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    KeyError: 'Non-existing cache value (may have been cleared).\nFile ... does not exist'

A `MemorizedResult` instance contains all that is necessary to read
the cached value. It can be pickled for transmission or storage, and
the printed representation can even be copy-pasted to a different
python interpreter.

.. topic:: Shelving when cache is disabled

    In the case where caching is disabled (e.g.
    `Memory(cachedir=None)`), the `call_and_shelve` method returns a
    `NotMemorizedResult` instance, that stores the full function
    output, instead of just a reference (since there is nothing to
    point to). All the above remains valid though, except for the
    copy-pasting feature.


Gotchas
--------

* **Across sessions, function cache is identified by the function's name**.
  Thus if you assign the same name to different functions, their cache will
  override each-others (you have 'name collisions'), and you will get
  unwanted re-run::

    >>> @memory.cache
    ... def func(x):
    ...     print('Running func(%s)' % x)

    >>> func2 = func
    
    >>> @memory.cache
    ... def func(x):
    ...     print('Running a different func(%s)' % x)

  As long as you stay in the same session, there are no collisions (in joblib
  0.8 and above), altough joblib does warn you that you are doing something
  dangerous::

    >>> func(1)
    Running a different func(1)

    >>> func2(1)  #doctest: +ELLIPSIS
    memory.rst:0: JobLibCollisionWarning: Possible name collisions between functions 'func' (<doctest memory.rst>:...) and 'func' (<doctest memory.rst>:...)
    Running func(1)

    >>> func(1) # No recomputation so far
    >>> func2(1) # No recomputation so far

  ..
     Empty the in-memory cache to simulate exiting and reloading the
     interpreter

     >>> import joblib.memory
     >>> joblib.memory._FUNCTION_HASHES.clear()

  But suppose you exit the interpreter and restart it, the cache will not
  be identified properly, and the functions will be rerun::

    >>> func(1) #doctest: +ELLIPSIS
    memory.rst:0: JobLibCollisionWarning: Possible name collisions between functions 'func' (<doctest memory.rst>:...) and 'func' (<doctest memory.rst>:...)
    Running a different func(1)
    >>> func2(1)  #doctest: +ELLIPSIS
    Running func(1)

  As long as you stay in the same session, you are not getting needless
  recomputation::

    >>> func(1) # No recomputation now
    >>> func2(1) # No recomputation now

* **lambda functions**

  Beware that with Python 2.6 lambda functions cannot be separated out::

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

* **caching methods**: you cannot decorate a method at class definition,
  because when the class is instantiated, the first argument (self) is
  *bound*, and no longer accessible to the `Memory` object. The following
  code won't work::

    class Foo(object):

        @mem.cache  # WRONG
        def method(self, args):
	    pass

  The right way to do this is to decorate at instantiation time::

    class Foo(object):

        def __init__(self, args):
            self.method = mem.cache(self.method)

        def method(self, ...):
	    pass

Ignoring some arguments
------------------------

It may be useful not to recalculate a function when certain arguments
change, for instance a debug flag. `Memory` provides the `ignore` list::

    >>> @memory.cache(ignore=['debug'])
    ... def my_func(x, debug=True):
    ...	    print('Called with x = %s' % x)
    >>> my_func(0)
    Called with x = 0
    >>> my_func(0, debug=False)
    >>> my_func(0, debug=True)
    >>> # my_func was not reevaluated


.. _memory_reference:

Reference documentation of the `Memory` class
----------------------------------------------

.. autoclass:: Memory
    :members: __init__, cache, eval, clear

Useful methods of decorated functions
--------------------------------------

Function decorated by :meth:`Memory.cache` are :class:`MemorizedFunc`
objects that, in addition of behaving like normal functions, expose
methods useful for cache exploration and management.

.. autoclass:: MemorizedFunc
    :members: __init__, call, clear, format_signature, format_call,
	      get_output_dir, load_output


..
 Let us not forget to clean our cache dir once we are finished::

    >>> import shutil
    >>> try:
    ...     shutil.rmtree(cachedir)
    ...     shutil.rmtree(cachedir2)
    ... except OSError:
    ...     pass  # this can sometimes fail under Windows
