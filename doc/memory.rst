==============================
The `Memory` class 
==============================

.. currentmodule:: joblib.memory

Usecase
--------

The `Memory` class defines a context for lazy evaluation of function, by
storing the results to the disk, and not rerunning the function twice for
the same arguments.

You can use it as a context, with its `eval` method:

.. automethod:: Memory.eval

or decorate functions with the `cache` method:

.. automethod:: Memory.cache

It works by explicitely saving the output to a file and it is designed to
work with non-hashable and potentially large input and output data types
such as numpy arrays. 

A simple example:
~~~~~~~~~~~~~~~~~

  First we create a temporary directory, for the cache::

    >>> from tempfile import mkdtemp
    >>> cachedir = mkdtemp()

    >>> from joblib import Memory

  We can then instance a memory context, using this cache directory::

    >>> memory = Memory(cachedir=cachedir)

  Then we can decorator a function to be cached in this context::

    >>> @memory.cache
    ... def f(x):
    ...     print 'Running f(%s)' % x
    ...     return x

  When we call this function twice with the same argument, it does not
  get executed the second time, an the output is loaded from the pickle
  file::

    >>> print f(1)
    Running f(1)
    1
    >>> print f(1)
    1

  However, when we call it a third time, with a different argument, the
  output gets recomputed::

    >>> print f(2)
    Running f(2)
    2

Comparison with `memoize`
~~~~~~~~~~~~~~~~~~~~~~~~~

The `memoize` decorator (http://code.activestate.com/recipes/52201/)
caches in memory all the inputs and outputs of a function call. It can
thus avoid running twice the same function, but with a very small
overhead. However, it compares input objects with those in cache on each
call. As a result, for big objects there is a huge overhead. More over
this approach does not work with numpy arrays, or other objects subject
to non-significant fluctuations. Finally, using `memoize` with large
object will consume all the memory.

In short, `memoize` is best suited for functions with "small" input and
output objects, whereas `Memory` is best suited for functions with complex
input and output objects, and agressive persistence to the disk.

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
    ... def f(x):
    ...     print 'A long-running calculation, with parameter', x
    ...     return np.hamming(x)

    >>> @memory.cache
    ... def g(x):
    ...     print 'A second long-running calculation, using f(x)'
    ...     return np.vander(x)

  If we call the function g with the array created by the same call to f,
  g is not re-run::

    >>> a = f(3)
    A long-running calculation, with parameter 3
    >>> a
    array([ 0.08,  1.  ,  0.08])
    >>> f(3)
    array([ 0.08,  1.  ,  0.08])
    >>> b = g(a)
    A second long-running calculation, using f(x)
    >>> b2 = g(a)
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

    >>> memory2 = Memory(cachedir=cachedir, mmap_mode='r')
    >>> square = memory2.cache(np.square)
    >>> a = np.vander(np.arange(3))
    >>> square(a)
    array([[ 0,  0,  1],
           [ 1,  1,  1],
           [16,  4,  1]])

If the `square` function is called with the same input argument, its
return value is loaded from the disk using memmapping::

    >>> square(a)
    memmap([[ 0,  0,  1],
           [ 1,  1,  1],
           [16,  4,  1]])

   
.. note::

   If the memory mapping mode used was 'r', as in the above example, the
   array will be read only, and will be impossible to modified in place.

   On the other hand, using 'r+' or 'w+' will enable modification of the
   array, but will propagate these modification to the disk, which will
   corrupt the cache. If you want modification of the array in memory, we
   suggest you use the 'c' mode: copy on write.


.. warning::

   Because in the first run the array is a plain ndarray, and in the
   second run the array is a memmap, you can have side effects of using
   the `Memory`, especially when using `mmap_mode='r'` as the array is
   writable in the first run, and not the second.

Gotchas
--------

* **Function cache is identified by the function's name**. Thus if you have 
  the same name to different functions, their cache will override each-others, 
  and you well get unwanted re-run::

    >>> @memory.cache
    ... def f(x):
    ...     print 'Running f(%s)' % x

    >>> g = f

    >>> @memory.cache
    ... def f(x):
    ...     print 'Running a different f(%s)' % x

    >>> f(1)
    Running a different f(1)
    >>> g(1)
    Running f(1)
    >>> f(1)
    Running a different f(1)
    >>> g(1)
    Running f(1)

  Beware that all lambda functions have the same name::

    >>> def my_print(x):
    ...     print x

    >>> f = memory.cache(lambda : my_print(1))
    >>> g = memory.cache(lambda : my_print(2))
    
    >>> f()
    1
    >>> f()
    >>> g()
    2
    >>> g()
    >>> f()
    1

..  
  Thus to use lambda functions reliably, you have to specify the name
  used for caching::
  
  FIXME
 
 #   >>> f = make(func=lambda : my_print(1), cachedir=cachedir, name='f')
 #   >>> g = make(func=lambda : my_print(2), cachedir=cachedir, name='g')
 #   
 #   >>> f()
 #   1
 #   >>> g()
 #   2
 #   >>> f()

* **make cannot be used on complex objects**, eg a callable
  object with a `__call__` method.

  Howevers, it works on numpy ufuncs::

    >>> sin = memory.cache(np.sin)
    >>> print sin(0)
    0.0

..
  FIXME: Check the above


.. _`reference chapter`: http://www.informit.com/articles/article.aspx?p=453682 

Let us not forget to clean our cache dir once we are finished::

    >>> import shutil
    >>> shutil.rmtree(cachedir)


Reference documentation of the `Memory` class
----------------------------------------------

.. autoclass:: Memory
    :members: __init__, cache, eval, clear


