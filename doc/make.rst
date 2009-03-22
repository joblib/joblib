==============================
The `make` function decorator
==============================

Usecase
--------

The `make` function decorator is for saving the output of a function, and
running the function again only if its code, or its input argument, have
changed. Unlike the `memoize` decorator, it works by explicitely saving
the output to a file with a specified format (thus potentialy useful for
later work), and it is designed to work with non-hashable input and
output data types such as numpy arrays. 

A simple example:
~~~~~~~~~~~~~~~~~

  First we create a temporary directory, for the cache::

    >>> from tempfile import mkdtemp
    >>> cachedir = mkdtemp()

    >>> from joblib.make import make, PickleFile

  Then we define our function, speicifying its cache directory, and that
  it persists its output using a pickle file in the chace directory::

    >>> @make(cachedir=cachedir, output=PickleFile(cachedir+'/f_output'))
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

The `memoize` decorator caches in memory all the inputs and outputs of a
function call. It can thus avoid running twice the same function, but
with a very small overhead. However, it compares input objects with those
in cache on each call. As a result, for big objects there is a huge
overhead. More over this approach does not work with numpy arrays, or
other objects subject to non-significant fluctuations. Finally, using
`memoize` with large object will consume all the memory, and even though
`memoize` can persist to disk, the resulting files are not easy to load
in different softwares.

In short, `memoize` is best suited for functions with "small" input and
output objects, whereas `make` is best suited for functions with complex
input and output objects.

Using with `numpy`
-------------------

The original motivation behind the `make` function decorator was to be
able to a memoize-like pattern on numpy arrays. The difficulty is that
numpy arrays cannot be well-hashed: it is computational expensive to do a
cache lookup, and, due to small numerical errors, many cache comparisons
fail for identical computations.

The time-stamp mechanism of `make` makes it robust to these problems. As
long as numpy arrays (or any complex mutable objects) are created through
a function decorated by `make`, the cache lookup will work:

An example
~~~~~~~~~~~

  We define two functions, the first with a number as an argument,
  outputting an array, used by the second one. We decorate both
  functions with `make`, persisting the output in numpy files::

    >>> import numpy as np
    >>> from joblib.make import NumpyFile

    >>> @make(cachedir=cachedir, output=NumpyFile(cachedir+'/f.npy'))
    ... def f(x):
    ...     print 'A long-running calculation, with parameter', x
    ...     return np.hamming(x)

    >>> @make(cachedir=cachedir, output=NumpyFile(cachedir+'/g.npy'))
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

  This works even if the input parameter to g is not the same object, as
  long as it comes from the same call to f::

    >>> a2 = f(3)
    >>> b3 = g(a2)
    >>> np.allclose(b, b3)
    True

  Note that `a` and `a2` are not the same object even though they are
  numerically equivalent::

    >>> a2 is a
    False
    >>> np.allclose(a2, a)
    True


`make` as a persistence model and lazy-re-evaluation execution engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gotchas
--------

* **Only the last result is cached**. As a consequence, if you call the same
  function with alternating values, it will be rerun::

    >>> @make(cachedir=cachedir, output=None)
    ... def f(x):
    ...     print 'Running f(%s)' % x

    >>> f(1)
    Running f(1)
    >>> f(2)
    Running f(2)
    >>> f(1)
    Running f(1)

  *Workaround*: You can define different function names, with different 
  persistence if needed::

    >>> def f(x):
    ...     print 'Running f(%s)' % x

    >>> def g(x):
    ...     return make(func=f, name=repr(x), cachedir=cachedir,
    ...                 output=None)(x)

    >>> g(1)
    Running f(1)
    >>> g(2)
    Running f(2)
    >>> g(1)
    
* **Function cache is identified by the function's name**. Thus if you have 
  the same name to different functions, their cache will override each-others, 
  and you well get unwanted re-run::

    >>> @make(cachedir=cachedir, output=None)
    ... def f(x):
    ...     print 'Running f(%s)' % x

    >>> g = f

    >>> @make(cachedir=cachedir, output=None)
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

    >>> f = make(func=lambda : my_print(1), cachedir=cachedir)
    >>> g = make(func=lambda : my_print(2), cachedir=cachedir)
    
    >>> f()
    1
    >>> g()
    2
    >>> f()
    1

  Thus to use lambda functions reliably, you have to specify the name
  used for caching::

    >>> f = make(func=lambda : my_print(1), cachedir=cachedir, name='f')
    >>> g = make(func=lambda : my_print(2), cachedir=cachedir, name='g')
    
    >>> f()
    1
    >>> g()
    2
    >>> f()

* **make cannot be used on objects more complex than a function**, eg an
  object with a `__call__` method.

* **make cannot track changes outside functions it decorates**.
  When tracking changes made to mutable objects (such as numpy arrays),
  `make` cannot track changes made out of functions it decorates::

    >>> @make(cachedir=cachedir, output=NumpyFile(cachedir+'/f.npy'))
    ... def f(x):
    ...     return np.array(x)

    >>> @make(cachedir=cachedir, output=NumpyFile(cachedir+'/g.npy'))
    ... def g(x):
    ...     print "Running g(%s)" % x
    ...     return x**2

    >>> a = f([1])
    >>> a
    array([1])
    >>> b = g(a)
    Running g([1])
    >>> a *= 2
    >>> b = g(a)
    >>> b
    array([1])

  This is why for more reliability, you should modify objects only in
  functions decorated by `make`: **do not break the chain of trust**.

..
  FIXME: I need to sort this out. I the latest changes seem to have made
  make more robust, and thus this obsolete.

..
    * **make tracks objects by identity, and not by name**.
    Between functions, the tracking of the objects is not
    made by name but by identity (if you don't understand this well, it
    might be worth reading the `reference chapter`_ on this, by David Beazley).
    As a result, reassigning to a variable will cause a rerun::
..
  .      >>> a = f([1])
  .      >>> b = g(a)
  .      >>> a = a.copy()
  .      >>> b = g(a)
  .      Running g([1])
  .      >>> b = g(a)
..
    Swapping object identities around, for the same names will also confuse
    `make`, but only if cannot keep track of the objects::
..
  .      >>> @make(cachedir=cachedir)
  .      ... def g(x, y):
  .      ...     print "Running g(%s, %s)" % (x, y)
..
  .      >>> a, b = f(1), f(2)
  .      >>> g(a, b)
  .      Running g(1, 2)
  .      >>> a, b = b, a
  .      >>> g(a, b)
  .      Running g(2, 1)
  .      >>> a, b
  .      (array(2), array(1))
..
  .      >>> a, b = f([1]), f([2])
  .      >>> g(a, b)
  .      Running g([1], [2])
  .      >>> a, b = b, a
  .      >>> g(a, b)
  .      Running g([2], [1])
  .      >>> a, b
  .      (array([2]), array([1]))
..
    In the above line, `g` thinks it is called with (array(1), array(2)).
    As a rule of thumb: **avoid mixing names and identities**.

* **Persisting can have side-effects**::

    >>> @make(cachedir=cachedir, output=NumpyFile(cachedir+'/f.npy'))
    ... def f(x):
    ...     return x
 
    >>> f(1)
    1
    >>> f(1)
    array(1)

  In the above lines, the returned value is saved as a numpy file, and
  thus restored in the second call as an array.


.. _`reference chapter`: http://www.informit.com/articles/article.aspx?p=453682 

Let us not forget to clean our cache dir once we are finished::

    >>> import shutil
    >>> shutil.rmtree(cachedir)

.. currentmodule:: joblib.make


Optional arguments to `make`
-----------------------------

.. autofunction:: make

Persistence objects
--------------------

Persistence objects provided with `make`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: PickleFile
    :members: __init__

.. autoclass:: NumpyFile
    :members: __init__

.. autoclass:: NiftiFile
    :members: __init__

.. autoclass:: MemMappedNiftiFile
    :members: __init__


Writing your own
~~~~~~~~~~~~~~~~~

A persistence object inherits from `joblib.make.Persister` and exposes a
`save` method, accepting the data as an argument, and a load method,
returning the data. The filename is usually set in the initializer.

How it works
-------------

Objects are tracked by their Python `id`. The `make` decorator stores
information on the history of each object in the cache for the different
functions, and reloads results only if objects given to a function are
newer or different than the objects used in the previous run, or it
cannot determine the history of these objects.

