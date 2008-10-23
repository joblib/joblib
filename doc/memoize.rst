
..
    Right now this is only a doctest module, but it shouldn't stay that
    way.

=========================
The `memoize` decorator
=========================

Some demos:
    
  We define a function caclulating the Fibonacci series in a recursive
  way. First without the `memoize` decorator::

    >>> def fibonacci(n):
    ...   "Return the n-th element of the Fibonacci series."
    ...   print('fibonacci(%i) called' % n)
    ...   if n < 3:
    ...      return 1
    ...   return fibonacci(n-1) + fibonacci(n-2)

  If we print the first four Fibonacci numbers using this function, we
  can see that, due to its recursive nature, it is called many times::

    >>> print([fibonacci(i) for i in xrange(1, 5)])
    fibonacci(1) called
    fibonacci(2) called
    fibonacci(3) called
    fibonacci(2) called
    fibonacci(1) called
    fibonacci(4) called
    fibonacci(3) called
    fibonacci(2) called
    fibonacci(1) called
    fibonacci(2) called
    [1, 1, 2, 3]

  We can use use the `memoize` decorator provided by `joblib` to cache
  the results to each call to `fibonacci`, thus reducing by a large
  amount the number of calls to the function.

  First let us create a temporary directory to store the cache, so that
  it is persistent from one Python session to another::

    >>> from tempfile import mkdtemp
    >>> cachedir = mkdtemp()

  Then we decorate the `fibonacci` function::

    >>> from joblib.memoize import memoize
    >>> fibonacci = memoize(cachedir=cachedir)(fibonacci)

  The same call to `fibonacci` now yields::

    >>> print([fibonacci(i) for i in xrange(1, 5)])
    fibonacci(1) called
    fibonacci(2) called
    fibonacci(3) called
    fibonacci(4) called
    [1, 1, 2, 3]

  For large numbers, the number of calls to the `fibonacci` function is
  greatly reduced (the algoritmic cost of calculating the `n` first
  numbers goes from 2^n to n), to the point that for n=100, printing the 
  list is unbearably slow without `memoize` but lightning fast with.

  Finally, we need to clean our cache directory::

    >>> import shutil
    >>> shutil.rmtree(cachedir)

