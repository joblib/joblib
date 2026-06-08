.. _persistence:

===========
Persistence
===========

.. currentmodule:: joblib.numpy_pickle

Use case
========

:func:`joblib.dump` and :func:`joblib.load` provide a replacement for
pickle to work efficiently on arbitrary Python objects containing large data,
in particular large numpy arrays.

.. warning::

  :func:`joblib.dump` and :func:`joblib.load` are based on the Python pickle
  serialization model, which means that arbitrary Python code can be executed
  when loading a serialized object with :func:`joblib.load`.

  :func:`joblib.load` should therefore never be used to load objects from an
  untrusted source or otherwise you will introduce a security vulnerability in
  your program.

.. note::

  As of Python 3.8 and numpy 1.16, pickle protocol 5 introduced in
  `PEP 574 <https://www.python.org/dev/peps/pep-0574/>`_ supports efficient
  serialization and de-serialization for large data buffers natively using
  the standard library::

      pickle.dump(large_object, fileobj, protocol=5)


A simple example
================

First create a temporary directory::

  >>> from tempfile import mkdtemp
  >>> savedir = mkdtemp()
  >>> import os
  >>> filename = os.path.join(savedir, 'test.joblib')

Then create an object to be persisted::

  >>> import numpy as np
  >>> to_persist = [('a', [1, 2, 3]), ('b', np.arange(10))]

which is saved into `filename`::

  >>> import joblib
  >>> joblib.dump(to_persist, filename)  # doctest: +ELLIPSIS
  ['...test.joblib']

The object can then be reloaded from the file::

  >>> joblib.load(filename)
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]


Persistence in file objects
===========================

Instead of filenames, :func:`joblib.dump` and :func:`joblib.load` functions
also accept file objects:

  >>> with open(filename, 'wb') as fo:  # doctest: +ELLIPSIS
  ...    joblib.dump(to_persist, fo)
  >>> with open(filename, 'rb') as fo:  # doctest: +ELLIPSIS
  ...    joblib.load(fo)
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]


Compressed joblib pickles
=========================

Setting the `compress` argument to `True` in :func:`joblib.dump` will allow to
save space on disk:

  >>> joblib.dump(to_persist, filename + '.compressed', compress=True)  # doctest: +ELLIPSIS
  ['...test.joblib.compressed']

If the filename extension corresponds to one of the supported compression
methods, the compressor will be used automatically:

  >>> joblib.dump(to_persist, filename + '.z')  # doctest: +ELLIPSIS
  ['...test.joblib.z']

By default, :func:`joblib.dump` uses the zlib compression method as it gives
the best tradeoff between speed and disk space. The other supported compression
methods are 'gzip', 'bz2', 'lzma' and 'xz':

  >>> # Dumping in a gzip compressed file using a compress level of 3.
  >>> joblib.dump(to_persist, filename + '.gz', compress=('gzip', 3))  # doctest: +ELLIPSIS
  ['...test.joblib.gz']
  >>> joblib.load(filename + '.gz')
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
  >>> joblib.dump(to_persist, filename + '.bz2', compress=('bz2', 3))  # doctest: +ELLIPSIS
  ['...test.joblib.bz2']
  >>> joblib.load(filename + '.bz2')
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

The ``compress`` parameter of the :func:`joblib.dump` function also accepts a
string corresponding to the name of the compressor used. When using this, the
default compression level is used by the compressor:

  >>> joblib.dump(to_persist, filename + '.gz', compress='gzip')  # doctest: +ELLIPSIS
  ['...test.joblib.gz']

.. note::

    Lzma and Xz compression methods are only available for python versions >= 3.3.

Compressor files provided by the python standard library can also be used to
compress pickle, e.g ``gzip.GzipFile``, ``bz2.BZ2File``, ``lzma.LZMAFile``:

  >>> # Dumping in a gzip.GzipFile object using a compression level of 3.
  >>> import gzip
  >>> with gzip.GzipFile(filename + '.gz', 'wb', compresslevel=3) as fo:  # doctest: +ELLIPSIS
  ...    joblib.dump(to_persist, fo)
  >>> with gzip.GzipFile(filename + '.gz', 'rb') as fo:  # doctest: +ELLIPSIS
  ...    joblib.load(fo)
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

If the ``lz4`` package is installed, this compression method is automatically
available with the dump function.

  >>> joblib.dump(to_persist, filename + '.lz4')  # doctest: +ELLIPSIS
  ['...test.joblib.lz4']
  >>> joblib.load(filename + '.lz4')
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

.. note::

    LZ4 compression is only available with python major versions >= 3

More details can be found in the :func:`joblib.dump` and
:func:`joblib.load` documentation.

Registering extra compressors
-----------------------------

Joblib provides :func:`joblib.register_compressor` in order to extend the list
of default compressors available.
To fit with Joblib internal implementation and features, such as
:func:`joblib.load` and :class:`joblib.Memory`, the registered compressor
should implement the Python file object interface.

Compatibility across python versions
------------------------------------

Compatibility of joblib pickles across python versions is not fully
supported. Note that, for a very restricted set of objects, this may appear to
work when saving a pickle with python 2 and loading it with python 3 but
relying on it is strongly discouraged.

If you are switching between python versions, you will need to save a
different joblib pickle for each python version.

Here are a few examples or exceptions:

  - Saving joblib pickle with python 2, trying to load it with python 3::

      Traceback (most recent call last):
        File "/home/lesteve/dev/joblib/joblib/numpy_pickle.py", line 453, in load
          obj = unpickler.load()
        File "/home/lesteve/miniconda3/lib/python3.4/pickle.py", line 1038, in load
          dispatch[key[0]](self)
        File "/home/lesteve/miniconda3/lib/python3.4/pickle.py", line 1176, in load_binstring
          self.append(self._decode_string(data))
        File "/home/lesteve/miniconda3/lib/python3.4/pickle.py", line 1158, in _decode_string
          return value.decode(self.encoding, self.errors)
      UnicodeDecodeError: 'ascii' codec can't decode byte 0x80 in position 1024: ordinal not in range(128)

      Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "/home/lesteve/dev/joblib/joblib/numpy_pickle.py", line 462, in load
          raise new_exc
        ValueError: You may be trying to read with python 3 a joblib pickle generated with python 2. This is not feature supported by joblib.


  - Saving joblib pickle with python 3, trying to load it with python 2::

      Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "joblib/numpy_pickle.py", line 453, in load
          obj = unpickler.load()
        File "/home/lesteve/miniconda3/envs/py27/lib/python2.7/pickle.py", line 858, in load
          dispatch[key](self)
        File "/home/lesteve/miniconda3/envs/py27/lib/python2.7/pickle.py", line 886, in load_proto
          raise ValueError, "unsupported pickle protocol: %d" % proto
      ValueError: unsupported pickle protocol: 3
