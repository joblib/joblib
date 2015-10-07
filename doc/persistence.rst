..
    For doctests:

    >>> from joblib.testing import warnings_to_stdout
    >>> warnings_to_stdout()

.. _persistence:

===========
Persistence
===========

.. currentmodule:: joblib.numpy_pickle

Usecase
=======

:func:`joblib.dump` and :func:`joblib.load` provide a replacement for
pickle to work efficiently on Python objects containing large data, in
particular large numpy arrays.

A simple example
================

First we create a temporary directory::

  >>> from tempfile import mkdtemp
  >>> savedir = mkdtemp()
  >>> import os
  >>> filename = os.path.join(savedir, 'test.pkl')

Then we create an object to be persisted::

  >>> import numpy as np
  >>> to_persist = [('a', [1, 2, 3]), ('b', np.arange(10))]

which we save into `savedir`::

  >>> import joblib
  >>> joblib.dump(to_persist, filename)  # doctest: +ELLIPSIS
  ['...test.pkl', '...test.pkl_01.npy']

We can then load the object from the file::

  >>> joblib.load(filename)
  [('a', [1, 2, 3]), ('b', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

.. note::

   As you can see from the output, joblib pickle tend to be spread
   across multiple files. More precisely, on top of the main joblib
   pickle file (passed into the `joblib.dump` function), for each
   numpy array that the persisted object contains, an auxiliary .npy
   file with the binary data of the array will be created. When moving
   joblib pickle files around, you will need to remember to keep all
   these files together.

Compressed joblib pickles
=========================

Setting the `compress` argument to `True` in :func:`joblib.dump` will allow to
save space on disk:

  >>> joblib.dump(to_persist, filename, compress=True)  # doctest: +ELLIPSIS
  ['.../test.pkl']

Another advantage it that it will create a single-file joblib pickle.

More details can be found in the :func:`joblib.dump` and
:func:`joblib.load` documentation.

Compatibility across python versions
------------------------------------

Compatibility of joblib pickles across python versions is not
supported. Note that this may appear to work when saving a pickle with
python 2 and loading it with python 3, for a very restricted set of
objects but relying on it is strongly discouraged.

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
