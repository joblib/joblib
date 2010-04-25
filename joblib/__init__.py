""" Joblib provides a set of tools for using long-running Python
functions as pipeline jobs:

  1. transparent disk-caching of the output values and lazy re-evaluation 

  2. logging and tracing of the execution

The original focus was on scientific-computing scripts, but any long-running
succession of operations can profit from the tools provided by joblib.

____

The latest documentation for `joblib` can be found on
http://packages.python.org/joblib/

Vision
--------

Joblib came out of long-running data-analysis Python scripts. The long
term vision is to provide tools for scientists to achieve better
reproducibility when running jobs, without changing the way numerical
code looks like. However, Joblib can also be used to provide a
light-weight make replacement.

The main problems identified are:

 1) **Lazy evaluation:** People need to rerun over and over the same
    script as it is tuned, but end up commenting out steps, or
    uncommenting steps, as they are needed, as they take long to run.

 2) **Persistance:** It is difficult to persist in an efficient way
    arbitrary objects containing large numpy arrays. In addition,
    hand-written persistence to disk does not link easily the file on
    disk to the corresponding Python object it was persists from in the
    script. This leads to people not a having a hard time resuming the
    job, eg after a crash and persistence getting in the way of work.

The approach take by Joblib to address these problems is not to build a
heavy framework and coerce user into using it (e.g. with an explicit
pipeline). It strives to leave your code and your flow control as
unmodified as possible.

The tools that have been identified and developped so far are:

  1) **Transparent and fast disk-caching of output value:** a make-like
     functionality for Python functions that works well with large numpy
     arrays. The goal is to separate operations in a set of steps with 
     well-defined inputs and outputs, that are saved and reran only if 
     necessary, by using standard Python functions::

        >>> from joblib import Memory
        >>> mem = Memory(cachedir='/tmp/joblib', debug=True)
        >>> import numpy as np
        >>> a = np.vander(np.arange(3))
        >>> square = mem.cache(np.square)
        >>> b = square(a)
        ________________________________________________________________________________
        [Memory] Calling square...
        square(array([[0, 0, 1],
               [1, 1, 1],
               [4, 2, 1]]))
        ___________________________________________________________square - 0.0s, 0.0min

        >>> c = square(a)
        >>> # The above call did not trigger an evaluation


  2) **Logging/tracing:** The functionalities described above will
     progressively acquire better logging mechanism to help track what
     has been ran, and capture I/O easily. In addition, Joblib will
     provide a few I/O primitives, to easily define define logging and
     display streams, and maybe provide a way of compiling a report. In
     the long run, we would like to be able to quickly inspect what has
     been run.

Status
-------

As stated on the project page, currently the project is in alpha quality. I am
testing heavily all the features, as I care more about robustness than having
plenty of features. On the other side, I expect to be playing with the API and
features for a while before I can figure out what is the right set of
functionalities to expose. 

The code is `hosted <https://launchpad.net/joblib>`_ on launchpad for the good reason that branching the project
and publishing it along-side my branch is dead-easy. 

.. 
    >>> import shutil ; shutil.rmtree('/tmp/joblib/')

"""

__version__ = '0.3.5'


from .memory import Memory
from .logger import PrintTime, Logger
from .hashing import hash
from .numpy_pickle import dump, load

