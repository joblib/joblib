"""
Joblib provides a set of tools for long running Python functions as
pipeline jobs:

  1. transparent disk caching of the output values and lazy re-evaluation 

  2. logging of the execution 

The original focus was on scientific-computing scripts, but any long-running
succession of functions can profit from the tools provided by joblib.

____

Joblib came out of long-running data-analysis Python scripts. The long
term vision is to provide tools for scientists to achieve better
reproducibility when running jobs. However, Joblib can also be used to
provide a light-weight make replacement.

The main problems identified are:

 1) Rerunning over and over the same script as it is tuned, but commenting
    out steps, or uncommenting steps, as they are needed, as they take
    long to run.

 2) Not ideal persistence model, too often hand-implemented by the
    scientist, which leads to people having a hard time resuming the job,
    eg after a crash.

The approach take by Joblib to address these problems is not to build a heavy
framework and coerce user into using it. It strives to build a set of
easy-to-use, light-weight tools that fit with the users's mind of running a
script, and not developing a library.

The tools that have been identified and developped so far are:

  1) A make-like functionality. The goal is to separate a script in a set
     of steps, with well-defined inputs and outputs, that can be saved
     and reran only if necessary. 

  2) The functionalities described above will progressively acquire
     better logging mechanism to help track what has been ran, and
     capture I/O easily. In addition, Joblib will provide a few I/O
     primitives, to easily define define logging and display streams,
     and maybe provide a way of compiling a report, probably with some
     graphics captured from pylab plots, or anything else (here arises to
     need to define an easy API for a visualization mechanism in addition
     to the one defined for persistence). In the long run, we would like to be
     able to quickly inspect what has been run, and visualize
     the results to be able to compare multiple runs. This would try to
     achieve a virtual lab-book. Moreover, combined with the persistence model,
     the lab-book would also have the data stored.

As stated on the project page, currently the project is in alpha quality. I am
testing heavily all the features, as I care more about robustness than having
plenty of features. On the other side, I expect to be playing with the API and
features for a while before I can figure out what is the right set of
functionalities to expose. 

The code is hosted on launchpad for the good reason that branching the project
and publishing it along-side my branch is dead-easy. I suspect that some of the
existing functionality (such as the make decorator) can already be useful.
"""

__version__ = '0.3a'


from .memory import Memory
from .logger import PrintTime, Logger
from .hashing import hash
from .numpy_pickle import dump, load

