#!/usr/bin/env python

from distutils.core import setup
import sys

import joblib


# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist', 'bdist_dumb', 'bdist_wininst', 'install_egg_info',
           'build_sphinx', 'egg_info', 'easy_install',
            )).intersection(sys.argv)) > 0:
    from setupegg import extra_setuptools_args

# extra_setuptools_args is injected by the setupegg.py script, for
# running the setup with setuptools.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


setup(name='joblib',
      version=joblib.__version__,
      summary='Tools to run long-running scripts as jobs.',
      author='Gael Varoquaux',
      author_email='gael.varoquaux@normalesup.org',
      url='https://launchpad.net/joblib',
      description="""
A set of tools to run Python scripts as jobs; namely: persistence and lazy
re-evaluation (between make and the memoize pattern), logging, and tools for
reusing scripts. 
""",
      long_description="""
Joblib provides a set of tools to run Python scripts as jobs; namely: 

  1. persistence and lazy re-evaluation (between make and the memoize pattern), 

  2. logging, 

  3. tools for reusing scripts. 

The original focus was on scientific-computing scripts, but any long-running
succession of operations can profit from the tools provided by joblib.

____

Joblib came out of long-running data-analysis Python scripts. The long term
vision is to provide tools for scientists to achieve better reproducibility
when running jobs. However strikes the set of functionalities needed is quite
general for long running jobs that build or compute something. For instance,
Joblib can be used to provide a light-weight make replacement.

The main problems identified are:

 1) Rerunning over and over the same script as it is tuned, but commenting
    out steps, or uncommenting steps, as they are needed, as they take
    long to run.

 2) Not ideal persistence model, too often hand-implemented by the
    scientist, which leads to people having a hard time resuming the job,
    eg after a crash.

 3) People writing scripts rather than reusable functions, as scientists
    perceive data processing, and sometimes simulations, as a sequential
    set of operations, but are not always able to identify reusable
    blocks. This leads to an incredible amount of code duplication, where
    a new processing job is often created by copying an old one and
    modifying it. From a software engineering point of view, this is a
    nightmare.

The approach take by Joblib to address these problems is not to build a heavy
framework and coerce user into using it. It strives to build a set of
easy-to-use, light-weight tools that fit with the users's mind of running a
script, and not developing a library.

The tools that have been identified and developped so far are:

  1) A make-like functionality. The goal is to separate a script in a set
     of steps, with well-defined inputs and outputs, that can be saved
     and reran only if necessary. This functionality help with solving
     problem 1), as well as problem 2) as it give a well-defined
     persistence model. In addition, identifying blocks can help with
     problem 3). This functionality is currently exposed as the make
     decorator (and a bit the memoize decorator). This is a fairly hard
     problem, but it seems that the current implementation is good-enough to 
     work on a set of problems. 

  2) A way of specifying default input parameters in scripts, that can
     afterwards be overridden using a glorified 'execfile'. This is a way
     to try and reuse standard processing steps written as script, thus
     addressing problem 3). This is exposed in the run_scripts.py module.

  3) The two functionalities described above will progressively acquire
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
""",
      license='BSD',
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Utilities',
      ],
      platforms='any',
      package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      **extra_setuptools_args)

