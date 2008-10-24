#!/usr/bin/env python

from distutils.core import setup

import joblib

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
revaluation (between make and the memoize pattern), logging, and tools for
reusing scripts. 
""",
      long_description="""
Joblib provides a set of tools to run Python scripts as jobs; namely: 

  1. persistence and lazy revaluation (between make and the memoize pattern), 

  2. logging, 

  3. tools for reusing scripts. 

The original focus was on scientific-computing scripts, but any long-running
succession of operations can profit from the tools provided by joblib.

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

