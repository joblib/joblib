#!/usr/bin/env python

from distutils.core import setup
import sys

import joblib


# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist_egg', 'bdist_rpm',
           'bdist', 'bdist_dumb', 'bdist_wininst', 'install_egg_info',
           'build_sphinx', 'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    from setupegg import extra_setuptools_args

# extra_setuptools_args is injected by the setupegg.py script, for
# running the setup with setuptools.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()

# if nose available, provide test command
try:
    from nose.commands import nosetests
    cmdclass = extra_setuptools_args.pop('cmdclass', {})
    cmdclass['test'] = nosetests
    cmdclass['nosetests'] = nosetests
    extra_setuptools_args['cmdclass'] = cmdclass
except ImportError:
    pass

setup(name='joblib',
      version=joblib.__version__,
      summary='Tools to use Python functions as pipeline jobs.',
      author='Gael Varoquaux',
      author_email='gael.varoquaux@normalesup.org',
      url='http://packages.python.org/joblib/',
      description="""
Lightweight pipelining: using Python functions as pipeline jobs.
""",
      long_description=joblib.__doc__,
      license='BSD',
      classifiers=[
          'Development Status :: 4 - Beta',
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
      #package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      **extra_setuptools_args)
