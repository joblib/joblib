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


setup(name='joblib',
      version=joblib.__version__,
      summary='Tools to use Python functions as pipeline jobs.',
      author='Gael Varoquaux',
      author_email='gael.varoquaux@normalesup.org',
      url='https://launchpad.net/joblib',
      description="""
Tools to use Python functions as pipeline jobs:
output-value caching and execution logging.
""",
      long_description=joblib.__doc__,
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
      #package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      **extra_setuptools_args)

