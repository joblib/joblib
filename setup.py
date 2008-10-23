#!/usr/bin/env python

from distutils.core import setup

import joblib

# extra_setuptools_args is injected by the setupegg.py script, for
# running the setup with setuptools.
if not 'extra_setuptools_args' in globals():
    extra_setuptools_args = dict()


setup(name='joblib',
      version=joblib.__version__,
      description='Tools to run long-running scripts as jobs.',
      author='Gael Varoquaux',
      author_email='gael.varoquaux@normalesup.org',
      url='None',
      license='BSD',
      package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      **extra_setuptools_args)

