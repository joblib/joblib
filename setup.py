#!/usr/bin/env python

from distutils.core import setup
from glob import glob

setup(name='joblib',
      version='0.1a',
      description='Tools to run long-running scripts as jobs.',
      author='Gael Varoquaux',
      author_email='gael.varoquaux@normalesup.org',
      url='None',
      package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      # Setuptools specific stuff (does no harm in being here)
      tests_require=['nose', 'coverage'],
      test_suite='nose.collector',
      )

