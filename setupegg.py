#!/usr/bin/env python
"""Wrapper to run setup.py using setuptools."""

import sys

# now, import setuptools and call the actual setup
import setuptools
execfile('setup.py', dict(__name__='__main__'))

# clean up the junk left around by setuptools
if "develop" not in sys.argv:
    os.system('rm -rf joblib.egg-info build')
