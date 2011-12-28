#!/usr/bin/env python

from distutils.core import setup
import os
import sys
import shutil

# Python 3 compatibility
local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
# python 3 compatibility stuff.
# Simplified version of scipy strategy: copy files into
# build/py3k, and patch them using lib2to3.
if sys.version_info[0] == 3:
    try:
        import lib2to3cache
    except ImportError:
        pass
    py3k_path = os.path.join(local_path, 'build', 'py3k')
    if os.path.exists(py3k_path):
        shutil.rmtree(py3k_path)
    print("Copying source tree into build/py3k for 2to3 transformation"
          "...")
    shutil.copytree(os.path.join(local_path, 'joblib'),
                    os.path.join(py3k_path, 'joblib'))
    import lib2to3.main
    from io import StringIO
    print("Converting to Python3 via 2to3...")
    try:
        sys.stdout = StringIO()  # supress noisy output
        res = lib2to3.main.main("lib2to3.fixes",
                                                 ['-x', 'import',
                                                  '-w', py3k_path])
    finally:
        sys.stdout = sys.__stdout__

    if res != 0:
        raise Exception('2to3 failed, exiting ...')

    os.chdir(py3k_path)
    sys.path.insert(0, py3k_path)

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
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
          'Programming Language :: Python :: 2.5',
          'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.0',
          'Programming Language :: Python :: 3.1',
          'Programming Language :: Python :: 3.2',
          'Topic :: Scientific/Engineering',
          'Topic :: Utilities',
          'Topic :: Software Development :: Libraries',
      ],
      platforms='any',
      #package_data={'joblib': ['joblib/*.rst'],},
      packages=['joblib', 'joblib.test'],
      **extra_setuptools_args)
