#!/usr/bin/env python

from distutils.core import setup
import sys

import joblib

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist_egg', 'bdist_rpm',
            'bdist', 'bdist_dumb', 'bdist_wininst', 'install_egg_info',
            'build_sphinx', 'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}

# if nose available, provide test command
try:
    from nose.commands import nosetests
    cmdclass = extra_setuptools_args.pop('cmdclass', {})
    cmdclass['test'] = nosetests
    cmdclass['nosetests'] = nosetests
    extra_setuptools_args['cmdclass'] = cmdclass
except ImportError:
    pass


if __name__ == '__main__':
    # Protect the call to the setup function to prevent a fork-bomb
    # when running the tests with:
    # python setup.py nosetests

    setup(name='joblib',
          version=joblib.__version__,
          author='Gael Varoquaux',
          author_email='gael.varoquaux@normalesup.org',
          url='http://pythonhosted.org/joblib/',
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
              'Programming Language :: Python :: 2.6',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Topic :: Scientific/Engineering',
              'Topic :: Utilities',
              'Topic :: Software Development :: Libraries',
          ],
          platforms='any',
          package_data={'joblib.test': ['data/*.gz',
                                        'data/*.gzip',
                                        'data/*.bz2',
                                        'data/*.xz',
                                        'data/*.lzma',
                                        'data/*.pkl',
                                        'data/*.npy',
                                        'data/*.npy.z']},
          packages=['joblib', 'joblib.test', 'joblib.test.data'],
          **extra_setuptools_args)
