#!/usr/bin/env python

from distutils.core import setup
import sys

import joblib

# For some commands, use setuptools
if len(set(('develop', 'sdist', 'release', 'bdist', 'bdist_egg', 'bdist_dumb',
            'bdist_rpm', 'bdist_wheel', 'bdist_wininst', 'install_egg_info',
            'egg_info', 'easy_install', 'upload',
            )).intersection(sys.argv)) > 0:
    import setuptools

extra_setuptools_args = {}


if __name__ == '__main__':
    setup(name='joblib',
          version=joblib.__version__,
          author='Gael Varoquaux',
          author_email='gael.varoquaux@normalesup.org',
          url='https://joblib.readthedocs.io',
          description=("Lightweight pipelining: using Python functions "
                       "as pipeline jobs."),
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
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
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
          packages=['joblib', 'joblib.test', 'joblib.test.data',
                    'joblib.externals', 'joblib.externals.cloudpickle',
                    'joblib.externals.loky', 'joblib.externals.loky.backend'],
          **extra_setuptools_args)
