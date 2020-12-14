#!/usr/bin/env python

from setuptools import setup
import pathlib
import joblib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.rst').read_text(encoding='utf-8')


setup(
    name='joblib',
    version=joblib.__version__,
    author='Gael Varoquaux',
    author_email='gael.varoquaux@normalesup.org',
    url='https://joblib.readthedocs.io',
    license='BSD',
    description="Lightweight pipelining with Python functions",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
        'Topic :: Software Development :: Libraries',
    ],
    platforms='any',
    package_data={
        'joblib.test': [
            'data/*.gz',
            'data/*.gzip',
            'data/*.bz2',
            'data/*.xz',
            'data/*.lzma',
            'data/*.pkl',
            'data/*.npy',
            'data/*.npy.z',
        ]
    },
    packages=[
        'joblib', 'joblib.test', 'joblib.test.data',
        'joblib.externals', 'joblib.externals.cloudpickle',
        'joblib.externals.loky', 'joblib.externals.loky.backend',
    ],
    python_requires='>=3.6',
)
