"""Helper module to test OpenMP support on Continuous Integration
"""
import os
import sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


if sys.platform == "darwin":
    os.environ["CC"] = "gcc-4.9"
    os.environ["CXX"] = "g++-4.9"

if sys.platform != "win32":
    extra_compile_args = ["-ffast-math", "-fopenmp"]
    extra_link_args = ['-fopenmp']
else:
    extra_compile_args = ["/openmp"]
    extra_link_args = None

ext_modules = [
    Extension(
        "parallel_sum",
        ["parallel_sum.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name='_openmp_test_helper',
    ext_modules=cythonize(ext_modules),
)
