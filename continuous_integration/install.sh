#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

create_new_conda_env() {
    conda update --yes conda
    TO_INSTALL="python=$PYTHON_VERSION pip pytest $EXTRA_CONDA_PACKAGES"
    conda create -n testenv --yes $TO_INSTALL
    source activate testenv
}

create_new_pypy3_env() {
    PYPY_FOLDER="pypy3-v6.0.0-linux64"
    wget https://bitbucket.org/pypy/pypy/downloads/$PYPY_FOLDER.tar.bz2
    tar xvf $PYPY_FOLDER.tar.bz2
    $PYPY_FOLDER/bin/pypy3 -m venv pypy3
    source pypy3/bin/activate
    pip install -U pip pytest
}

if [[ "$PYTHON_VERSION" == "pypy3" ]]; then
    create_new_pypy3_env
else
    create_new_conda_env
fi

# Install py.test timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout"

if [ -n "$NUMPY_VERSION" ]; then
    # We want to ensure no memory copies are performed only when numpy is
    # installed. This also ensures that we don't keep a strong dependency on
    # memory_profiler. We also want to ensure that joblib can be used with and
    # without lz4 compressor package installed.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES memory_profiler"
    if [ "$NO_LZ4" != "true" ]; then
        PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES lz4"
    fi
fi

if [[ "$COVERAGE" == "true" ]]; then
    # TODO: unpin when https://github.com/nedbat/coveragepy/issues/883 is fixed
    # Weird issues with recent version of coverage: unpin when not causing
    # pytest to raise INTERNALERROR exceptions.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES coverage==4.5.4 pytest-cov codecov"
fi

if [[ "2.7 3.4 pypy3" != *"$PYTHON_VERSION"* ]]; then
    # threadpoolctl is only available for python 3.5+.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES threadpoolctl"
fi

pip install $PIP_INSTALL_PACKAGES


if [[ "$CYTHON" == "true" ]]; then
    pip install cython
    cd joblib/test/_openmp_test_helper
    python setup.py build_ext -i
    cd ../../..
fi

pip install -v .
