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

print_conda_requirements() {
    # Echo a conda requirement string for example
    # "pip python=2.7.3 scikit-learn=*". It has a hardcoded
    # list of possible packages to install and looks at _VERSION
    # environment variables to know whether to install a given package and
    # if yes which version to install. For example:
    #   - for numpy, NUMPY_VERSION is used
    #   - for scikit-learn, SCIKIT_LEARN_VERSION is used
    TO_INSTALL_ALWAYS="pip pytest"
    REQUIREMENTS="$TO_INSTALL_ALWAYS"
    TO_INSTALL_MAYBE="python numpy distributed flake8 scikit-learn"
    for PACKAGE in $TO_INSTALL_MAYBE; do
        # Capitalize package name and add _VERSION
        PACKAGE_VERSION_VARNAME="${PACKAGE^^}_VERSION"
        # replace - by _, needed for scikit-learn for example
        PACKAGE_VERSION_VARNAME="${PACKAGE_VERSION_VARNAME//-/_}"
        # dereference $PACKAGE_VERSION_VARNAME to figure out the
        # version to install
        PACKAGE_VERSION="${!PACKAGE_VERSION_VARNAME}"
        if [ -n "$PACKAGE_VERSION" ]; then
            REQUIREMENTS="$REQUIREMENTS $PACKAGE=$PACKAGE_VERSION"
        fi
    done
    echo $REQUIREMENTS
}

create_new_conda_env() {
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
         -O miniconda.sh
    MINICONDA_PREFIX=/home/travis/miniconda
    chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PREFIX
    export PATH=$MINICONDA_PREFIX/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    REQUIREMENTS=$(print_conda_requirements)
    echo "conda requirements string: $REQUIREMENTS"
    conda create -n testenv --yes -c conda-forge $REQUIREMENTS
    source activate testenv
}

create_new_pypy3_env() {
    deactivate
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
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES pytest-cov codecov"
fi

if [[ "2.7 3.4 pypy3" != *"$PYTHON_VERSION"* ]]; then
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
