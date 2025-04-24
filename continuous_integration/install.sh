#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -xe

create_new_conda_env() {
    conda config --set solver libmamba
    if [[ "$PYTHON_VERSION" == free-threaded* ]]; then
        PYTHON_VERSION=${PYTHON_VERSION/free-threaded-/}
        EXTRA_CONDA_PACKAGES="python-freethreading $EXTRA_CONDA_PACKAGES"
    fi
    to_install="python=$PYTHON_VERSION pip pytest $EXTRA_CONDA_PACKAGES binutils"
    conda create -n testenv --yes -c conda-forge $to_install
    conda activate testenv
}

create_new_conda_env

# Install pytest timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout pytest-asyncio==0.21.1 threadpoolctl"

if [ "$NO_NUMPY" != "true" ]; then
    # We want to ensure no memory copies are performed only when numpy is
    # installed. This also ensures that we don't keep a strong dependency on
    # memory_profiler.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES memory_profiler numpy"
    # We also want to ensure that joblib can be used with and
    # without lz4 compressor package installed.
    if [ "$NO_LZ4" != "true" ]; then
        PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES lz4"
    fi
fi

if [[ "$COVERAGE" == "true" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES coverage pytest-cov"
fi

pip install $PIP_INSTALL_PACKAGES

if [[ "$NO_LZMA" == "true" ]]; then
    # Delete the LZMA module from the standard lib to make sure joblib has no
    # hard dependency on it:
    LZMA_PATH=`python -c "import lzma; print(lzma.__file__)"`
    echo "Deleting $LZMA_PATH..."
    rm $LZMA_PATH
fi


if [[ "$CYTHON" == "true" ]]; then
    pip install cython
    cd joblib/test/_openmp_test_helper
    python setup.py build_ext -i
    cd ../../..
fi

pip install -v .
