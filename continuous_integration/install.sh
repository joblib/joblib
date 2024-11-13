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
    to_install="python=$PYTHON_VERSION pip pytest $EXTRA_CONDA_PACKAGES"
    conda create -n testenv --yes -c conda-forge $to_install
    conda activate testenv
}

create_new_pypy3_env() {
    pypy_folder="$PYTHON_VERSION-v$PYPY_VERSION-linux64"
    wget https://downloads.python.org/pypy/$pypy_folder.tar.bz2
    tar xvf $pypy_folder.tar.bz2
    $pypy_folder/bin/pypy3 -m venv pypy3
    source pypy3/bin/activate
    pip install -U pip pytest
}

create_new_free_threaded_env() {
    sudo apt-get -yq update
    sudo apt-get install -yq ccache
    sudo apt-get install -yq software-properties-common
    sudo add-apt-repository --yes ppa:deadsnakes/nightly
    sudo apt-get update -yq
    sudo apt-get install -yq --no-install-recommends python3.13-dev python3.13-venv python3.13-nogil

    python3.13t -m venv testenv
    source testenv/bin/activate
}

if [[ "$PYTHON_VERSION" == pypy3* ]]; then
    create_new_pypy3_env
elif [[ "$PYTHON_VERSION" == free-threaded* ]]; then
     create_new_free_threaded_env
else
    create_new_conda_env
fi

# Install pytest timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout pytest-asyncio==0.21.1 threadpoolctl"

if [ "$NO_NUMPY" != "true" ]; then
    # We want to ensure no memory copies are performed only when numpy is
    # installed. This also ensures that we don't keep a strong dependency on
    # memory_profiler.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES memory_profiler numpy"
    # We also want to ensure that joblib can be used with and
    # without lz4 compressor package installed. Note that using `lz4` with
    # PyPy leads to segfaults, so we skip it in that case.
    if [ "$NO_LZ4" != "true" ] & [ "$PYTHON_VERSION" != pypy3* ]; then
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
