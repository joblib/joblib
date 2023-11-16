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
    conda update --yes conda conda-libmamba-solver
    conda config --set solver libmamba
    TO_INSTALL="python=$PYTHON_VERSION pip pytest $EXTRA_CONDA_PACKAGES"
    conda create -n testenv --yes -c conda-forge $TO_INSTALL
    source activate testenv
}

create_new_pypy3_env() {
    PYPY_FOLDER="$PYTHON_VERSION-v$PYPY_VERSION-linux64"
    wget https://downloads.python.org/pypy/$PYPY_FOLDER.tar.bz2
    tar xvf $PYPY_FOLDER.tar.bz2
    $PYPY_FOLDER/bin/pypy3 -m venv pypy3
    source pypy3/bin/activate
    pip install -U pip 'pytest'
}

if [[ "$PYTHON_VERSION" == pypy3* ]]; then
    create_new_pypy3_env
else
    create_new_conda_env
fi

# Install pytest timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout threadpoolctl"

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
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES coverage pytest-cov"
fi

pip install $PIP_INSTALL_PACKAGES

if [[ "$NO_LZMA" == "1" ]]; then
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
