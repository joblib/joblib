#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -xe

CLOUDPICKLE="cloudpickle"
NUMPY="numpy"
DISTRIBUTED="distributed"

# Install pytest-timeout to fasten failure in deadlocking tests
PIP_INSTALL_PACKAGES="pytest-timeout pytest-asyncio threadpoolctl"

create_new_conda_env() {
    # Check python version
    if [[ $PYTHON_VERSION == free-threaded* ]]; then

        PYTHON_VERSION=${PYTHON_VERSION/free-threaded-/}
        EXTRA_CONDA_PACKAGES="$EXTRA_CONDA_PACKAGES python-freethreading"
        # pytest-run-parallel is used to run the same test in parallel on free-threaded
        # test runs, to catch thread-safety issues:
        PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES pytest-run-parallel"

    elif [[ $PYTHON_VERSION == "oldest_supported" ]]; then

        PYTHON_VERSION=$OLDEST_PYTHON_VERSION
        CLOUDPICKLE="cloudpickle==$OLDEST_CLOUDPICKLE_VERSION"
        NUMPY="numpy==$OLDEST_NUMPY_VERSION"
        DISTRIBUTED="distributed==$OLDEST_DISTRIBUTED_VERSION"

    elif [[ $PYTHON_VERSION == "latest_supported" ]]; then

        PYTHON_VERSION=$LATEST_PYTHON_VERSION

    fi

    # sklearn_tests requires scipy
    if [[ $SKLEARN_TESTS == "true" ]]; then
        EXTRA_CONDA_PACKAGES="$EXTRA_CONDA_PACKAGES scipy"
    fi

    to_install="python=$PYTHON_VERSION pip pytest $EXTRA_CONDA_PACKAGES"
    conda config --set solver libmamba
    conda create -n testenv --yes -c conda-forge $to_install
    conda activate testenv
}

create_new_conda_env

# Install cloudpickle with the correct version
PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES $CLOUDPICKLE"

if [ "$NO_NUMPY" != "true" ]; then
    # We want to ensure no memory copies are performed only when numpy is
    # installed. This also ensures that we don't keep a strong dependency on
    # memory_profiler.
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES memory_profiler $NUMPY"
    # We also want to ensure that joblib can be used with and
    # without lz4 compressor package installed.
    if [ "$NO_LZ4" != "true" ]; then
        PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES lz4"
    fi
fi

if [[ $USE_DISTRIBUTED == "true" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES $DISTRIBUTED"
fi

# We do not use coverage for sklearn_tests
if [[ "$COVERAGE" == "true" && $SKLEARN_TESTS != "true" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES coverage pytest-cov"
fi

# sklearn_tests requires cython
if [[ $CYTHON == "true" || $SKLEARN_TESTS == "true" ]]; then
    PIP_INSTALL_PACKAGES="$PIP_INSTALL_PACKAGES cython"
fi

pip install $PIP_INSTALL_PACKAGES

# Delete the LZMA module from the standard lib to make sure joblib has no
# hard dependency on it:
if [[ "$NO_LZMA" == "true" ]]; then
    LZMA_PATH=`python -c "import lzma; print(lzma.__file__)"`
    echo "Deleting $LZMA_PATH..."
    rm $LZMA_PATH
fi

if [[ $CYTHON == "true" && $SKLEARN_TESTS != "true" ]]; then
    pip install setuptools
    cd joblib/test/_openmp_test_helper
    python setup.py build_ext -i
    cd ../../..
fi

# Can't just install '.[test]' because, for example, we want some runs to omit
# NumPy:
pip install -v .

# Install the nightly build of scikit-learn after joblib
if [[ $SKLEARN_TESTS == "true" ]]; then
    pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
fi
