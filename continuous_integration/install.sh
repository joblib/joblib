#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# License: 3-clause BSD

set -xe

conda config --set solver libmamba

if [[ $PYTHON_VERSION == free-threaded* ]]; then
    PYTHON_VERSION=${PYTHON_VERSION/free-threaded-/}
    EXTRA_CONDA_PACKAGES="python-freethreading pip $EXTRA_CONDA_PACKAGES"
    FREE_THREADED="true"
fi

to_install="python=$PYTHON_VERSION \
    pytest pytest-timeout pytest-asyncio \
    $EXTRA_CONDA_PACKAGES"

if [ "$NO_NUMPY" != "true" ]; then
    # We want to ensure no memory copies are performed only when numpy is
    # installed. This also ensures that we don't keep a strong dependency on
    # memory_profiler.
    to_install="$to_install numpy memory_profiler"

    # We want to test threadpool limitations only when numpy is intalled
    # and multiprocessing is used.
    if [ "$JOBLIB_MULTIPROCESSING" != "0" ]; then
        to_install="$to_install threadpoolctl"
    fi
fi

# We also want to ensure that joblib can be used with and
# without lz4 compressor package installed.
if [[ $NO_LZ4 != "true" && $FREE_THREADED != "true" ]]; then
    to_install="$to_install lz4"
fi

if [[ $CYTHON == "true" ]]; then
    to_install="$to_install cython"
fi

if [[ $COVERAGE == "true" && $SKLEARN_TESTS != "true" ]]; then
    to_install="$to_install coverage pytest-cov"
fi

conda create -n testenv --yes -c conda-forge $to_install

conda activate testenv

# When using python-freethreading, lz4 should be installed with pip
if [[ $NO_LZ4 != "true" && $FREE_THREADED == "true" ]]; then
    to_install="$to_install lz4"
fi

if [[ "$NO_LZMA" == "true" ]]; then
    # Delete the LZMA module from the standard lib to make sure joblib has no
    # hard dependency on it:
    LZMA_PATH=`python -c "import lzma; print(lzma.__file__)"`
    echo "Deleting $LZMA_PATH..."
    rm $LZMA_PATH
fi

if [[ "$CYTHON" == "true" ]]; then
    cd joblib/test/_openmp_test_helper
    python setup.py build_ext -i
    cd ../../..
fi

pip install -v .
