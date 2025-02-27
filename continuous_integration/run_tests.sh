#!/bin/bash

set -xe

echo "Activating test environment:"
if [[ "$PYTHON_VERSION" == pypy3* ]]; then
    source pypy3/bin/activate
elif [[ "$PYTHON_VERSION" == free-threaded* ]]; then
    source testenv/bin/activate
    # This is needed because for now some C extensions have not declared their
    # thread-safety with free-threaded Python, for example numpy and coverage.tracer
    export PYTHON_GIL=0
else
    conda activate testenv
fi
which python
python -V
python -c "import multiprocessing as mp; print('multiprocessing.cpu_count():', mp.cpu_count())"
python -c "import joblib; print('joblib.cpu_count():', joblib.cpu_count())"

if [[ "$SKLEARN_TESTS" != "true" ]]; then
    pytest joblib -vl --timeout=120 --cov=joblib --cov-report xml

    # doctests are not compatile with default_backend=threading
    if [[ "$JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND" != "threading" ]]; then
        make test-doc
    fi
else
    # Install the nightly build of scikit-learn and test against the installed
    # development version of joblib.
    conda install -y -c conda-forge cython numpy scipy
    pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"

    # Move to a dedicated folder to avoid being polluted by joblib specific conftest.py
    # and disable the doctest plugin to avoid issues with doctests in scikit-learn
    # docstrings that require setting print_changed_only=True temporarily.
    NEW_TEST_DIR=$(mktemp -d)
    cd $NEW_TEST_DIR

    pytest -vl --maxfail=5 -p no:doctest \
        --pyargs sklearn
fi
