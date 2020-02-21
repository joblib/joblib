#!/bin/bash

set -e

echo "Activating test environment:"
if [[ "$PYTHON_VERSION" == "pypy3" ]]; then
    source pypy3/bin/activate
else
    source activate testenv
fi
which python
python -V
python -c "import multiprocessing as mp; print('multiprocessing.cpu_count():', mp.cpu_count())"
python -c "import joblib; print('joblib.cpu_count():', joblib.cpu_count())"

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [ "$COVERAGE" == "true" ]; then
        # Enable coverage-related options. --cov-append is needed to combine
        # the test run and the test-doc run coverage.
        export PYTEST_ADDOPTS="--cov=joblib --cov-append"
    fi

    pytest joblib -vl --timeout=60 --junitxml="${JUNITXML}"
    if [[ "$PYTHON_VERSION" != "2.7" ]]; then
        make test-doc
    fi
fi

if [[ "$SKLEARN_TESTS" == "true" ]]; then
    # Install scikit-learn from conda and test against the installed
    # development version of joblib.
    conda remove -y numpy
    conda install -y -c conda-forge cython pillow scikit-learn
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"

    # Move to a dedicated folder to avoid being polluted by joblib specific conftest.py
    # and disable the doctest plugin to avoid issues with doctests in scikit-learn
    # docstrings that require setting print_changed_only=True temporarily.
    cd "/tmp"
    pytest -vl --maxfail=5 -p no:doctest -k "not test_import_is_deprecated" --pyargs sklearn
fi
