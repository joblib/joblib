#!/bin/bash

set -e

echo "Activating test environment:"
if [[ "$PYTHON_VERSION" == pypy3* ]]; then
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

    pytest joblib -vl --timeout=120 --junitxml="${JUNITXML}"
    make test-doc
fi

if [[ "$SKLEARN_TESTS" == "true" ]]; then
    # Install the nightly build of scikit-learn and test against the installed
    # development version of joblib.
    # TODO: unpin pip once either https://github.com/pypa/pip/issues/10825
    # accepts invalid HTML or Anaconda is fixed.
    conda install -y -c conda-forge cython pillow numpy scipy "pip<22"
    pip install --pre --extra-index https://pypi.anaconda.org/scipy-wheels-nightly/simple scikit-learn
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"

    # Move to a dedicated folder to avoid being polluted by joblib specific conftest.py
    # and disable the doctest plugin to avoid issues with doctests in scikit-learn
    # docstrings that require setting print_changed_only=True temporarily.
    NEW_TEST_DIR=$(mktemp -d)
    cd $NEW_TEST_DIR

    pytest -vl --maxfail=5 -p no:doctest \
        -k "not test_import_is_deprecated" \
        -k "not test_check_memory" \
        --pyargs sklearn

    # Justification for skipping some tests:
    #
    # test_import_is_deprecated: Don't worry about deprecated imports: this is
    # tested for real in upstream scikit-learn and this is not joblib's
    # responsibility. Let's skip this test to avoid false positives in joblib's
    # CI.
    #
    # test_check_memory: scikit-learn test need to be updated to avoid using
    # cachedir: https://github.com/scikit-learn/scikit-learn/pull/22365
fi
