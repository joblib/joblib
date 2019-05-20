#!/bin/sh

set -e

python -c "import multiprocessing as mp; print('multiprocessing.cpu_count():', mp.cpu_count())"
python -c "import joblib; print('joblib.cpu_count():', joblib.cpu_count())"

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/travis/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [ "$COVERAGE" == "true" ]; then
        # Enable coverage-related options. --cov-append is needed to combine
        # the test run and the test-doc run coverage.
        export PYTEST_ADDOPTS="--cov=joblib --cov-append"
    fi
    make
    make test-doc
fi

if [[ "$SKLEARN_TESTS" == "true" ]]; then
    # Install scikit-learn from conda and test against the installed
    # development version of joblib.
    conda install --yes cython pillow scikit-learn scipy==1.2.1
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"
    # Skip test_lars_cv_max_iter because of a warning that is (probably)
    # not related to joblib. To be confirmed once the following PR is
    # merged:
    # https://github.com/scikit-learn/scikit-learn/pull/12597
    pytest -vl -k "not test_lars_cv_max_iter" --pyargs sklearn
fi
