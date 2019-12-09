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
    conda remove -y numpy
    conda install -y -c conda-forge cython pillow scikit-learn
    python -c "import sklearn; print('Testing scikit-learn', sklearn.__version__)"

    # Hack to workaround shadowing of public function by compat modules:
    # https://github.com/scikit-learn/scikit-learn/issues/15842
    SKLEARN_ROOT=`python -c "import sklearn; print(sklearn.__path__[0])"`
    rm -rf "$SKLEARN_ROOT/decomposition/dict_learning.py"
    rm -rf "$SKLEARN_ROOT/inspection/partial_dependence.py"
    pytest -vl --maxfail=5 -k "not test_import_is_deprecated" --pyargs sklearn
fi
