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
    # Install scikit-learn from conda, patch it to use this version of joblib
    # and run the scikit-learn tests with pytest.
    conda install --yes scikit-learn nose
    export SKLEARN=`python -c "import sklearn; print(sklearn.__path__[0])"`
    cp $TRAVIS_BUILD_DIR/continuous_integration/travis/copy_joblib.sh $SKLEARN/externals
    (cd $SKLEARN/externals && bash copy_joblib.sh $TRAVIS_BUILD_DIR)
    cp $TRAVIS_BUILD_DIR/continuous_integration/travis/conftest.py $SKLEARN/..
    pytest -vl --ignore $SKLEARN/externals --pyargs sklearn
fi
