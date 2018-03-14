#!/bin/sh

set -e

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/travis/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [ "$COVERAGE" == "true" ]; then
        # Add coverage option to setup.cfg file if current test run
        # has to generate report for codecov ...
        export PYTEST_ADDOPTS="--cov=joblib"
    fi
    make
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
