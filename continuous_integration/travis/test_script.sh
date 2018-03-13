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
    export BUILD_ROOT="$HOME/build"
    conda install --yes numpy scipy cython
    git clone --depth=1 https://github.com/scikit-learn/scikit-learn.git \
        $BUILD_ROOT/scikit-learn/scikit-learn
    cd $BUILD_ROOT/scikit-learn/scikit-learn/sklearn/externals
    bash copy_joblib.sh $BUILD_ROOT/joblib/joblib
    cd $BUILD_ROOT/scikit-learn/scikit-learn
    make
fi
