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
