#!/bin/sh

set -e

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [ -z "$NUMPY_VERSION" ]; then
        # We want to disable doctests because they need numpy to
        # run. I could not find a way to override the
        # --doctest-modules in setup.cfg so we remove the
        # doctest-related lines in setup.cfg instead
        sed -i '/--doctest/d' setup.cfg
    fi

    if [ "$COVERAGE" == "true" ]; then
        # Add coverage option to setup.cfg file if current test run
        # has to generate report for codecov ...
        export PYTEST_ADDOPTS="--cov=joblib"
    fi
    make
fi
