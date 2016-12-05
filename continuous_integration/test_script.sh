#!/bin/sh

set -e

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [ -z "$NUMPY_VERSION" ]; then
        # We want to disable doctests because they need numpy to run. I
        # could not find a way to override the with-doctest value in
        # setup.cfg so doing it the hacky way ...
        cat setup.cfg | grep -v '    --doctest' > setup.cfg.new
        mv setup.cfg{.new,}
    fi

    if [ "$COVERAGE" == "true" ]; then
        # Add coverage option to setup.cfg file if current test run
        # has to generate report for coveralls ...
        export PYTEST_ADDOPTS="--cov=joblib"
    fi
    make
fi
