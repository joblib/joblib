#!/bin/sh

set -e

if [[ -n "$FLAKE8_VERSION" ]]; then
    source continuous_integration/flake8_diff.sh
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    PYTEST_VERSION=$(python -c "import pytest; print(pytest.__version__)")

    if [ ${PYTEST_VERSION:0:1} == "3" ]; then
        # pytest >= 3.0.0 Produces a very long error log with pytest
        # warnings for deprecated yield based tests. This is a hacky way
        # to produce cleaner CI logs by disabling warnings.
        # I am not adding this to setup.cfg directly as it is not available
        # for pytest < 2.8.0
        sed -i '/addopts/a\    --disable-pytest-warnings' setup.cfg
    fi

    if [ -z "$NUMPY_VERSION" ]; then
        # We want to disable doctests because they need numpy to run. I
        # could not find a way to override the with-doctest value in
        # setup.cfg so doing it the hacky way ...
        cat setup.cfg | grep -v '    --doctest' > setup.cfg.new
        mv setup.cfg{.new,}
    fi
    make
fi
