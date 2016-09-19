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
        cat setup.cfg | grep -v 'with-doctest=' > setup.cfg.new
        mv setup.cfg{.new,}
    fi
    make
fi
