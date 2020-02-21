#!/bin/bash

set -e

echo "Activating test environment:"
if [[ "$PYTHON_VERSION" == "pypy3" ]]; then
    source pypy3/bin/activate
else
    conda activate testenv
fi
which python
python -V
python -c "import multiprocessing as mp; print('multiprocessing.cpu_count():', mp.cpu_count())"
python -c "import joblib; print('joblib.cpu_count():', joblib.cpu_count())"

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

    # Move to a dedicated folder to avoid being polluted by joblib specific conftest.py
    # and disable the doctest plugin to avoid issues with doctests in scikit-learn
    # docstrings that require setting print_changed_only=True temporarily.
    cd "/tmp"
    pytest -vl --maxfail=5 -p no:doctest -k "not test_import_is_deprecated" --pyargs sklearn
fi
