#!/bin/bash

set -xe

echo "Activating test environment:"
conda activate testenv

if [[ $PYTHON_VERSION == free-threaded* ]]; then
    # This is needed because for now some C extensions have not declared their
    # thread-safety with free-threaded Python, for example numpy and coverage.tracer
    export PYTHON_GIL=0
    # For free-threaded Python, run parallel tests to validate thread-safety (at
    # least somewhat):
    NUM_CORES=$(python -c "import joblib; print(joblib.cpu_count())")
    PARALLEL_PYTEST_ARGS="--parallel-threads $NUM_CORES --iterations 1"
else
    PARALLEL_PYTEST_ARGS=""
fi

if [[ "$ONE_CPU" == "1" ]]; then
    # Note that ONE_CPU should only be set on Linux:
    PYTEST_PREFIX="taskset -c 0"
else
    PYTEST_PREFIX=""
fi

which python
# Show python version and build information (e.g. free-threaded or not)
python -VV
python -c "import multiprocessing as mp; print('multiprocessing.cpu_count():', mp.cpu_count())"
python -c "import joblib; print('joblib.cpu_count():', joblib.cpu_count())"

$PYTEST_PREFIX pytest joblib -vl --timeout=120 --cov=joblib --cov-report xml $PARALLEL_PYTEST_ARGS

# doctests are not compatile with default_backend=threading
if [[ $JOBLIB_TESTS_DEFAULT_PARALLEL_BACKEND != "threading" ]]; then
    make test-doc
fi
