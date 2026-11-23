#!/bin/bash

set -xe

get_version() {
    python -c "from continuous_integration.versions import get_oldest_pypy_package_version; print(get_oldest_pypy_package_version('$1', '$2'))"
}

get_python_versions() {
    python -c "from continuous_integration.versions import get_adjacent_python_versions; print(*get_adjacent_python_versions('$1'))"
}

read -r PREV_OLDEST_PYTHON_VERSION NEXT_OLDEST_PYTHON_VERSION <<< $(get_python_versions $OLDEST_PYTHON_VERSION)

conda config --set solver libmamba

for PRE in "PREV_" "" "NEXT_"; do
    CONDA_ENV="${PRE}oldest"
    PYTHON_VERSION="${PRE}OLDEST_PYTHON_VERSION"
    NUMPY_VERSION=$(get_version numpy ${!PYTHON_VERSION})
    JOBLIB_VERSION=$(get_version joblib ${!PYTHON_VERSION})

    conda create -n $CONDA_ENV --yes -c conda-forge python=${!PYTHON_VERSION} pip
    conda activate $CONDA_ENV
    pip install numpy==$NUMPY_VERSION joblib==$JOBLIB_VERSION lz4
    conda deactivate
done
