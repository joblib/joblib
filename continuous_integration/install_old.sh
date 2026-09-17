#!/bin/bash

set -xe

conda config --set solver libmamba

for PRE in "BEFORE_" "" "AFTER_"; do
    CONDA_ENV="${PRE}oldest"
    delcare -n PYTHON_VERSION="${PRE}OLDEST_PYTHON_VERSION"
    delcare -n NUMPY_VERSION="${PRE}OLDEST_NUMPY_VERSION"
    delcare -n JOBLIB_VERSION="${PRE}OLDEST_JOBLIB_VERSION"

    conda create -n $CONDA_ENV --yes -c conda-forge python=$PYTHON_VERSION pip
    conda activate $CONDA_ENV
    pip install numpy==$NUMPY_VERSION joblib==$JOBLIB_VERSION lz4
done
