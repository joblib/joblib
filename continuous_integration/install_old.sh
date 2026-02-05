#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# License: 3-clause BSD

set -xe

conda config --set solver libmamba

conda create -n py27-np16 --yes -c conda-forge python=2.7 numpy=1.6 lz4 joblib=0.8.4
conda create -n py27-np18 --yes -c conda-forge python=2.7 numpy=1.8 lz4 joblib=0.9.2
conda create -n py34-np110 --yes -c conda-forge python=3.4 numpy=1.10 lz4 joblib=0.10.0
conda create -n py35-np112 --yes -c conda-forge python=3.5 numpy=1.12 lz4 joblib=0.11.0
conda create -n py36-np114 --yes -c conda-forge python=3.6 numpy=1.14 lz4 joblib=0.12.2
