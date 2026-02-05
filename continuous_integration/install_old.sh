#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# License: 3-clause BSD

set -xe

conda config --set solver libmamba

conda create -n py27-np19 --yes -c conda-forge python=2.7 pip
conda activate py27-np19
pip install numpy==1.9 joblib==0.8.4 lz4
conda create -n py27-np110 --yes -c conda-forge python=2.7 pip
conda activate py27-np110
pip install numpy==1.10 joblib==0.9.2 lz4
conda create -n py35-np111 --yes -c conda-forge python=3.5 pip
conda activate py35-np111
pip install numpy==1.11 joblib==0.10.0 lz4
conda create -n py35-np112 --yes -c conda-forge python=3.5 pip
conda activate py35-np112
pip install numpy==1.12 joblib==0.11.0 lz4
conda create -n py36-np113 --yes -c conda-forge python=3.6 pip
conda activate py36-np113
pip install numpy==1.13 joblib==0.12.2 lz4
