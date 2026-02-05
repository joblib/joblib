#!/bin/bash
# The behavior of the script is controlled by environment variabled defined
# in the .github/workflows/test.yml file defining the github action to run
# for the project.
#
# License: 3-clause BSD

set -xe

conda config --set solver libmamba

conda create -n py27-np16 --yes -c conda-forge python=2.7 pip lz4
conda activate py27-np16
pip install numpy==1.6 joblib==0.8.4
conda create -n py27-np18 --yes -c conda-forge python=2.7 pip lz4
conda activate py27-np18
pip install numpy==1.8 joblib==0.9.2
conda create -n py34-np110 --yes -c conda-forge python=3.4 pip lz4
conda activate py34-np110
pip install numpy==1.10 joblib==0.10.0
conda create -n py35-np112 --yes -c conda-forge python=3.5 pip lz4
conda activate py35-np112
pip install numpy==1.12 joblib==0.11.0
conda create -n py36-np114 --yes -c conda-forge python=3.6 pip lz4
conda activate py36-np114
pip install numpy==1.14 joblib==0.12.2
