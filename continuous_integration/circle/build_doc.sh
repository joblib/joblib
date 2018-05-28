#!/usr/bin/env bash
set -x
set -e

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
    deactivate
fi

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
     -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

conda create -n $CONDA_ENV_NAME --yes --quiet python=3
source activate $CONDA_ENV_NAME

conda install --yes --quiet pip numpy sphinx=1.6.3 matplotlib pillow dask distributed
pip install sphinx-gallery numpydoc

python setup.py develop

make doc 2>&1
