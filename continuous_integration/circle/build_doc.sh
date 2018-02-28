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

conda install --yes --quiet pip numpy sphinx=1.6.3 matplotlib pillow
pip install sphinx-gallery

cd "$HOME/$CIRCLE_PROJECT_REPONAME"
ls -l
python setup.py develop

# The pipefail is requested to propagate exit code
set -o pipefail && make doc 2>&1 | tee ~/log.txt

cd -
set +o pipefail
