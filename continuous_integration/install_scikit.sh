set -xe

conda config --set solver libmamba
conda create -n testenv --yes -c conda-forge python=$LATEST_PYTHON_VERSION pip pytest scipy
conda activate testenv

pip install numpy cython # really necessary??
pip install -v .
pip install --pre --extra-index https://pypi.anaconda.org/scientific-python-nightly-wheels/simple scikit-learn
