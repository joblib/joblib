#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is adapted from a similar script from the scikit-learn repository.
#
# License: 3-clause BSD

set -e

create_new_venv() {
    # At the time of writing numpy 1.9.1 is included in the travis
    # virtualenv but we want to make sure that joblib has only a soft
    # dependence on numpy. For this we create a new virtualenv from
    # scratch
    deactivate
    virtualenv --system-site-packages testvenv
    source testvenv/bin/activate
    pip install dill
    pip install nose
}

if [[ "$WITHOUT_NUMPY" == "true" ]]; then
    create_new_venv
    # We want to disable doctests because they need numpy to run. I
    # could not find a way to override the with-doctest value in
    # setup.cfg so doing it the hacky way ...
    cat setup.cfg | grep -v 'with-doctest=' > setup.cfg.new
    mv setup.cfg{.new,}
fi

python setup.py install
