#!/bin/sh
# Do a local install of dill.

rm -rf dill
pip install --target=. dill
rm -rf *.egg-info *.dist-info

# Note: BSD sed -i needs an argument unders OSX
# so first renaming to .bak and then deleting backup files
find dill -name "*.py" | xargs sed -i.bak -r 's/from dill\.?/from ./;/^\s*(>>>|\.\.\.).+$/d'
find dill -name "*.bak" | xargs rm
