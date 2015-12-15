#!/bin/sh
# Do a local install of cloudpickle.

pip install --target=. cloudpickle
rm -rf *.dist-info
find cloudpickle -name "*.py" | xargs sed -i.bak "s/from cloudpickle\(\.\)\{0,1\}/from ./"
find cloudpickle -name "*.bak" | xargs rm
