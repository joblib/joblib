#!/bin/bash

# Change the list according to your local conda/virtualenv env.
CONDA_ENVS="py27-np16 py27-np18 py34-np110 py35-np112 py36-np114"
COMPRESS_METHODS="zlib gzip bz2 xz lzma lz4"

for i in $CONDA_ENVS
do
    . activate $i
    # Generate non compressed pickles.
    python create_numpy_pickle.py

    # Generate compressed pickles for each compression methods supported
    for method in $COMPRESS_METHODS
    do
        python create_numpy_pickle.py --compress --method $method
    done
done
. deactivate
