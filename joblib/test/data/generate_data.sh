#!/bin/bash

# Change the list according to your local conda/virtualenv env.
CONDA_ENVS="py27-np19 py35-np111 py36-np113"
COMPRESS_METHODS="gzip bz2 xz lzma"
EXPECTED=0

for env in $CONDA_ENVS
do
    conda activate $env

    # Generate non compressed pickles.
    python create_numpy_pickle.py
    EXPECTED=$((EXPECTED+1))

    # Generate compressed pickle with zlib
    python create_numpy_pickle.py --compress --method zlib
    EXPECTED=$((EXPECTED+1))

    if [[ $env == "py27-np19" ]]; then continue; fi

    # Generate compressed pickles for each compression methods supported
    for method in $COMPRESS_METHODS
    do
        python create_numpy_pickle.py --compress --method $method
        EXPECTED=$((EXPECTED+1))
    done

    if [[ $env == "py35-np111" ]]; then continue; fi

    # Generate compressed pickle with lz4
    python create_numpy_pickle.py --compress --method lz4
    EXPECTED=$((EXPECTED+1))
done

ls

GENERATED=$(ls *.pkl* | wc -l)
if [[ $GENERATED != $EXPECTED ]]; then
    echo "Error: $GENERATED files generetad, while $EXPECTED expected files..."
    exit 1;
fi
