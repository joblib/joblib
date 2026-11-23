#!/bin/bash

COMPRESS_METHODS="zlib gzip bz2 xz lzma lz4"
EXPECTED=0

for PRE in "PREV_" "" "NEXT_"; do
    env="${PRE}oldest"
    conda activate $env

    # Generate non compressed pickles.
    python create_numpy_pickle.py
    EXPECTED=$((EXPECTED+1))

    # Generate compressed pickles for each compression methods supported
    for method in $COMPRESS_METHODS; do
        python create_numpy_pickle.py --compress --method $method
        EXPECTED=$((EXPECTED+1))
    done
done

echo "======= GENERATED FILES ======="
ls *.pkl*
echo "==============================="

GENERATED=$(ls *.pkl* | wc -l)
if [[ $GENERATED != $EXPECTED ]]; then
    echo "Error: $GENERATED files generetad, while $EXPECTED expected files..."
    exit 1;
fi
