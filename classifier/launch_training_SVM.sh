#!/usr/bin/env bash

# Define the parameter lists
FOLDINDEXES=(0 1 2 3 4)
DATASETS=("mnist" "cifar10" "fashion_mnist" "imagenette_128")
FACTORS=(1 2 3 4)
COMPRESSION_LEVELS=(0)
FACTORING=("multicopy")

# Run in parallel
parallel -j 5 python utils/svm_training.py \
    --timestamp $(date +%Y%m%d_%H%M%S) \
    --foldindex {1} \
    --dataset {2} \
    --factors {3} \
    --compression_level {4} \
    --factoring {5} \
    ::: "${FOLDINDEXES[@]}" \
    ::: "${DATASETS[@]}" \
    ::: "${FACTORS[@]}" \
    ::: "${COMPRESSION_LEVELS[@]}" \
    ::: "${FACTORING[@]}"
