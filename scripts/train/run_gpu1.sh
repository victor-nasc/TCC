#!/bin/bash
# Run second half of the training experiments on GPU 1, fold0 only.
# Models run strictly one at a time (each must finish before the next starts).
# Usage: bash scripts/train/run_gpu1.sh

set -e
DIR="$(dirname "${BASH_SOURCE[0]}")"

MODELS=(mcdropout prob_unet flow_matching)

for model in "${MODELS[@]}"; do
    echo "=== [GPU 1] starting $model ==="
    bash "$DIR/${model}.sh" --folds "0" --gpu 1
    echo "=== [GPU 1] finished $model ==="
done
