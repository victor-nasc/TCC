#!/bin/bash
# Run first half of the training experiments on GPU 0, fold0 only.
# Models run strictly one at a time (each must finish before the next starts).
# Usage: bash scripts/train/run_gpu0.sh

set -e
DIR="$(dirname "${BASH_SOURCE[0]}")"

MODELS=(baseline bbb ssn)

for model in "${MODELS[@]}"; do
    echo "=== [GPU 0] starting $model ==="
    bash "$DIR/${model}.sh" --folds "0" --gpu 0
    echo "=== [GPU 0] finished $model ==="
done
