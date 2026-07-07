#!/bin/bash
# Train all Deep Ensemble members: 5 seeds x folds, via scripts/train/baseline.sh.
# Already-trained (seed, fold) pairs are skipped automatically.
#
# Usage: bash scripts/train/ensemble_members.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1] [--seeds "42 123"]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
SEEDS="123 456 789 1337"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --seeds) SEEDS=$2;  shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

for seed in $SEEDS; do
    bash "$ROOT/scripts/train/baseline.sh" \
        --seed "$seed" --folds "$FOLDS" --gpu "$GPU_ID" --wandb "$WANDB"
done
