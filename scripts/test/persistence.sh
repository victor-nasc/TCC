#!/bin/bash
# Persistence baseline on all folds (or a subset): predicts the last observed
# binary active fire mask as the next-day fire map. No training involved.
# Results: logs/persistence/fold${N}/test.log
#
# Usage: bash scripts/test/persistence.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Persistence baseline | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/persistence/fold${fold}"
    mkdir -p "$OUT"
    echo "=== persistence | fold $fold ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/PersistenceModel/full_run.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.data_fold_id=$fold \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
