#!/bin/bash
# Train baseline UTAE (one seed) on all folds (or a subset).
# Checkpoints/logs: logs/seed${SEED}/fold${N}/
# Folds that already have an epoch checkpoint are skipped.
#
# Usage: bash scripts/train/baseline.sh [--seed 42] [--folds "0 1 2"] [--gpu 0] [--wandb 1]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --seed)  SEED=$2;   shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Baseline UTAE | seed $SEED | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/seed${SEED}/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== seed $SEED | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== seed $SEED | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/all_features.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="seed${SEED}_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --do_train=True \
        --do_test=True \
        --seed_everything=$SEED \
        > "$OUT/train.log" 2>&1
done
