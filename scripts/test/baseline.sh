#!/bin/bash
# Evaluate baseline UTAE checkpoints on all folds (or a subset).
# Uses the best checkpoint in logs/seed${SEED}/fold${N}; writes fold${N}/test.log.
#
# Usage: bash scripts/test/baseline.sh [--seed 42] [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#        bash scripts/test/baseline.sh --fold 0 --ckpt path/to/model.ckpt   # single explicit ckpt

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
SEED=42
CKPT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --seed)  SEED=$2;   shift 2 ;;
        --ckpt)  CKPT=$2;   shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

if [[ -n "$CKPT" && $(echo $FOLDS | wc -w) -gt 1 ]]; then
    echo "Error: --ckpt requires a single fold (--fold N)." >&2
    exit 1
fi

cd "$ROOT"
echo "Test baseline | seed $SEED | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpt="$CKPT"
    if [[ -z "$ckpt" ]]; then
        ckpt=$(best_ckpt "logs/seed${SEED}/fold${fold}")
    fi
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "[fold$fold] No checkpoint found, skipping."
        continue
    fi

    OUT="logs/seed${SEED}/fold${fold}"
    mkdir -p "$OUT"
    echo "=== baseline | fold $fold | $ckpt ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/all_features.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --do_train=False \
        --do_validate=True \
        --do_test=True \
        --ckpt_path="$ckpt" \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
