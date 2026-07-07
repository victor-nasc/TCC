#!/bin/bash
# Train Bayes by Backpropagation UTAE on all folds (or a subset).
# Checkpoints/logs: logs/${NAME}/fold${N}/  (default NAME=bbb_klfix)
# Folds that already have an epoch checkpoint are skipped.
#
# Default NAME is bbb_klfix because cfgs/UTAE/bbb.yaml now uses the ELBO-correct
# KL scaling (kl_weight=1.0 with kl_scale_by_batches=true). The old logs/bbb runs
# were trained with kl_weight=1e-6 (posterior collapsed to deterministic) and are
# kept as an ablation.
#
# Usage: bash scripts/train/bbb.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1] [--name bbb_klfix]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
NAME="bbb_klfix"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --name)  NAME=$2;   shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "BBB | name $NAME | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/${NAME}/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== $NAME | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== $NAME | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/bbb.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="${NAME}_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --do_train=True \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/train.log" 2>&1
done
