#!/bin/bash
# Evaluate Deep Ensembles on all folds (or a subset): for each fold, gather the
# best checkpoint from each independently trained seed run (logs/seed*/foldN)
# and test DeepEnsembleLightning with averaged predictions.
# Results: logs/deep_ensemble/fold${N}/test.log
#
# Usage: bash scripts/test/deep_ensemble.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#                                           [--seeds "42 123 456 789 1337"]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
SEEDS="42 123 456 789 1337"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)   GPU_ID=$2; shift 2 ;;
        --wandb) WANDB=$2;  shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --seeds) SEEDS=$2;  shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Test Deep Ensemble | seeds: $SEEDS | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpts=()
    for seed in $SEEDS; do
        ckpt=$(best_ckpt "logs/seed${seed}/fold${fold}")
        if [[ -z "$ckpt" ]]; then
            echo "[fold$fold] Missing checkpoint for seed$seed, skipping fold."
            continue 2
        fi
        ckpts+=("\"$ckpt\"")
    done

    ckpt_list=$(IFS=,; echo "[${ckpts[*]}]")
    OUT="logs/deep_ensemble/fold${fold}"
    mkdir -p "$OUT"

    echo "=== deep_ensemble | fold $fold | ${#ckpts[@]} members ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/deep_ensemble.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="deep_ensemble_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.ensemble_ckpt_paths="$ckpt_list" \
        --do_train=False \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
