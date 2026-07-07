#!/bin/bash
# Re-evaluate MC Dropout checkpoints (trained WITH dropout via scripts/train/mcdropout.sh)
# on all folds (or a subset), e.g. with a different number of MC samples.
# Uses the best checkpoint in logs/mcdropout/fold${N}; writes fold${N}/test.log.
#
# --rate must match the training dropout rate: --ckpt_path only restores weights,
# not the hyperparameters the model was constructed with.
#
# Usage: bash scripts/test/mcdropout.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#                                       [--samples 20] [--rate 0.2]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
MC_SAMPLES=20
DROPOUT_RATE=0.2

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;       shift 2 ;;
        --wandb)   WANDB=$2;        shift 2 ;;
        --fold|--folds) FOLDS=$2;   shift 2 ;;
        --samples) MC_SAMPLES=$2;   shift 2 ;;
        --rate)    DROPOUT_RATE=$2; shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Test MC Dropout | rate $DROPOUT_RATE | samples $MC_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpt=$(best_ckpt "logs/mcdropout/fold${fold}")
    if [[ -z "$ckpt" ]]; then
        echo "[fold$fold] No checkpoint found, skipping."
        continue
    fi

    OUT="logs/mcdropout/fold${fold}"
    echo "=== mcdropout | fold $fold | $ckpt ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/mcdropout.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.mc_dropout_rate=$DROPOUT_RATE \
        --model.init_args.mc_dropout_samples=$MC_SAMPLES \
        --do_train=False \
        --do_test=True \
        --ckpt_path="$ckpt" \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
