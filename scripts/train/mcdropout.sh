#!/bin/bash
# Train MC Dropout UTAE (decoder Dropout2d active during training) on all folds
# (or a subset), then run its own test pass with stochastic MC sampling.
# Checkpoints/logs: logs/mcdropout/fold${N}/
# Folds that already have an epoch checkpoint are skipped.
#
# NOTE: MC Dropout requires dropout during TRAINING (Gal & Ghahramani 2016).
# Evaluating with test-time dropout on a baseline checkpoint trained with
# mc_dropout_rate=0.0 is invalid.
#
# Usage: bash scripts/train/mcdropout.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1]
#                                        [--rate 0.2] [--samples 20]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
DROPOUT_RATE=0.2
MC_SAMPLES=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;       shift 2 ;;
        --wandb)   WANDB=$2;        shift 2 ;;
        --fold|--folds) FOLDS=$2;   shift 2 ;;
        --rate)    DROPOUT_RATE=$2; shift 2 ;;
        --samples) MC_SAMPLES=$2;   shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "MC Dropout | rate $DROPOUT_RATE | samples $MC_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/mcdropout/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== mcdropout | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== mcdropout | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/mcdropout.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="mcdropout_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.mc_dropout_rate=$DROPOUT_RATE \
        --model.init_args.mc_dropout_samples=$MC_SAMPLES \
        --do_train=True \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/train.log" 2>&1
done
