#!/bin/bash
# Train Probabilistic U-Net (UTAE backbone + Gaussian latent space, Kohl et al.
# 2018) on all folds (or a subset), then run its own test pass sampling
# prob_unet_samples latents from the prior (aleatoric uncertainty).
# Checkpoints/logs: logs/prob_unet/fold${N}/
# Folds that already have an epoch checkpoint are skipped.
#
# Usage: bash scripts/train/prob_unet.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1]
#                                        [--beta 10.0] [--samples 16]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
BETA=1.0
PU_SAMPLES=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;     shift 2 ;;
        --wandb)   WANDB=$2;      shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --beta)    BETA=$2;       shift 2 ;;
        --samples) PU_SAMPLES=$2; shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Prob U-Net | beta $BETA | samples $PU_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/prob_unet/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== prob_unet | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== prob_unet | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/prob_unet.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="prob_unet_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.beta=$BETA \
        --model.init_args.prob_unet_samples=$PU_SAMPLES \
        --do_train=True \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/train.log" 2>&1
done
