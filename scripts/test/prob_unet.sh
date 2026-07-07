#!/bin/bash
# Re-evaluate Probabilistic U-Net checkpoints on all folds (or a subset), e.g.
# with a different number of samples.
# Uses the best checkpoint in logs/prob_unet/fold${N}; writes fold${N}/test.log.
#
# Usage: bash scripts/test/prob_unet.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#                                       [--beta 10.0] [--samples 16]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
BETA=10.0
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
echo "Test Prob U-Net | beta $BETA | samples $PU_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpt=$(best_ckpt "logs/prob_unet/fold${fold}")
    if [[ -z "$ckpt" ]]; then
        echo "[fold$fold] No checkpoint found, skipping."
        continue
    fi

    OUT="logs/prob_unet/fold${fold}"
    echo "=== prob_unet | fold $fold | $ckpt ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/prob_unet.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.beta=$BETA \
        --model.init_args.prob_unet_samples=$PU_SAMPLES \
        --do_train=False \
        --do_test=True \
        --ckpt_path="$ckpt" \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
