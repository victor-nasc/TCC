#!/bin/bash
# Re-evaluate Stochastic Segmentation Network checkpoints on all folds (or a
# subset), e.g. with a different number of samples.
# Uses the best checkpoint in logs/ssn/fold${N}; writes fold${N}/test.log.
#
# Usage: bash scripts/test/ssn.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#                                 [--rank 10] [--samples 16]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
RANK=10
SSN_SAMPLES=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;      shift 2 ;;
        --wandb)   WANDB=$2;       shift 2 ;;
        --fold|--folds) FOLDS=$2;  shift 2 ;;
        --rank)    RANK=$2;        shift 2 ;;
        --samples) SSN_SAMPLES=$2; shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Test SSN | rank $RANK | samples $SSN_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpt=$(best_ckpt "logs/ssn/fold${fold}")
    if [[ -z "$ckpt" ]]; then
        echo "[fold$fold] No checkpoint found, skipping."
        continue
    fi

    OUT="logs/ssn/fold${fold}"
    echo "=== ssn | fold $fold | $ckpt ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/ssn.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.ssn_rank=$RANK \
        --model.init_args.ssn_samples=$SSN_SAMPLES \
        --do_train=False \
        --do_test=True \
        --ckpt_path="$ckpt" \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
