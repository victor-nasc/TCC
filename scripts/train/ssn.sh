#!/bin/bash
# Train Stochastic Segmentation Network (UTAE backbone + low-rank Gaussian over
# logits, Monteiro et al. 2020) on all folds (or a subset), then run its own
# test pass sampling ssn_samples coherent logit maps (aleatoric uncertainty).
# Checkpoints/logs: logs/ssn/fold${N}/
# Folds that already have an epoch checkpoint are skipped.
#
# Usage: bash scripts/train/ssn.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1]
#                                  [--rank 10] [--samples 16]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
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
echo "SSN | rank $RANK | samples $SSN_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/ssn/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== ssn | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== ssn | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/ssn.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="ssn_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.ssn_rank=$RANK \
        --model.init_args.ssn_samples=$SSN_SAMPLES \
        --do_train=True \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/train.log" 2>&1
done
