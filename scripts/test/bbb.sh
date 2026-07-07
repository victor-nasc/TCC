#!/bin/bash
# Re-evaluate BBB checkpoints on all folds (or a subset), e.g. with a different
# number of posterior samples.
# Uses the best checkpoint in logs/${NAME}/fold${N}; writes fold${N}/test.log.
#
# Usage: bash scripts/test/bbb.sh [--folds "0 1 2"] [--gpu 0] [--wandb 0]
#                                 [--samples 10] [--name bbb_klfix]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=0
BBB_SAMPLES=10
NAME="bbb_klfix"

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;      shift 2 ;;
        --wandb)   WANDB=$2;       shift 2 ;;
        --fold|--folds) FOLDS=$2;  shift 2 ;;
        --samples) BBB_SAMPLES=$2; shift 2 ;;
        --name)    NAME=$2;        shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Test BBB | name $NAME | samples $BBB_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    ckpt=$(best_ckpt "logs/${NAME}/fold${fold}")
    if [[ -z "$ckpt" ]]; then
        echo "[fold$fold] No checkpoint found, skipping."
        continue
    fi

    OUT="logs/${NAME}/fold${fold}"
    echo "=== $NAME | fold $fold | $ckpt ==="
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/bbb.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.bbb_samples=$BBB_SAMPLES \
        --do_train=False \
        --do_test=True \
        --ckpt_path="$ckpt" \
        --seed_everything=42 \
        > "$OUT/test.log" 2>&1
done
