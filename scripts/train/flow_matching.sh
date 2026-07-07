#!/bin/bash
# Train conditional Flow Matching (UTAE-conditioned velocity field over next-day
# fire masks, Lipman et al. 2023) on all folds (or a subset), then run its own
# test pass integrating fm_samples ODE trajectories (aleatoric uncertainty).
# Checkpoints/logs: logs/flow_matching/fold${N}/
# Folds that already have an epoch checkpoint are skipped.
#
# Usage: bash scripts/train/flow_matching.sh [--folds "0 1 2"] [--gpu 0] [--wandb 1]
#                                            [--steps 20] [--samples 16]

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
WANDB=1
ODE_STEPS=20
FM_SAMPLES=16

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)     GPU_ID=$2;     shift 2 ;;
        --wandb)   WANDB=$2;      shift 2 ;;
        --fold|--folds) FOLDS=$2; shift 2 ;;
        --steps)   ODE_STEPS=$2;  shift 2 ;;
        --samples) FM_SAMPLES=$2; shift 2 ;;
        *) unknown_arg "$1" ;;
    esac
done

cd "$ROOT"
echo "Flow Matching | steps $ODE_STEPS | samples $FM_SAMPLES | folds: $FOLDS | GPU $GPU_ID | wandb $(wandb_mode)"

for fold in $FOLDS; do
    OUT="logs/flow_matching/fold${fold}"
    if [[ -n "$(best_ckpt "$OUT")" ]]; then
        echo "=== flow_matching | fold $fold — checkpoint exists, skipping ==="
        continue
    fi
    echo "=== flow_matching | fold $fold ==="
    mkdir -p "$OUT"
    WANDB_MODE=$(wandb_mode) python3 src/train.py \
        -c cfgs/UTAE/flow_matching.yaml \
        --trainer cfgs/trainer_single_gpu.yaml \
        --trainer.devices="[$GPU_ID]" \
        --trainer.default_root_dir="$OUT" \
        --trainer.logger.init_args.name="flow_matching_fold${fold}" \
        --data cfgs/data_monotemporal_full_features.yaml \
        --data.data_dir "$DATA_DIR" \
        --data.return_doy=True \
        --data.features_to_keep="$FEATURES" \
        --data.data_fold_id=$fold \
        --model.init_args.fm_ode_steps=$ODE_STEPS \
        --model.init_args.fm_samples=$FM_SAMPLES \
        --do_train=True \
        --do_test=True \
        --seed_everything=42 \
        > "$OUT/train.log" 2>&1
done
