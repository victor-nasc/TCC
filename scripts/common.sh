# Shared helpers for scripts/train/* and scripts/test/*. Source, don't execute.
#
# Conventions for all scripts:
#   --folds "0 1 2"   folds to run (default: all 12). --folds 0 runs only fold 0.
#   --gpu <id>        GPU index (default 0)
#   --wandb 0|1       WandB logging (train scripts default 1, test scripts 0)
# DATA_DIR can be overridden via environment: DATA_DIR=/path bash scripts/...

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-/scratch/victornasc/WildfireSpreadTS/HDF5/}"
# Vegetation + Active Fire feature set (best in Chakravarty 2025 ablation):
# VIIRS M11, I2, I1, NDVI, EVI2, active fire time, binary fire mask
FEATURES="[0,1,2,3,4,38,39]"

GPU_ID=1
FOLDS="$(seq 0 11)"

wandb_mode() {
    [ "$WANDB" -eq 1 ] && echo "online" || echo "disabled"
}

# best_ckpt <dir> — epoch checkpoint with the highest step count under <dir>
best_ckpt() {
    find "$1" -name "epoch*.ckpt" 2>/dev/null |
        sed 's/.*step=\([0-9]*\)\.ckpt/\1 &/' |
        sort -rn |
        head -1 |
        cut -d' ' -f2-
}

unknown_arg() {
    echo "Unknown argument: $1" >&2
    exit 1
}
