# Experiment scripts

All scripts run **all 12 folds by default**; restrict with `--fold 0` or `--folds "0 3 7"`.
Common flags: `--gpu <id>` (default 0), `--wandb 0|1` (train default 1, test default 0).
Override the data location with the `DATA_DIR` environment variable.

Train scripts skip folds that already have a checkpoint, so they can be re-run to resume.

## Train (`scripts/train/`)

| Script | Model | Output |
|---|---|---|
| `baseline.sh` | UTAE baseline (`--seed 42`) | `logs/seed${SEED}/fold${N}/` |
| `ensemble_members.sh` | 5 seeds × folds for Deep Ensembles | `logs/seed*/fold${N}/` |
| `mcdropout.sh` | UTAE + decoder dropout (`--rate 0.2 --samples 20`) | `logs/mcdropout/fold${N}/` |
| `bbb.sh` | Bayes by Backprop (`--name bbb_klfix`) | `logs/bbb_klfix/fold${N}/` |

## Test (`scripts/test/`)

| Script | Evaluates | Output |
|---|---|---|
| `baseline.sh` | best ckpt per fold of one seed (`--seed 42`, or `--fold N --ckpt PATH`) | `test.log` next to ckpts |
| `mcdropout.sh` | MC-dropout ckpts with `--samples`/`--rate` | `logs/mcdropout/fold${N}/test.log` |
| `bbb.sh` | BBB ckpts with `--samples` (`--name bbb_klfix`) | `logs/${NAME}/fold${N}/test.log` |
| `deep_ensemble.sh` | 5-member ensemble from `logs/seed*/fold${N}` | `logs/deep_ensemble/fold${N}/test.log` |
| `persistence.sh` | persistence baseline (no ckpt) | `logs/persistence/fold${N}/test.log` |

## Notes

- MC Dropout requires dropout **during training**; never evaluate MC sampling on
  baseline (`mc_dropout_rate=0.0`) checkpoints.
- `logs/bbb` holds the old collapsed-posterior BBB runs (`kl_weight=1e-6`); new
  runs with ELBO-scaled KL go to `logs/bbb_klfix`.
- `scripts/legacy/` holds the superseded single-purpose root scripts.
- Run scripts with the `uq` conda env active (they call `python3`).
