# Running UTAE Training and Evaluation

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

> `torch_scatter` may require a version-matched install. See [pytorch-scatter](https://github.com/rusty1s/pytorch-scatter) for instructions.

---

## Training

Runs 5-fold cross-validation by default. Results are saved under `--res_dir` (default: `./results`).

```bash
python train.py \
  --dataset_folder /path/to/PASTIS \
  --res_dir ./results \
  --model utae \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.001 \
  --num_workers 8 \
  --device cuda
```

### Train a single fold

```bash
python train.py \
  --dataset_folder /path/to/PASTIS \
  --res_dir ./results \
  --fold 1   # 1–5
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_folder` | *(required)* | Path to PASTIS dataset |
| `--res_dir` | `./results` | Output directory for weights and logs |
| `--model` | `utae` | Architecture: `utae`, `unet3d`, `fpn`, `convlstm`, `convgru`, `uconvlstm`, `buconvlstm` |
| `--epochs` | `100` | Epochs per fold |
| `--batch_size` | `4` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--fold` | `None` | Run a single fold (1–5); omit for all 5 folds |
| `--rdm_seed` | `1` | Random seed |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--cache` | `False` | Load full dataset into RAM |

### Output structure

```
results/
  conf.json          # Training config
  overall.json       # Aggregated metrics (all folds)
  Fold_1/
    model.pth.tar    # Best checkpoint (by val mIoU)
    trainlog.json    # Per-epoch train/val metrics
    test_metrics.json
    conf_mat.pkl
  Fold_2/ ...
```

---

## Evaluation (pre-trained weights)

Runs inference using saved checkpoints. Reads `conf.json` from `--weight_folder` to reconstruct model config automatically.

```bash
python test.py \
  --weight_folder ./results \
  --dataset_folder /path/to/PASTIS \
  --res_dir ./inference_utae \
  --device cuda
```

### Single fold

```bash
python test.py \
  --weight_folder ./results \
  --dataset_folder /path/to/PASTIS \
  --fold 1
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--weight_folder` | *(required)* | Path to training output dir (must contain `conf.json` and `Fold_*/model.pth.tar`) |
| `--dataset_folder` | *(required)* | Path to PASTIS dataset |
| `--res_dir` | `./inference_utae` | Output directory for inference results |
| `--fold` | `None` | Evaluate a single fold; omit for all 5 |
| `--device` | `cuda` | `cuda` or `cpu` |

---

## UTAE Architecture Defaults

| Parameter | Default |
|---|---|
| `encoder_widths` | `[64, 64, 64, 128]` |
| `decoder_widths` | `[32, 32, 64, 128]` |
| `out_conv` | `[32, 20]` |
| `n_head` | `16` |
| `d_model` | `256` |
| `d_k` | `4` |
| `agg_mode` | `att_group` |
| `encoder_norm` | `group` |
