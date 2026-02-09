# MIMIC-CXR PA Disease Classification (Swin-Base + EMA)

This repository provides a **reproducible** training pipeline to train **Swin-Base** on **MIMIC-CXR-JPG v2.0.0**
using **CheXpert 14 labels** and **EMA (Exponential Moving Average)**.

It is adapted from a Swin-Base + EMA ChestXray14 pipeline and updated for the **MIMIC folder structure**,
the official `mimic-cxr-2.0.0-split.csv` splits, and CheXpert labels.

## What this repo does (PA-only baseline)

- Uses **only PA views** (filtered using `mimic-cxr-2.0.0-metadata.csv` -> `ViewPosition == "PA"`).
- Uses **all 14 CheXpert labels** from `mimic-cxr-2.0.0-chexpert.csv`.
- Uses **U-Zeros** uncertainty policy: `-1 -> 0`.
- Keeps **all PA images**, even if multiple PA images exist for the same study (same study label repeated).

## Quick start (example on ASU SOL)

### 0) Install deps
```bash
cd MIMIC-Swinbase-Classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Prepare PA-only CSV splits (one-time)
This reads:
- `mimic-cxr-2.0.0-split.csv`
- `mimic-cxr-2.0.0-metadata.csv`
- `mimic-cxr-2.0.0-chexpert.csv`
- `IMAGE_FILENAMES`

...and writes:
- `data/processed_pa/train.csv`
- `data/processed_pa/validate.csv`
- `data/processed_pa/test.csv`
- `data/processed_pa/class_names.json`

Run:
```bash
python -m src.prepare_mimic   --mimic_root /scratch/pkrish52/MIMIC   --output_dir ./data/processed_pa   --views PA   --uncertainty_policy u_zeros
```

### 2) Train (with EMA, early stopping, auto-resume)
```bash
python -m src.train --config configs/train_mimic_pa_512_ema.yaml
```

#### Resume training (important for 16-hour SOL windows)
This repo supports **auto-resume**:
- If `checkpoints/<experiment_name>/last.pth` exists, training will automatically resume from it.

You can also explicitly resume:
```bash
python -m src.train --config configs/train_mimic_pa_512_ema.yaml --resume checkpoints/<exp_name>/last.pth
```

### 3) Evaluate on test split (and plot per-class AUC)
```bash
python -m src.evaluate --config configs/train_mimic_pa_512_ema.yaml
```

Outputs:
- `plots/<exp_name>/test_per_class_auc.png`
- `plots/<exp_name>/test_roc_all_classes.png`
- `checkpoints/<exp_name>/test_per_class_auc.txt`

## Outputs

During training:
- `checkpoints/<exp_name>/last.pth` (overwritten every epoch)
- `checkpoints/<exp_name>/best.pth` (updated when val mAUC improves)
- `checkpoints/<exp_name>/history.csv`
- `plots/<exp_name>/train_loss.png` (overwritten each epoch)
- `plots/<exp_name>/val_mean_auc.png` (overwritten each epoch)

## Config notes

Edit `configs/train_mimic_pa_512_ema.yaml` to change:
- batch size
- image size (e.g., 384 vs 512)
- early stopping patience
- optimizer LR / weight decay
- EMA decay
- warmup epochs

## Notes
This repo is scoped to **PA-only** training.

