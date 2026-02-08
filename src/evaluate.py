from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .checkpointing import load_checkpoint
from .data_utils import load_class_names, load_config, load_split_csv
from .dataset import MIMICCXRJPGDataset
from .model import create_swin_base
from .plotting import plot_per_class_auc

def _strip_prefix_if_present(state_dict: dict, prefix: str = "module.") -> dict:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(isinstance(k, str) and k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _safe_load_state_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """Load state dict with a couple of common fallbacks."""
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except Exception:
        pass
    # try stripping 'module.' prefix
    model.load_state_dict(_strip_prefix_if_present(state_dict), strict=True)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on MIMIC-CXR-JPG test split.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config used for training.")
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path. If not set, uses checkpoints/<exp_name>/best.pth if exists, else last.pth.",
    )
    return p.parse_args()


def choose_checkpoint(exp_name: str, checkpoint_dir: Path, checkpoint_arg: Optional[str]) -> Path:
    if checkpoint_arg is not None:
        p = Path(checkpoint_arg)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {p}")
        return p

    best = checkpoint_dir / exp_name / "best.pth"
    last = checkpoint_dir / exp_name / "last.pth"
    if best.exists():
        return best
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir/exp_name} (expected best.pth or last.pth).")


@torch.no_grad()
def forward_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_scores = []

    use_amp = mixed_precision and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else None

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if autocast_ctx is not None:
            with autocast_ctx():
                logits = model(images)
        else:
            logits = model(images)

        probs = torch.sigmoid(logits)

        all_targets.append(targets.detach().cpu())
        all_scores.append(probs.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_score = torch.cat(all_scores).numpy()
    return y_true, y_score


def plot_roc_all_classes(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: List[str],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    for i, name in enumerate(class_names):
        yt = y_true[:, i]
        ys = y_score[:, i]
        if np.unique(yt).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        try:
            auc = roc_auc_score(yt, ys)
        except ValueError:
            auc = float("nan")
        label = f"{name} (AUC={auc:.3f})" if not np.isnan(auc) else f"{name} (AUC=nan)"
        plt.plot(fpr, tpr, label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.legend(fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    exp_name = cfg.get("experiment_name", "swin_base_mimic")
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device: {device}")

    mimic_root = Path(data_cfg["mimic_root"])
    test_csv = data_cfg.get("test_csv", None)
    if test_csv is None:
        raise ValueError("Config is missing data.test_csv")
    class_names = load_class_names(data_cfg["class_names_json"])
    image_size = int(data_cfg.get("image_size", 512))
    num_workers = int(data_cfg.get("num_workers", 8))

    batch_size = int(train_cfg.get("batch_size", 32))
    mixed_precision = bool(train_cfg.get("mixed_precision", True))

    checkpoint_root = Path(train_cfg.get("checkpoint_dir", "./checkpoints"))
    plots_root = Path(train_cfg.get("plots_dir", "./plots"))

    ckpt_path = choose_checkpoint(exp_name, checkpoint_root, args.checkpoint)
    print(f"[eval] Using checkpoint: {ckpt_path}")

    test_df = load_split_csv(test_csv, class_names)
    test_dataset = MIMICCXRJPGDataset(
        df=test_df,
        mimic_root=mimic_root,
        class_names=class_names,
        image_size=image_size,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Build model
    model = create_swin_base(num_classes=len(class_names), pretrained=False, img_size=image_size)
    model = model.to(device)

    ckpt = load_checkpoint(ckpt_path, device=device)

    # Prefer EMA weights if present
    ema_state = ckpt.get("ema_state", None)
    if ema_state:
        _safe_load_state_dict(model, ema_state)
        print("[eval] Loaded EMA weights from checkpoint.")
    else:
        _safe_load_state_dict(model, ckpt["model_state"])
        print("[eval] Loaded model_state weights from checkpoint (no EMA in checkpoint).")

    y_true, y_score = forward_all(model, test_loader, device=device, mixed_precision=mixed_precision)

    per_class_auc: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        try:
            per_class_auc[name] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        except ValueError:
            per_class_auc[name] = float("nan")

    mean_auc = float(np.nanmean(list(per_class_auc.values())))
    print(f"[eval] Test mean AUC (nanmean over classes) = {mean_auc:.6f}")

    # Write per-class AUC txt
    out_dir = checkpoint_root / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / "test_per_class_auc.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"mean_auc={mean_auc:.6f}\n")
        for k, v in per_class_auc.items():
            f.write(f"{k}\t{v:.6f}\n")
    print(f"[eval] Wrote per-class AUC -> {txt_path}")

    # Plots
    plot_dir = plots_root / exp_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_class_plot = plot_dir / "test_per_class_auc.png"
    plot_per_class_auc(per_class_auc, per_class_plot, title=f"{exp_name} - Test Per-class AUC")

    roc_plot = plot_dir / "test_roc_all_classes.png"
    plot_roc_all_classes(y_true, y_score, class_names, roc_plot, title=f"{exp_name} - Test ROC (all classes)")

    print(f"[eval] Saved plots -> {plot_dir}")


if __name__ == "__main__":
    main()
