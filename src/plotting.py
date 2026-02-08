from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_train_loss(history: Mapping[str, List[float]], out_path: str | Path, title: str = "Train Loss"):
    epochs = history.get("epoch", [])
    losses = history.get("train_loss", [])
    if not epochs or not losses:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, losses, marker="o", linestyle="-")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_val_mean_auc(history: Mapping[str, List[float]], out_path: str | Path, title: str = "Val Mean AUC"):
    epochs = history.get("epoch", [])
    aucs = history.get("val_mean_auc", [])
    if not epochs or not aucs:
        return

    # filter out nan for plotting
    x = []
    y = []
    for e, a in zip(epochs, aucs):
        if a is None:
            continue
        try:
            if np.isnan(a):
                continue
        except TypeError:
            pass
        x.append(e)
        y.append(a)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("mAUC")
    plt.plot(x, y, marker="s", linestyle="--")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_per_class_auc(
    per_class_auc: Mapping[str, float],
    out_path: str | Path,
    title: str = "Per-class AUC",
):
    names = list(per_class_auc.keys())
    values = [per_class_auc[k] for k in names]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.bar(range(len(names)), values)
    plt.gca().set_xticks(range(len(names)))
    plt.gca().set_xticklabels(names)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
