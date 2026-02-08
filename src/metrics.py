from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


@torch.no_grad()
def evaluate_auc(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    use_mixed_precision: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate mean ROC-AUC and per-class ROC-AUC."""
    model.eval()
    all_targets = []
    all_probs = []

    use_amp = bool(use_mixed_precision) and device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    with autocast_ctx():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.sigmoid(logits)

            all_targets.append(targets.detach().cpu())
            all_probs.append(probs.detach().cpu())

    y_true = torch.cat(all_targets).numpy()
    y_score = torch.cat(all_probs).numpy()

    per_class_aucs: Dict[str, float] = {}
    for i, name in enumerate(class_names):
        try:
            per_class_aucs[name] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        except ValueError:
            per_class_aucs[name] = float("nan")

    mean_auc = float(np.nanmean(list(per_class_aucs.values())))
    return mean_auc, per_class_aucs
