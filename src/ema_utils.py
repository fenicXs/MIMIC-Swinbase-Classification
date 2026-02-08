from __future__ import annotations

from typing import Optional
import torch
from torch import nn
from timm.utils import ModelEmaV2


class EMAWrapper:
    """Simple wrapper around timm's ModelEmaV2."""

    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[str] = None):
        self.ema = ModelEmaV2(model, decay=decay, device=device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.ema.update(model)
