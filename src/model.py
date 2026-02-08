from __future__ import annotations

from torch import nn
import timm


def create_swin_base(num_classes: int, pretrained: bool = True, img_size: int = 512) -> nn.Module:
    """Create a Swin-Base model (timm) and override img_size.

    Using timm model:
      swin_base_patch4_window12_384

    Swin uses relative position bias, so overriding `img_size` allows using 512x512,
    while still loading pretrained weights.
    """
    model = timm.create_model(
        "swin_base_patch4_window12_384",
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
    )
    return model
