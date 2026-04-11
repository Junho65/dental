from __future__ import annotations

import timm

from .dataset import SEVERITY_CLASS_NAMES


def build_severity_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = len(SEVERITY_CLASS_NAMES),
    pretrained: bool = True,
):
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
