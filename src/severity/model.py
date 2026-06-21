from __future__ import annotations

from pathlib import Path

import torch.nn as nn
import torchxrayvision as xrv

from .dataset import SEVERITY_CLASS_NAMES


DEFAULT_SEVERITY_MODEL_NAME = "xrv_densenet121"
DEFAULT_XRV_WEIGHTS = "densenet121-res224-all"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_XRV_CACHE_DIR = _PROJECT_ROOT / ".torchxrayvision"
_XRV_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _build_xrv_densenet121(
    num_classes: int,
    pretrained: bool,
    xrv_weights: str = DEFAULT_XRV_WEIGHTS,
):
    if pretrained:
        model = xrv.models.DenseNet(weights=xrv_weights, cache_dir=str(_XRV_CACHE_DIR))
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        model.op_threshs = None
        model.apply_sigmoid = False
    else:
        model = xrv.models.DenseNet(num_classes=num_classes)

    model.targets = list(SEVERITY_CLASS_NAMES)
    model.pathologies = list(SEVERITY_CLASS_NAMES)
    return model


def build_severity_model(
    model_name: str = DEFAULT_SEVERITY_MODEL_NAME,
    num_classes: int = len(SEVERITY_CLASS_NAMES),
    pretrained: bool = True,
    xrv_weights: str = DEFAULT_XRV_WEIGHTS,
):
    if model_name == "xrv_densenet121":
        return _build_xrv_densenet121(
            num_classes=num_classes,
            pretrained=pretrained,
            xrv_weights=xrv_weights,
        )
    raise ValueError(
        f"Unsupported severity model_name: {model_name}. "
        f"Supported: {DEFAULT_SEVERITY_MODEL_NAME}"
    )


def get_severity_classifier(model_name: str, model: nn.Module) -> nn.Module:
    if model_name == "xrv_densenet121":
        return model.classifier
    raise ValueError(
        f"Unsupported severity model_name: {model_name}. "
        f"Supported: {DEFAULT_SEVERITY_MODEL_NAME}"
    )


def configure_head_only_finetuning(model_name: str, model: nn.Module) -> nn.Module:
    for param in model.parameters():
        param.requires_grad = False

    classifier = get_severity_classifier(model_name, model)
    for param in classifier.parameters():
        param.requires_grad = True
    return classifier
