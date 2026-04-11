import timm
import torch.nn as nn


def build_model(num_classes: int = 4):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return model


class MultiLabelLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        return self.loss(logits, targets)
