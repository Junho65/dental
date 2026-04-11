from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .dataset import SEVERITY_CLASS_NAMES, build_transforms
from .model import build_severity_model


class SeverityPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu"):
        ckpt_path = Path(checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        self.class_names = checkpoint.get("class_names", SEVERITY_CLASS_NAMES)
        self.img_size = int(checkpoint.get("img_size", 224))
        self.model_name = checkpoint.get("model_name", "efficientnet_b0")
        self.device = device
        self.transforms = build_transforms(img_size=self.img_size, train=False)

        self.model = build_severity_model(
            model_name=self.model_name,
            num_classes=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_pil(self, image: Image.Image) -> dict:
        tensor = self.transforms(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        class_id = int(torch.argmax(probs).item())
        confidence = float(probs[class_id].item())
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))
            },
        }
