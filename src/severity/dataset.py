from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchxrayvision as xrv
from torchvision import transforms


SEVERITY_CLASS_NAMES = ["caries", "deep_caries"]
SEVERITY_CLASS_TO_ID = {name: idx for idx, name in enumerate(SEVERITY_CLASS_NAMES)}
DEFAULT_SEVERITY_MODEL_NAME = "xrv_densenet121"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resample_bilinear():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR


def _flip_left_right():
    if hasattr(Image, "Transpose"):
        return Image.Transpose.FLIP_LEFT_RIGHT
    return Image.FLIP_LEFT_RIGHT


class _XRayVisionTransform:
    def __init__(self, img_size: int = 224, train: bool = True):
        self.img_size = img_size
        self.train = train
        self.resample = _resample_bilinear()
        self.flip_left_right = _flip_left_right()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L")
        if self.train and random.random() < 0.5:
            image = image.transpose(self.flip_left_right)
        if self.train:
            angle = random.uniform(-8.0, 8.0)
            image = image.rotate(angle, resample=self.resample, fillcolor=0)
        image = image.resize((self.img_size, self.img_size), resample=self.resample)
        image_np = np.asarray(image, dtype=np.float32)
        image_np = xrv.datasets.normalize(image_np, 255)
        image_np = np.expand_dims(image_np, axis=0)
        return torch.from_numpy(np.ascontiguousarray(image_np))


def build_transforms(
    img_size: int = 224,
    train: bool = True,
    model_name: str = DEFAULT_SEVERITY_MODEL_NAME,
):
    if model_name == "xrv_densenet121":
        return _XRayVisionTransform(img_size=img_size, train=train)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class SeverityCropDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        train: bool = True,
        img_size: int = 224,
        model_name: str = DEFAULT_SEVERITY_MODEL_NAME,
        class_names: list[str] | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if "label" not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain a 'label' column.")
        if "weight" not in self.df.columns:
            self.df["weight"] = 1.0
        self.transforms = build_transforms(img_size=img_size, train=train, model_name=model_name)
        _names = class_names if class_names is not None else SEVERITY_CLASS_NAMES
        self.class_to_id = {name: idx for idx, name in enumerate(_names)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label_name = str(row["label"])
        if label_name not in self.class_to_id:
            raise KeyError(f"Unknown severity label '{label_name}' in {self.csv_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        label = self.class_to_id[label_name]
        weight = float(row.get("weight", 1.0))
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)
