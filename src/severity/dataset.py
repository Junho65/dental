from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SEVERITY_CLASS_NAMES = ["caries", "deep_caries"]
SEVERITY_CLASS_TO_ID = {name: idx for idx, name in enumerate(SEVERITY_CLASS_NAMES)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = True) -> transforms.Compose:
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
    def __init__(self, csv_path: str | Path, train: bool = True, img_size: int = 224):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        if "label" not in self.df.columns:
            raise ValueError(f"{self.csv_path} must contain a 'label' column.")
        if "weight" not in self.df.columns:
            self.df["weight"] = 1.0
        self.transforms = build_transforms(img_size=img_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])
        label_name = str(row["label"])
        if label_name not in SEVERITY_CLASS_TO_ID:
            raise KeyError(f"Unknown severity label '{label_name}' in {self.csv_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transforms(image)
        label = SEVERITY_CLASS_TO_ID[label_name]
        weight = float(row.get("weight", 1.0))
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)
