import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.dataset import CLASS_NAMES, DentalMultiLabelDataset
from src.models.model import MultiLabelLoss, build_model


@torch.no_grad()
def evaluate(checkpoint_path: str, test_csv: str, threshold: float = 0.5, img_size: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = DentalMultiLabelDataset(test_csv, train=False, img_size=img_size)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    criterion = MultiLabelLoss()

    losses = []
    probs_all, targets_all = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = y.cpu().numpy()
        losses.append(loss.item())
        probs_all.append(probs)
        targets_all.append(targets)

    probs_all = np.concatenate(probs_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)
    preds = (probs_all >= threshold).astype(np.int32)

    metrics = {
        "test_loss": float(np.mean(losses)),
        "f1_macro": float(f1_score(targets_all, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(targets_all, preds, average="micro", zero_division=0)),
    }
    try:
        metrics["auroc_macro"] = float(
            roc_auc_score(targets_all, probs_all, average="macro")
        )
        metrics["auroc_micro"] = float(
            roc_auc_score(targets_all, probs_all, average="micro")
        )
    except Exception:
        metrics["auroc_macro"] = None
        metrics["auroc_micro"] = None

    per_class_recall = {}
    for i, cls in enumerate(CLASS_NAMES):
        tp = ((preds[:, i] == 1) & (targets_all[:, i] == 1)).sum()
        fn = ((preds[:, i] == 0) & (targets_all[:, i] == 1)).sum()
        recall = float(tp / max(tp + fn, 1))
        per_class_recall[cls] = recall
    metrics["recall_per_class"] = per_class_recall

    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/eval_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=256)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_csv, args.threshold, args.img_size)
