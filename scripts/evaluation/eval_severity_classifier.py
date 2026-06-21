"""Evaluate a lesion-crop severity classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.severity.dataset import DEFAULT_SEVERITY_MODEL_NAME, SeverityCropDataset
from src.severity.model import DEFAULT_XRV_WEIGHTS, build_severity_model


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def evaluate(
    checkpoint_path: Path,
    test_csv: Path,
    out_json: Path,
    predictions_csv: Path | None,
    batch_size: int,
    device: str,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = list(checkpoint.get("class_names", []))
    if not class_names:
        raise ValueError(f"{checkpoint_path} is missing class_names metadata.")

    img_size = int(checkpoint.get("img_size", 224))
    model_name = checkpoint.get("model_name", DEFAULT_SEVERITY_MODEL_NAME)
    xrv_weights = checkpoint.get("xrv_weights", DEFAULT_XRV_WEIGHTS)

    eval_device = _resolve_device(device)
    model = build_severity_model(
        model_name=model_name,
        num_classes=len(class_names),
        pretrained=False,
        xrv_weights=xrv_weights,
    ).to(eval_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = SeverityCropDataset(
        test_csv,
        train=False,
        img_size=img_size,
        model_name=model_name,
        class_names=class_names,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss(reduction="none")

    losses: list[float] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels, sample_weights in loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)
            sample_weights = sample_weights.to(eval_device)

            logits = model(images)
            per_sample_loss = criterion(logits, labels)
            loss = (per_sample_loss * sample_weights).mean()
            losses.append(float(loss.item()))

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels.cpu().numpy())

    probs_np = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, len(class_names)))
    labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.empty((0,), dtype=np.int64)
    preds_np = probs_np.argmax(axis=1) if len(probs_np) else np.empty((0,), dtype=np.int64)
    conf_np = probs_np.max(axis=1) if len(probs_np) else np.empty((0,), dtype=np.float32)

    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            labels_np,
            preds_np,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
        if len(labels_np)
        else (
            np.zeros(len(class_names)),
            np.zeros(len(class_names)),
            np.zeros(len(class_names)),
            np.zeros(len(class_names)),
        )
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "test_csv": str(test_csv),
        "samples": int(len(dataset)),
        "class_names": class_names,
        "img_size": img_size,
        "model_name": model_name,
        "device": eval_device,
        "test_loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(labels_np, preds_np)) if len(labels_np) else 0.0,
        "f1_macro": float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
        if len(labels_np)
        else 0.0,
        "f1_weighted": float(f1_score(labels_np, preds_np, average="weighted", zero_division=0))
        if len(labels_np)
        else 0.0,
        "confusion_matrix": confusion_matrix(
            labels_np,
            preds_np,
            labels=list(range(len(class_names))),
        ).tolist()
        if len(labels_np)
        else [[0 for _ in class_names] for _ in class_names],
        "per_class": {
            class_names[idx]: {
                "precision": float(per_class_precision[idx]),
                "recall": float(per_class_recall[idx]),
                "f1": float(per_class_f1[idx]),
                "support": int(per_class_support[idx]),
            }
            for idx in range(len(class_names))
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    predictions_path = predictions_csv
    if predictions_path is None:
        predictions_path = out_json.with_name(f"{out_json.stem}_predictions.csv")

    prediction_rows = dataset.df.copy()
    predicted_labels = [class_names[idx] for idx in preds_np.tolist()] if len(preds_np) else []
    prediction_rows["predicted_label"] = predicted_labels
    prediction_rows["predicted_confidence"] = conf_np.tolist()
    for idx, class_name in enumerate(class_names):
        prediction_rows[f"prob_{class_name}"] = probs_np[:, idx].tolist() if len(probs_np) else []
    prediction_rows.to_csv(predictions_path, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Metrics written to {out_json}")
    print(f"Predictions written to {predictions_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a multiclass severity classifier on a labeled crop CSV.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        out_json=args.out_json,
        predictions_csv=args.predictions_csv,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
