"""Evaluate an Ultralytics detector checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def _normalize_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [raw_names[idx] for idx in sorted(raw_names)]
    return list(raw_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/detection/yolov8n_dentex/weights/best.pt")
    parser.add_argument("--data", default="data/detection/dentex_detection.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/detection_metrics.json"))
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, split=args.split, workers=args.workers)
    class_names = _normalize_names(getattr(model, "names", getattr(metrics, "names", [])))

    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    per_class: list[dict] = []
    for row in metrics.summary(normalize=True, decimals=6):
        per_class.append(
            {
                "class_name": row["Class"],
                "images": int(row["Images"]),
                "instances": int(row["Instances"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "f1": float(row["Box-F1"]),
                "map50": float(row["mAP50"]),
                "map50_95": float(row["mAP50-95"]),
            }
        )

    out = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "split": args.split,
        "workers": args.workers,
        "class_order": class_names,
        "overall": overall,
        "per_class": per_class,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
