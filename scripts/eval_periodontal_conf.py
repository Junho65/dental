"""Evaluate a periodontal detector: standard val metrics, F1-optimal confidence, and
false-positive behavior on true-negative (background) images.

For each model we report:
  * overall + per-class precision/recall/mAP from model.val()
  * F1-optimal confidence threshold (overall and per class), extracted from the
    F1-confidence curve — this tells us whether our operating point sits near 0.1 or 0.4
  * false-positive box counts on the background (healthy) val images at several
    candidate confidence thresholds — this directly quantifies over-detection on normals

Usage:
  python scripts/eval_periodontal_conf.py --weights <path> --data <yaml> --split val
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
CANDIDATE_CONFS = [0.05, 0.1, 0.25, 0.4]


def _normalize_names(raw):
    if isinstance(raw, dict):
        return [raw[i] for i in sorted(raw)]
    return list(raw)


def _extract_f1_optimal(metrics, class_names):
    """Return {'overall': conf, '<class>': conf, ...} at the F1 peak of each curve."""
    conf_axis = np.linspace(0, 1, 1000)
    f1_curve = None

    # Preferred: curves_results carries labeled curves across recent ultralytics versions.
    curves = getattr(metrics, "curves_results", None)
    if curves:
        for entry in curves:
            # entry = [x_array, y_array(shape nc x N or N), xlabel, ylabel]
            if len(entry) >= 4 and str(entry[3]).lower().startswith("f1"):
                conf_axis = np.asarray(entry[0]).reshape(-1)
                f1_curve = np.asarray(entry[1])
                break

    if f1_curve is None:
        f1_curve = np.asarray(getattr(metrics.box, "f1_curve", []))

    if f1_curve.size == 0:
        return {"overall": None, "per_class": {}}

    if f1_curve.ndim == 1:
        f1_curve = f1_curve[None, :]
    # Align axis length if needed.
    n = f1_curve.shape[1]
    if conf_axis.shape[0] != n:
        conf_axis = np.linspace(0, 1, n)

    overall = float(conf_axis[int(f1_curve.mean(axis=0).argmax())])
    per_class = {}
    for i, name in enumerate(class_names):
        if i < f1_curve.shape[0]:
            per_class[name] = {
                "f1_optimal_conf": float(conf_axis[int(f1_curve[i].argmax())]),
                "f1_at_optimal": float(f1_curve[i].max()),
            }
    return {"overall": overall, "per_class": per_class}


def _background_images(data_yaml: Path, split: str):
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    bg = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        # Background = empty or missing label file (no positive boxes).
        if not lbl.exists() or lbl.stat().st_size == 0 or not lbl.read_text().strip():
            bg.append(img)
    return bg


def _fp_on_backgrounds(model, bg_images, imgsz, device):
    """For each candidate conf, count FP boxes (any detection on a true-negative image)."""
    out = {}
    for conf in CANDIDATE_CONFS:
        total_boxes = 0
        images_with_fp = 0
        for img in bg_images:
            res = model.predict(str(img), conf=conf, imgsz=imgsz, verbose=False, device=device)
            nb = 0 if (not res or res[0].boxes is None) else len(res[0].boxes)
            total_boxes += nb
            if nb > 0:
                images_with_fp += 1
        out[str(conf)] = {
            "fp_boxes_total": int(total_boxes),
            "images_with_fp": int(images_with_fp),
            "fp_boxes_per_image": round(total_boxes / max(len(bg_images), 1), 3),
        }
    return out


def evaluate(weights: str, data: str, split: str, imgsz: int, device: str, workers: int):
    model = YOLO(weights)
    class_names = _normalize_names(getattr(model, "names", []))
    metrics = model.val(data=data, split=split, imgsz=imgsz, workers=workers, device=device, verbose=False)

    per_class = []
    for row in metrics.summary(normalize=True, decimals=6):
        per_class.append(
            {
                "class_name": row["Class"],
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "f1": float(row["Box-F1"]),
                "map50": float(row["mAP50"]),
                "map50_95": float(row["mAP50-95"]),
            }
        )

    bg_images = _background_images(Path(data), split)
    fp = _fp_on_backgrounds(model, bg_images, imgsz, device)

    return {
        "weights": str(Path(weights).resolve()),
        "split": split,
        "imgsz": imgsz,
        "class_order": class_names,
        "overall": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
        "f1_optimal_conf": _extract_f1_optimal(metrics, class_names),
        "background_images": len(bg_images),
        "false_positives_on_backgrounds": fp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/detection_periodontal_pdcnn_2class_bg/periodontal_pdcnn_2class.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = evaluate(args.weights, args.data, args.split, args.imgsz, args.device, args.workers)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
