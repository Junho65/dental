"""Prepare lesion crops for supervised severity classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split


SEVERITY_CLASS_NAMES = ["caries", "deep_caries"]


def bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def expand_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(width, int(round(x2 + margin_x)))
    bottom = min(height, int(round(y2 + margin_y)))
    return left, top, right, bottom


def crop_and_save(image_path: Path, bbox_xyxy: tuple[float, float, float, float], out_path: Path, margin_ratio: float) -> None:
    image = Image.open(image_path).convert("RGB")
    left, top, right, bottom = expand_bbox(*bbox_xyxy, *image.size, margin_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).save(out_path)


def map_from_validation_category(name: str) -> str:
    key = name.lower()
    if "deep" in key:
        return "deep_caries"
    if "caries" in key:
        return "caries"
    return ""


def build_labeled_entries(raw_root: Path) -> list[dict]:
    val_json = raw_root / "DENTEX" / "validation_triple.json"
    val_img_dir = (
        raw_root
        / "DENTEX"
        / "validation_data"
        / "validation_data"
        / "quadrant_enumeration_disease"
        / "xrays"
    )
    payload = json.loads(val_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in payload["images"]}
    disease_map = {c["id"]: c["name"] for c in payload["categories_3"]}

    entries: list[dict] = []
    for ann in payload["annotations"]:
        label = map_from_validation_category(disease_map.get(ann.get("category_id_3", -1), ""))
        if label not in SEVERITY_CLASS_NAMES:
            continue
        image_info = images[ann["image_id"]]
        image_path = val_img_dir / image_info["file_name"]
        if not image_path.exists():
            continue
        entries.append(
            {
                "image_path": image_path,
                "label": label,
                "bbox_xyxy": bbox_xywh_to_xyxy([float(v) for v in ann["bbox"]]),
                "stem": f"{image_path.stem}_ann{ann['id']}",
                "source": "dentex_validation_triple",
            }
        )
    return entries


def _split_entries(entries: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [entry["label"] for entry in entries]
    train_entries, temp_entries = train_test_split(
        entries,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [entry["label"] for entry in temp_entries]
    val_entries, test_entries = train_test_split(
        temp_entries,
        test_size=2 / 3,
        random_state=seed,
        stratify=temp_labels,
    )
    return train_entries, val_entries, test_entries


def write_labeled_split(entries: list[dict], split: str, out_root: Path, margin_ratio: float) -> pd.DataFrame:
    rows = []
    for entry in entries:
        out_path = out_root / "images" / split / entry["label"] / f"{entry['stem']}.png"
        crop_and_save(entry["image_path"], entry["bbox_xyxy"], out_path, margin_ratio)
        rows.append(
            {
                "image_path": str(out_path.resolve()),
                "label": entry["label"],
                "source": entry["source"],
                "weight": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"{split}.csv", index=False)
    return df


def resolve_yolo_root(data_path: Path) -> tuple[Path, dict]:
    config = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root, config


def yolo_bbox_to_xyxy(parts: list[str], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    cls_id, cx, cy, bw, bh = [float(value) for value in parts]
    box_w = bw * width
    box_h = bh * height
    center_x = cx * width
    center_y = cy * height
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return x1, y1, x2, y2


def build_unlabeled_caries_entries(caries_yolo_yaml: Path, margin_ratio: float, out_root: Path) -> None:
    dataset_root, config = resolve_yolo_root(caries_yolo_yaml)
    for split in ("train", "val", "test"):
        split_rel = Path(config[split])
        image_dir = dataset_root / split_rel
        label_dir = dataset_root / "labels" / split_rel.name
        rows = []
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            for idx, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = raw_line.split()
                if len(parts) != 5:
                    continue
                bbox_xyxy = yolo_bbox_to_xyxy(parts, image.size)
                out_path = out_root / "images" / split / f"{image_path.stem}_box{idx}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                left, top, right, bottom = expand_bbox(*bbox_xyxy, *image.size, margin_ratio)
                image.crop((left, top, right, bottom)).save(out_path)
                rows.append(
                    {
                        "image_path": str(out_path.resolve()),
                        "source": "cariesxrays_decay",
                    }
                )
        pd.DataFrame(rows).to_csv(out_root / f"{split}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build labeled severity crops from DENTEX and optional unlabeled crops from CariesXrays."
    )
    parser.add_argument("--dentex-raw", type=Path, default=Path("data/raw/dentex"))
    parser.add_argument("--out", type=Path, default=Path("data/severity"))
    parser.add_argument(
        "--caries-yolo",
        type=Path,
        default=Path("data/detection_cariesxrays/cariesxrays_yolo.yaml"),
        help="Optional CariesXrays YOLO YAML for unlabeled crop export.",
    )
    parser.add_argument(
        "--unlabeled-out",
        type=Path,
        default=Path("data/severity_unlabeled"),
        help="Output directory for unlabeled lesion crops used by pseudo-labeling.",
    )
    parser.add_argument("--margin-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dentex_raw = args.dentex_raw.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    entries = build_labeled_entries(dentex_raw)
    if not entries:
        raise RuntimeError("No labeled caries/deep_caries crops found in validation_triple.json.")

    train_entries, val_entries, test_entries = _split_entries(entries, seed=args.seed)
    train_df = write_labeled_split(train_entries, "train", out_root, args.margin_ratio)
    val_df = write_labeled_split(val_entries, "val", out_root, args.margin_ratio)
    test_df = write_labeled_split(test_entries, "test", out_root, args.margin_ratio)

    stats = {
        "train": train_df["label"].value_counts().to_dict(),
        "val": val_df["label"].value_counts().to_dict(),
        "test": test_df["label"].value_counts().to_dict(),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Severity dataset prepared at {out_root}")
    print(json.dumps(stats, indent=2))

    caries_yolo_yaml = args.caries_yolo.resolve()
    if caries_yolo_yaml.exists():
        unlabeled_out = args.unlabeled_out.resolve()
        unlabeled_out.mkdir(parents=True, exist_ok=True)
        build_unlabeled_caries_entries(caries_yolo_yaml, args.margin_ratio, unlabeled_out)
        print(f"Unlabeled CariesXrays crops exported to {unlabeled_out}")
    else:
        print(f"Skipped unlabeled crop export because {caries_yolo_yaml} does not exist.")


if __name__ == "__main__":
    main()
