from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# Match the current extension taxonomy used by the project so this dataset can
# be merged later without remapping class ids again.
CLASS_NAMES = [
    "caries_family",
    "periapical_lesion",
    "impacted_tooth",
    "bone_loss",
    "cyst",
    "retained_root",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Source labels are in Chinese. We only keep labels that are both clinically
# meaningful on panoramic X-rays and directly useful for the current detector.
SOURCE_TO_TARGET = {
    "龋病": "caries_family",
    "根尖周炎": "periapical_lesion",
}
SKIPPED_SOURCE_LABELS = {
    "深窝沟",  # anatomy / caries-risk finding, not a current detection target
    "牙髓炎",  # weak direct panoramic target for this detector
    "牙齿发育异常",  # too broad to map safely to impacted_tooth
    "其他",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _bbox_to_yolo(x_min: float, y_min: float, x_max: float, y_max: float, width: float, height: float) -> str:
    box_w = max(x_max - x_min, 0.0)
    box_h = max(y_max - y_min, 0.0)
    if box_w <= 0.0 or box_h <= 0.0:
        raise ValueError("Invalid bounding box with non-positive size.")
    cx = (x_min + box_w / 2.0) / width
    cy = (y_min + box_h / 2.0) / height
    nw = box_w / width
    nh = box_h / height
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        if src.resolve().drive.lower() == dst.resolve().drive.lower():
            os.link(src, dst)
            return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _collect_entries(raw_root: Path) -> tuple[list[dict[str, Any]], Counter]:
    stats: Counter = Counter()
    entries: list[dict[str, Any]] = []

    for source_split, source_name in (("Train", "train"), ("Test", "test")):
        image_dir = raw_root / source_split / "images"
        label_dir = raw_root / source_split / "label"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing expected split folders under {raw_root}: {source_split}")

        for label_path in sorted(label_dir.glob("*.json")):
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            image_path = _find_image(image_dir, label_path.stem)
            if image_path is None:
                stats["missing_images"] += 1
                continue

            width = max(float(payload.get("imageWidth", 0) or 0), 1.0)
            height = max(float(payload.get("imageHeight", 0) or 0), 1.0)
            label_lines: list[str] = []
            present_targets: set[str] = set()

            for shape in payload.get("shapes", []):
                source_label = str(shape.get("label", "")).strip()
                target_label = SOURCE_TO_TARGET.get(source_label)
                if target_label is None:
                    if source_label in SKIPPED_SOURCE_LABELS:
                        stats[f"skipped::{source_label}"] += 1
                    else:
                        stats[f"unknown::{source_label}"] += 1
                    continue

                points = shape.get("points", [])
                if len(points) < 2:
                    stats["invalid_shapes"] += 1
                    continue

                xs = [float(point[0]) for point in points]
                ys = [float(point[1]) for point in points]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                try:
                    yolo_box = _bbox_to_yolo(x_min, y_min, x_max, y_max, width, height)
                except ValueError:
                    stats["invalid_boxes"] += 1
                    continue

                class_id = CLASS_TO_ID[target_label]
                label_lines.append(f"{class_id} {yolo_box}")
                present_targets.add(target_label)
                stats[target_label] += 1

            if not label_lines:
                stats["images_without_selected_labels"] += 1
                continue

            combo = tuple(sorted(present_targets))
            stats[f"combo::{'+'.join(combo)}"] += 1
            entries.append(
                {
                    "source_split": source_name,
                    "image_path": image_path,
                    "label_lines": label_lines,
                    "combo_key": "|".join(combo),
                }
            )
            stats[f"images::{source_name}"] += 1

    return entries, stats


def _write_entry(entry: dict[str, Any], split: str, out_root: Path, stem_prefix: str) -> None:
    image_path: Path = entry["image_path"]
    out_stem = f"{stem_prefix}{entry['source_split']}_{image_path.stem}"
    out_image = out_root / "images" / split / f"{out_stem}{image_path.suffix.lower()}"
    out_label = out_root / "labels" / split / f"{out_stem}.txt"
    _link_or_copy(image_path, out_image)
    out_label.write_text("\n".join(entry["label_lines"]) + "\n", encoding="utf-8")


def _split_entries(entries: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_train = [entry for entry in entries if entry["source_split"] == "train"]
    raw_test = [entry for entry in entries if entry["source_split"] == "test"]
    if not raw_train or not raw_test:
        raise RuntimeError("Both raw Train and raw Test must contain usable entries.")

    stratify = [entry["combo_key"] for entry in raw_train]
    train_entries, val_entries = train_test_split(
        raw_train,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return train_entries, val_entries, raw_test


def _build_split_stats(entries: list[dict[str, Any]]) -> dict[str, int]:
    stats: Counter = Counter()
    stats["images"] = len(entries)
    for entry in entries:
        for raw_line in entry["label_lines"]:
            class_id = int(raw_line.split()[0])
            stats[CLASS_NAMES[class_id]] += 1
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert pediatric panoramic detection raw data into YOLO train/val/test for the project taxonomy."
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "kaggle" / "archive_4_bundle" / "Dental_dataset" / "Pediatric dental disease detection dataset",
        help="Raw pediatric detection dataset root containing Train/ and Test/ folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "detection_kaggle_pediatric_selected_6class",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of raw Train entries to reserve for val. Raw Test is kept as test.",
    )
    parser.add_argument(
        "--stem-prefix",
        default="kped_",
        help="Prefix used to avoid filename collisions after merging.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    entries, ingest_stats = _collect_entries(raw_root)
    train_entries, val_entries, test_entries = _split_entries(entries, args.val_fraction, args.seed)

    _ensure_dirs(out_root)
    for entry in train_entries:
        _write_entry(entry, "train", out_root, args.stem_prefix)
    for entry in val_entries:
        _write_entry(entry, "val", out_root, args.stem_prefix)
    for entry in test_entries:
        _write_entry(entry, "test", out_root, args.stem_prefix)

    split_stats = {
        "train": _build_split_stats(train_entries),
        "val": _build_split_stats(val_entries),
        "test": _build_split_stats(test_entries),
    }
    metadata = {
        "source_root": str(raw_root),
        "output_root": str(out_root),
        "class_names": CLASS_NAMES,
        "selected_source_to_target": SOURCE_TO_TARGET,
        "skipped_source_labels": sorted(SKIPPED_SOURCE_LABELS),
        "ingest_stats": dict(ingest_stats),
        "split_stats": split_stats,
        "notes": {
            "test_policy": "Raw Test kept as test; val sampled from raw Train only.",
            "label_policy": "Only labels directly useful for the current detector taxonomy were retained.",
        },
    }
    (out_root / "dataset_stats.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    yaml_path = out_root / "pediatric_selected_6class.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {CLASS_NAMES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared YOLO dataset at {out_root}")
    print(f"YAML: {yaml_path}")
    print(
        "Splits: "
        f"train={len(train_entries)} "
        f"val={len(val_entries)} "
        f"test={len(test_entries)}"
    )
    for split_name in ("train", "val", "test"):
        stats = split_stats[split_name]
        print(
            f"{split_name}: images={stats.get('images', 0)} "
            + " ".join(f"{name}={stats.get(name, 0)}" for name in CLASS_NAMES)
        )


if __name__ == "__main__":
    main()
