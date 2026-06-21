"""Prepare periodontal YOLO datasets from PDCNN annotations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


TARGET_NAMES = ["bone_loss", "furcation_involvement"]
TASKS = {
    "BL": {
        "json_name": "via_export_coco_BL.json",
        "target_name": "bone_loss",
        "positive_categories": {"mild", "medium", "severe"},
    },
    "FI": {
        "json_name": "via_export_coco_FI.json",
        "target_name": "furcation_involvement",
        "positive_categories": {"mild", "severe"},
    },
}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_task_annotations(
    raw_root: Path,
) -> tuple[dict[str, dict], dict[str, list[dict]], dict[str, dict], dict[str, dict]]:
    images_by_name: dict[str, dict] = {}
    annotations_by_image: dict[str, list[dict]] = defaultdict(list)
    task_summaries: dict[str, dict] = {}
    # Every image that appears in either COCO file, regardless of whether it carries a
    # positive (mild/medium/severe) box. Used to recover true-negative background images.
    all_images_by_name: dict[str, dict] = {}

    for task_name, task_config in TASKS.items():
        coco_path = raw_root / task_config["json_name"]
        data = json.loads(coco_path.read_text(encoding="utf-8"))
        categories = {category["id"]: category["name"] for category in data["categories"]}
        images_by_id = {image["id"]: image for image in data["images"]}
        for image in data["images"]:
            all_images_by_name.setdefault(image["file_name"], image)
        positive_categories = task_config["positive_categories"]
        target_name = task_config["target_name"]
        target_id = TARGET_NAMES.index(target_name)

        counts: Counter = Counter()
        positive_images: set[str] = set()
        skipped_no_category = 0
        for annotation in data["annotations"]:
            category_id = annotation.get("category_id")
            if category_id is None:
                skipped_no_category += 1
                continue
            original_category = categories[category_id]
            if original_category not in positive_categories:
                continue
            image = images_by_id[annotation["image_id"]]
            file_name = image["file_name"]
            images_by_name[file_name] = image
            positive_images.add(file_name)
            annotations_by_image[file_name].append(
                {
                    "task": task_name,
                    "target_id": target_id,
                    "target_name": target_name,
                    "original_category": original_category,
                    "bbox": annotation["bbox"],
                }
            )
            counts[original_category] += 1
            counts["boxes"] += 1

        task_summaries[task_name] = {
            "json": str(coco_path),
            "target_name": target_name,
            "positive_categories": sorted(positive_categories),
            "source_images": len(data["images"]),
            "source_annotations": len(data["annotations"]),
            "positive_images": len(positive_images),
            "skipped_annotations_without_category": skipped_no_category,
            "positive_box_counts": dict(counts),
        }

    return images_by_name, annotations_by_image, task_summaries, all_images_by_name


def _clip_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x_min, y_min, box_width, box_height = bbox
    x_max = min(width, x_min + box_width)
    y_max = min(height, y_min + box_height)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 1 or box_height <= 1:
        return None
    return x_min, y_min, box_width, box_height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert PDCNN bone-loss and furcation-involvement COCO annotations to 2-class YOLO."
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/pdcnn_periodontitis_bone_loss"))
    parser.add_argument("--out", type=Path, default=Path("data/detection_periodontal_pdcnn_2class_bg"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-background",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include true-negative images (no bone_loss/furcation box) as empty-label "
        "background images to reduce false positives. Use --no-include-background for the "
        "legacy positives-only dataset.",
    )
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    image_root = raw_root / "Images"
    images_by_name, annotations_by_image, task_summaries, all_images_by_name = _load_task_annotations(raw_root)
    positive_file_names = sorted(annotations_by_image)
    random.Random(args.seed).shuffle(positive_file_names)

    train_end = round(len(positive_file_names) * 0.8)
    val_end = round(len(positive_file_names) * 0.9)
    splits = {
        "train": positive_file_names[:train_end],
        "val": positive_file_names[train_end:val_end],
        "test": positive_file_names[val_end:],
    }

    # True-negative background images: present in either COCO file but with no positive box.
    # Added with empty label files so YOLO treats them as backgrounds, which teaches the
    # detector that healthy mouths/teeth should not be flagged and also lets val/test
    # measure false positives on normal cases.
    background_file_names: list[str] = []
    background_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    if args.include_background:
        background_file_names = sorted(set(all_images_by_name) - set(annotations_by_image))
        random.Random(args.seed + 1).shuffle(background_file_names)
        bg_train_end = round(len(background_file_names) * 0.8)
        bg_val_end = round(len(background_file_names) * 0.9)
        background_splits = {
            "train": background_file_names[:bg_train_end],
            "val": background_file_names[bg_train_end:bg_val_end],
            "test": background_file_names[bg_val_end:],
        }

    split_counts: dict[str, dict[str, int]] = {}
    manifest_rows: list[dict[str, str | int | float]] = []
    missing_images: list[str] = []

    for split, file_names in splits.items():
        counts: Counter = Counter()
        for file_name in file_names:
            image = images_by_name[file_name]
            src_image = image_root / file_name
            if not src_image.exists():
                missing_images.append(str(src_image))
                counts["missing_images"] += 1
                continue

            width = int(image["width"])
            height = int(image["height"])
            stem = Path(file_name).stem
            dst_image = out_root / "images" / split / f"pdcnn_{stem}{src_image.suffix.lower()}"
            dst_label = out_root / "labels" / split / f"pdcnn_{stem}.txt"

            lines: list[str] = []
            image_manifest_rows: list[dict[str, str | int | float]] = []
            for annotation in annotations_by_image[file_name]:
                clipped = _clip_box(annotation["bbox"], width=width, height=height)
                if clipped is None:
                    counts["invalid_boxes_skipped"] += 1
                    continue
                x_min, y_min, box_width, box_height = clipped
                x_center = (x_min + box_width / 2) / width
                y_center = (y_min + box_height / 2) / height
                yolo_width = box_width / width
                yolo_height = box_height / height
                target_id = int(annotation["target_id"])
                lines.append(f"{target_id} {x_center:.6f} {y_center:.6f} {yolo_width:.6f} {yolo_height:.6f}")
                counts[TARGET_NAMES[target_id]] += 1
                counts[f"{annotation['target_name']}_{annotation['original_category']}"] += 1
                counts["boxes"] += 1
                image_manifest_rows.append(
                    {
                        "split": split,
                        "image_path": dst_image.relative_to(out_root).as_posix(),
                        "label_path": dst_label.relative_to(out_root).as_posix(),
                        "source_file_name": file_name,
                        "source_task": annotation["task"],
                        "detector_class_id": target_id,
                        "detector_class": annotation["target_name"],
                        "original_pdcnn_category": annotation["original_category"],
                        "x_center": round(x_center, 6),
                        "y_center": round(y_center, 6),
                        "width": round(yolo_width, 6),
                        "height": round(yolo_height, 6),
                    }
                )

            if not lines:
                counts["images_without_valid_boxes"] += 1
                continue

            _link_or_copy(src_image, dst_image)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            manifest_rows.extend(image_manifest_rows)
            counts["images"] += 1

        # Background (true-negative) images for this split: copy image + empty label file.
        for file_name in background_splits[split]:
            image = all_images_by_name[file_name]
            src_image = image_root / file_name
            if not src_image.exists():
                missing_images.append(str(src_image))
                counts["missing_background_images"] += 1
                continue
            stem = Path(file_name).stem
            dst_image = out_root / "images" / split / f"pdcnn_{stem}{src_image.suffix.lower()}"
            dst_label = out_root / "labels" / split / f"pdcnn_{stem}.txt"
            _link_or_copy(src_image, dst_image)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("", encoding="utf-8")
            counts["background_images"] += 1
            counts["images"] += 1

        split_counts[split] = dict(counts)

    yaml_path = out_root / "periodontal_pdcnn_2class.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {TARGET_NAMES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = out_root / "periodontal_bbox_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    else:
        manifest_path.write_text("", encoding="utf-8")

    summary = {
        "source": "PuckBlink/PDCNN perio-dataset",
        "raw_root": str(raw_root),
        "raw_images_zip": str(raw_root / "Images.zip"),
        "target_names": TARGET_NAMES,
        "severity_policy": "Detector classes collapse PDCNN severity labels; original categories are preserved in the manifest.",
        "include_background": args.include_background,
        "background_policy": "True-negative images (no positive box in BL or FI) added as empty-label background images.",
        "task_summaries": task_summaries,
        "positive_images_union": len(positive_file_names),
        "background_images_union": len(background_file_names),
        "background_split_sizes": {split: len(names) for split, names in background_splits.items()},
        "missing_positive_images": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "split_counts": split_counts,
        "yaml": str(yaml_path),
        "manifest": str(manifest_path),
    }
    (out_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
