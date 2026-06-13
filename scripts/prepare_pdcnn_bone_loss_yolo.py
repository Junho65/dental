from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDCNN bone-loss COCO annotations to YOLO.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/pdcnn_periodontitis_bone_loss"))
    parser.add_argument("--out", type=Path, default=Path("data/detection_pdcnn_bone_loss"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    image_root = raw_root / "Images"
    coco_path = raw_root / "via_export_coco_BL.json"
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category["name"] for category in data["categories"]}

    annotations: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        if categories.get(annotation["category_id"]) == "healthy":
            continue
        annotations[annotation["image_id"]].append(annotation)

    positive_ids = [image_id for image_id, items in annotations.items() if items]
    rng = random.Random(args.seed)
    rng.shuffle(positive_ids)

    train_end = round(len(positive_ids) * 0.8)
    val_end = round(len(positive_ids) * 0.9)
    splits = {
        "train": positive_ids[:train_end],
        "val": positive_ids[train_end:val_end],
        "test": positive_ids[val_end:],
    }

    split_counts: dict[str, dict[str, int]] = {}
    missing_images: list[str] = []
    for split, image_ids in splits.items():
        counts: Counter = Counter()
        for image_id in image_ids:
            image = images[image_id]
            src_image = image_root / image["file_name"]
            if not src_image.exists():
                missing_images.append(str(src_image))
                counts["missing_images"] += 1
                continue

            stem = Path(image["file_name"]).stem
            dst_image = out_root / "images" / split / f"pdcnn_{stem}{src_image.suffix.lower()}"
            dst_label = out_root / "labels" / split / f"pdcnn_{stem}.txt"

            width = image["width"]
            height = image["height"]
            lines: list[str] = []
            for annotation in annotations[image_id]:
                x_min, y_min, box_width, box_height = annotation["bbox"]
                x_max = min(width, x_min + box_width)
                y_max = min(height, y_min + box_height)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                box_width = x_max - x_min
                box_height = y_max - y_min
                if box_width <= 1 or box_height <= 1:
                    counts["invalid_boxes_skipped"] += 1
                    continue
                x_center = (x_min + box_width / 2) / width
                y_center = (y_min + box_height / 2) / height
                lines.append(f"0 {x_center:.6f} {y_center:.6f} {box_width / width:.6f} {box_height / height:.6f}")
                counts[categories[annotation["category_id"]]] += 1
                counts["boxes"] += 1

            if not lines:
                counts["images_without_valid_boxes"] += 1
                continue
            _link_or_copy(src_image, dst_image)
            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            counts["images"] += 1
        split_counts[split] = dict(counts)

    yaml_path = out_root / "pdcnn_bone_loss.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names: ['bone_loss']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "source": "PuckBlink/PDCNN perio-dataset",
        "raw_json": str(coco_path),
        "raw_images_zip": str(raw_root / "Images.zip"),
        "categories": categories,
        "positive_categories": ["mild", "medium", "severe"],
        "total_source_images": len(images),
        "positive_images": len(positive_ids),
        "missing_positive_images": len(missing_images),
        "missing_images_sample": missing_images[:20],
        "split_counts": split_counts,
        "yaml": str(yaml_path),
    }
    (out_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
