"""Prepare severity crop datasets for bone_loss and furcation_involvement lesions.

Reads the manifest CSV from the 2-class periodontal detection dataset, crops
each bounding box from its source image (with optional margin), and writes
train/val/test CSV files consumable by train_severity_classifier.py.

Output structure:
    data/severity_periodontal/
        bone_loss/
            crops/
            train.csv  (image_path, label, source, weight)
            val.csv
            test.csv
            stats.json
        furcation_involvement/
            crops/
            train.csv
            val.csv
            test.csv
            stats.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


DEFAULT_MANIFEST = Path(
    "data/detection_periodontal_pdcnn_2class_bg/periodontal_bbox_manifest.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/severity_periodontal")
DEFAULT_MARGIN = 0.15


def _yolo_to_pixel(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_w: int,
    img_h: int,
    margin: float,
) -> tuple[int, int, int, int]:
    x1 = (x_center - width / 2) * img_w
    y1 = (y_center - height / 2) * img_h
    x2 = (x_center + width / 2) * img_w
    y2 = (y_center + height / 2) * img_h
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(img_w, int(x2 + mx))
    y2 = min(img_h, int(y2 + my))
    return x1, y1, x2, y2


def _process_lesion(
    df: pd.DataFrame,
    lesion_type: str,
    output_dir: Path,
    margin: float,
) -> None:
    lesion_out = output_dir / lesion_type
    crop_dir = lesion_out / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for image_path, group in tqdm(
        df.groupby("image_path"), desc=lesion_type, unit="img"
    ):
        img = Image.open(image_path)
        img_w, img_h = img.size

        for row_idx, row in group.iterrows():
            x1, y1, x2, y2 = _yolo_to_pixel(
                row["x_center"],
                row["y_center"],
                row["width"],
                row["height"],
                img_w,
                img_h,
                margin,
            )
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img.crop((x1, y1, x2, y2))
            stem = Path(image_path).stem
            crop_name = f"{lesion_type}_{stem}_{row_idx:07d}.png"
            crop_path = crop_dir / crop_name
            crop.save(crop_path)

            split = str(row["split"])
            if split not in rows_by_split:
                continue
            rows_by_split[split].append(
                {
                    "image_path": crop_path.relative_to(lesion_out).as_posix(),
                    "label": str(row["original_pdcnn_category"]),
                    "source": "labeled",
                    "weight": 1.0,
                }
            )

    stats: dict[str, dict] = {}
    for split, records in rows_by_split.items():
        split_df = pd.DataFrame(records)
        csv_path = lesion_out / f"{split}.csv"
        split_df.to_csv(csv_path, index=False)
        counts = split_df["label"].value_counts().to_dict() if not split_df.empty else {}
        stats[split] = {"total": len(split_df), "counts": counts}
        print(f"  {split}: {len(split_df)} crops  {counts}")

    (lesion_out / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop periodontal severity patches from the 2-class detection manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to periodontal_bbox_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for severity crop datasets.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Fractional margin to expand each bounding box (default 0.15).",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve() if not args.manifest.is_absolute() else args.manifest
    if not manifest_path.exists():
        # try relative to script location
        manifest_path = Path(__file__).resolve().parents[2] / args.manifest
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    for column in ("image_path", "label_path"):
        if column in df.columns:
            df[column] = df[column].map(
                lambda value: str(
                    Path(value)
                    if Path(value).is_absolute()
                    else manifest_path.parent / Path(value)
                )
            )
    print(f"Loaded manifest: {len(df)} rows from {manifest_path}")

    for lesion_type in ["bone_loss", "furcation_involvement"]:
        sub = df[df["detector_class"] == lesion_type].copy()
        print(f"\n--- {lesion_type}: {len(sub)} annotations ---")
        label_counts = sub["original_pdcnn_category"].value_counts().to_dict()
        print(f"  labels: {label_counts}")
        _process_lesion(sub, lesion_type, args.output_dir.resolve(), args.margin)

    print(f"\nDone. Output: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
