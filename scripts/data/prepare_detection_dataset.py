"""Convert DENTEX annotations into the project's YOLO detection layout."""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def map_from_validation_category(name: str) -> str:
    key = name.lower()
    if "deep" in key:
        return "deep_caries"
    if "caries" in key:
        return "caries"
    if "periapical" in key:
        return "periapical_lesion"
    if "impact" in key:
        return "impacted_tooth"
    return ""


def map_from_test_label(label_text: str) -> str:
    key = label_text.lower()
    if "çürük" in key or "caries" in key:
        return "caries"
    if "kanal" in key or "periapical" in key:
        return "periapical_lesion"
    if "gömülü" in key or "impacted" in key:
        return "impacted_tooth"
    return ""


def bbox_to_yolo(bbox: List[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return cx, cy, nw, nh


def ensure_dirs(root: Path):
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_validation_entries(raw_root: Path) -> List[Dict]:
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
    anns_by_image: Dict[int, List[str]] = {}

    for ann in payload["annotations"]:
        cls = map_from_validation_category(disease_map.get(ann.get("category_id_3", -1), ""))
        if not cls:
            continue
        img = images[ann["image_id"]]
        cx, cy, w, h = bbox_to_yolo(ann["bbox"], img["width"], img["height"])
        anns_by_image.setdefault(ann["image_id"], []).append(
            f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        )

    entries = []
    for image_id, lines in anns_by_image.items():
        img = images[image_id]
        img_path = val_img_dir / img["file_name"]
        if img_path.exists():
            entries.append({"image_path": img_path, "label_lines": lines})
    return entries


def parse_test_entries(raw_root: Path) -> List[Dict]:
    label_dir = raw_root / "DENTEX" / "test_data" / "disease" / "label"
    img_dir = raw_root / "DENTEX" / "test_data" / "disease" / "input"
    entries = []
    for jf in sorted(label_dir.glob("*.json")):
        payload = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        image_name = payload.get("imagePath", f"{jf.stem}.png")
        image_path = img_dir / image_name
        if not image_path.exists():
            continue
        lines = []
        for shape in payload.get("shapes", []):
            cls = map_from_test_label(str(shape.get("label", "")))
            if not cls:
                continue
            points = shape.get("points", [])
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            width = max(float(payload.get("imageWidth", 1)), 1.0)
            height = max(float(payload.get("imageHeight", 1)), 1.0)
            cx, cy, w, h = bbox_to_yolo([x_min, y_min, x_max - x_min, y_max - y_min], width, height)
            lines.append(f"{CLASS_TO_ID[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            entries.append({"image_path": image_path, "label_lines": lines})
    return entries


def _link_or_copy_image(src: Path, dst: Path) -> None:
    """Hardlink when same volume (saves ~1GB vs full copies of DENTEX images)."""
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


def copy_entry(entry: Dict, split: str, out_root: Path):
    image_path: Path = entry["image_path"]
    lines: List[str] = entry["label_lines"]
    image_dst = out_root / "images" / split / image_path.name
    label_dst = out_root / "labels" / split / f"{image_path.stem}.txt"
    _link_or_copy_image(image_path, image_dst)
    label_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    raw_root = Path("data/raw/dentex")
    out_root = Path("data/detection")
    ensure_dirs(out_root)

    entries = []
    entries.extend(parse_validation_entries(raw_root))
    entries.extend(parse_test_entries(raw_root))
    if not entries:
        raise RuntimeError("No detection entries parsed from dataset.")

    train_entries, temp_entries = train_test_split(entries, test_size=0.3, random_state=42)
    val_entries, test_entries = train_test_split(temp_entries, test_size=2 / 3, random_state=42)

    for e in train_entries:
        copy_entry(e, "train", out_root)
    for e in val_entries:
        copy_entry(e, "val", out_root)
    for e in test_entries:
        copy_entry(e, "test", out_root)

    data_yaml = out_root / "dentex_detection.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {CLASS_NAMES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Detection dataset prepared at {out_root}")
    print(f"train={len(train_entries)} val={len(val_entries)} test={len(test_entries)}")
    print(f"YAML: {data_yaml}")


if __name__ == "__main__":
    main()
