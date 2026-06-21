"""
Convert CariesXrays (Pascal VOC) to Ultralytics YOLO layout matching DENTEX class order.

CariesXrays (AAAI 2024): https://github.com/Binz-Chen/AAAI2024_CariesXrays
Full images: Google Drive link in repo README / CariesXrays_Dataset(100%).txt

Annotation class in the paper release is typically "Decay" (caries). We map that to
DENTEX class index 0 ("caries"). Other classes are skipped with a warning.

Upstream layout is usually:
  <root>/Annotations/*.xml
  <root>/JEPGImages/*.jpg   # note typo "JEPG" in the published sample
or:
  <root>/JPEGImages/*.jpg

Output uses the same 4 names as dentex_detection.yaml so you can train with
scripts/training/train_detection.py and merge with scripts/data/merge_yolo_detection_datasets.py.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split


CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
CARIES_CLASS_ID = 0

# VOC <name> values seen in CariesXrays / FPCL pascal_voc_classes.json
DECAY_ALIASES = {"decay", "caries", "dental caries"}


def bbox_to_yolo(bbox: List[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return cx, cy, nw, nh


def find_voc_roots(raw: Path) -> Tuple[Path, Path]:
    """Return (annotations_dir, images_dir)."""
    for ann_dir in sorted(raw.rglob("Annotations")):
        if not ann_dir.is_dir():
            continue
        if not any(ann_dir.glob("*.xml")):
            continue
        root = ann_dir.parent
        img_dir: Optional[Path] = None
        for name in ("JEPGImages", "JPEGImages", "jpegimages", "JpegImages"):
            cand = root / name
            if cand.is_dir():
                img_dir = cand
                break
        if img_dir is None:
            for ch in root.iterdir():
                if ch.is_dir() and ch.name.lower() in ("jpegimages", "jepgimages"):
                    img_dir = ch
                    break
        if img_dir is not None:
            return ann_dir, img_dir

    raise FileNotFoundError(
        f"No VOC Annotations/*.xml + JEPGImages|JPEGImages under {raw}. "
        "Download CariesXrays from the Google Drive link in the official repo and unzip here."
    )


def map_voc_name_to_class_id(name: str) -> Optional[int]:
    key = (name or "").strip().lower()
    if key in DECAY_ALIASES:
        return CARIES_CLASS_ID
    return None


def parse_voc_xml(xml_path: Path) -> Optional[Tuple[Path, int, int, List[str]]]:
    tree = ET.parse(xml_path)
    el = tree.getroot()
    fn_el = el.find("filename")
    if fn_el is None or not (fn_el.text or "").strip():
        return None
    filename = (fn_el.text or "").strip()

    size = el.find("size")
    if size is None:
        return None
    w = int(size.findtext("width", "0"))
    h = int(size.findtext("height", "0"))
    if w <= 0 or h <= 0:
        return None

    lines: List[str] = []
    for obj in el.findall("object"):
        difficult = int(obj.findtext("difficult", "0") or "0")
        if difficult == 1:
            continue
        cls_id = map_voc_name_to_class_id(obj.findtext("name", "") or "")
        if cls_id is None:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin", "0"))
        ymin = float(bb.findtext("ymin", "0"))
        xmax = float(bb.findtext("xmax", "0"))
        ymax = float(bb.findtext("ymax", "0"))
        bw = max(xmax - xmin, 0.0)
        bh = max(ymax - ymin, 0.0)
        if bw <= 0 or bh <= 0:
            continue
        cx, cy, nw, nh = bbox_to_yolo([xmin, ymin, bw, bh], float(w), float(h))
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    rel_image = Path(filename)
    return rel_image, w, h, lines


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def collect_entries(ann_dir: Path, img_dir: Path, stem_prefix: str) -> List[Dict]:
    entries: List[Dict] = []
    skipped_img = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        parsed = parse_voc_xml(xml_path)
        if parsed is None:
            continue
        rel_image, _w, _h, lines = parsed
        if not lines:
            continue
        image_path = img_dir / rel_image.name
        if not image_path.exists():
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                alt = img_dir / f"{rel_image.stem}{ext}"
                if alt.exists():
                    image_path = alt
                    break
            else:
                skipped_img += 1
                continue
        stem = f"{stem_prefix}{image_path.stem}"
        entries.append(
            {
                "image_path": image_path,
                "label_lines": lines,
                "out_stem": stem,
                "suffix": image_path.suffix,
            }
        )
    if skipped_img:
        print(f"Warning: skipped {skipped_img} XML files with missing image files under {img_dir}")
    return entries


def copy_entry(entry: Dict, split: str, out_root: Path) -> None:
    img_src: Path = entry["image_path"]
    lines: List[str] = entry["label_lines"]
    stem: str = entry["out_stem"]
    suffix: str = entry["suffix"]
    image_dst = out_root / "images" / split / f"{stem}{suffix}"
    label_dst = out_root / "labels" / split / f"{stem}.txt"
    shutil.copy2(img_src, image_dst)
    label_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CariesXrays VOC -> YOLO (DENTEX class names).")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/cariesxrays"),
        help="Root folder after unzipping the CariesXrays archive (contains Annotations/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/detection_cariesxrays"),
        help="YOLO dataset output directory.",
    )
    parser.add_argument(
        "--stem-prefix",
        default="cx_",
        help="Prefix for output stems to avoid filename clashes when merging with DENTEX.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    ann_dir, img_dir = find_voc_roots(raw_root)
    print(f"Annotations: {ann_dir}")
    print(f"Images: {img_dir}")

    entries = collect_entries(ann_dir, img_dir, args.stem_prefix)
    if not entries:
        raise RuntimeError(
            "No usable image/label pairs. Download the full CariesXrays images from Google Drive; "
            "the GitHub sample often contains XML only."
        )

    train_e, temp_e = train_test_split(entries, test_size=0.3, random_state=args.seed)
    val_e, test_e = train_test_split(temp_e, test_size=2 / 3, random_state=args.seed)

    ensure_dirs(out_root)
    for e in train_e:
        copy_entry(e, "train", out_root)
    for e in val_e:
        copy_entry(e, "val", out_root)
    for e in test_e:
        copy_entry(e, "test", out_root)

    data_yaml = out_root / "cariesxrays_yolo.yaml"
    data_yaml.write_text(
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
    print(f"CariesXrays YOLO dataset -> {out_root}")
    print(f"train={len(train_e)} val={len(val_e)} test={len(test_e)}")
    print(f"YAML: {data_yaml}")
    print("All boxes mapped to class index 0 (caries). Merge with DENTEX via merge_yolo_detection_datasets.py.")


if __name__ == "__main__":
    main()
