"""
Merge two YOLO detection folders that share the same class list (e.g. DENTEX + CariesXrays).

Both sources must use identical `names` order (default: DENTEX four classes).
Typical flow:
  1) python scripts/prepare_detection_dataset.py
  2) python scripts/prepare_cariesxrays_yolo.py --stem-prefix cx_
  3) python scripts/merge_yolo_detection_datasets.py \\
       --base data/detection --extra data/detection_cariesxrays --out data/detection_merged

Train with:  ultralytics ... data=data/detection_merged/merged_detection.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]


def _link_or_copy_image(src: Path, dst: Path) -> None:
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


def copy_split(split: str, src_root: Path, dst_root: Path, stem_prefix: str = "") -> tuple[int, int]:
    img_n = lab_n = 0
    src_img = src_root / "images" / split
    src_lab = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lab = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)

    if not src_img.is_dir():
        return 0, 0

    for p in sorted(src_img.iterdir()):
        if not p.is_file():
            continue
        src_label = src_lab / f"{p.stem}.txt"
        if not src_label.exists():
            continue
        stem = f"{stem_prefix}{p.stem}" if stem_prefix else p.stem
        out_img = dst_img / f"{stem}{p.suffix}"
        out_lab = dst_lab / f"{stem}.txt"
        _link_or_copy_image(p, out_img)
        shutil.copy2(src_label, out_lab)
        img_n += 1
        lab_n += 1
    return img_n, lab_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets with the same class order.")
    parser.add_argument("--base", type=Path, required=True, help="Primary dataset root (e.g. data/detection).")
    parser.add_argument("--extra", type=Path, required=True, help="Second dataset (e.g. data/detection_cariesxrays).")
    parser.add_argument("--out", type=Path, required=True, help="Output root (e.g. data/detection_merged).")
    parser.add_argument(
        "--extra-prefix",
        default="",
        help="Optional extra prefix on stems from --extra (usually unnecessary if CariesXrays already uses cx_).",
    )
    args = parser.parse_args()

    base = args.base.resolve()
    extra = args.extra.resolve()
    out = args.out.resolve()

    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Output {out} is not empty. Choose a new folder or delete it first.")

    out.mkdir(parents=True, exist_ok=True)

    total_img = total_lab = 0
    for split in ("train", "val", "test"):
        bi, bl = copy_split(split, base, out, "")
        ei, el = copy_split(split, extra, out, args.extra_prefix)
        print(f"{split}: base images={bi} labels={bl} | extra images={ei} labels={el}")
        total_img += bi + ei
        total_lab += bl + el

    yaml_path = out / "merged_detection.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {CLASS_NAMES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged {total_img} images ({total_lab} label files written). YAML: {yaml_path}")


if __name__ == "__main__":
    main()
