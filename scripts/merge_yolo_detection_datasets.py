"""
Merge two YOLO detection folders that share the same class list and split structure.

Typical flow:
  1) Prepare two YOLO datasets with matching `names` order
  2) python scripts/merge_yolo_detection_datasets.py \
       --base data/detection_main_4class_no_cyst_no_periodontal \
       --extra data/detection_kaggle_pediatric_selected_4class \
       --out data/detection_main_4class_with_pediatric \
       --yaml-name main_4class_with_pediatric.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _class_names(config: dict) -> list[str]:
    names = config.get("names", [])
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return list(names)


def _discover_yaml(root: Path) -> Path:
    yaml_files = sorted(root.glob("*.yaml"))
    if len(yaml_files) != 1:
        raise SystemExit(f"Expected exactly one YAML file under {root}, found {len(yaml_files)}")
    return yaml_files[0]


def _link_or_copy_file(src: Path, dst: Path) -> None:
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


def _copy_split(split: str, src_root: Path, dst_root: Path, stem_prefix: str = "") -> tuple[int, int]:
    img_n = lab_n = 0
    src_img = src_root / "images" / split
    src_lab = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lab = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)

    if not src_img.is_dir():
        return 0, 0

    for image_path in sorted(src_img.iterdir()):
        if not image_path.is_file():
            continue
        src_label = src_lab / f"{image_path.stem}.txt"
        if not src_label.exists():
            continue
        stem = f"{stem_prefix}{image_path.stem}" if stem_prefix else image_path.stem
        out_img = dst_img / f"{stem}{image_path.suffix.lower()}"
        out_lab = dst_lab / f"{stem}.txt"
        _link_or_copy_file(image_path, out_img)
        shutil.copy2(src_label, out_lab)
        img_n += 1
        lab_n += 1
    return img_n, lab_n


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets with the same class order.")
    parser.add_argument("--base", type=Path, required=True, help="Primary dataset root.")
    parser.add_argument("--extra", type=Path, required=True, help="Second dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output merged dataset root.")
    parser.add_argument("--base-yaml", type=Path, default=None, help="Optional explicit base YAML path.")
    parser.add_argument("--extra-yaml", type=Path, default=None, help="Optional explicit extra YAML path.")
    parser.add_argument("--yaml-name", default="merged_detection.yaml", help="Output YAML filename.")
    parser.add_argument(
        "--extra-prefix",
        default="",
        help="Optional filename stem prefix for samples copied from the extra dataset.",
    )
    args = parser.parse_args()

    base_root = args.base.resolve()
    extra_root = args.extra.resolve()
    out_root = args.out.resolve()
    base_yaml = args.base_yaml.resolve() if args.base_yaml is not None else _discover_yaml(base_root)
    extra_yaml = args.extra_yaml.resolve() if args.extra_yaml is not None else _discover_yaml(extra_root)

    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    base_names = _class_names(_load_yaml(base_yaml))
    extra_names = _class_names(_load_yaml(extra_yaml))
    if base_names != extra_names:
        raise SystemExit(
            f"Dataset class order mismatch.\nBase:  {base_names}\nExtra: {extra_names}"
        )

    out_root.mkdir(parents=True, exist_ok=True)

    split_stats: dict[str, dict[str, int]] = {}
    total_img = total_lab = 0
    for split in ("train", "val", "test"):
        base_images, base_labels = _copy_split(split, base_root, out_root, "")
        extra_images, extra_labels = _copy_split(split, extra_root, out_root, args.extra_prefix)
        split_stats[split] = {
            "base_images": base_images,
            "base_labels": base_labels,
            "extra_images": extra_images,
            "extra_labels": extra_labels,
        }
        print(
            f"{split}: base images={base_images} labels={base_labels} | "
            f"extra images={extra_images} labels={extra_labels}"
        )
        total_img += base_images + extra_images
        total_lab += base_labels + extra_labels

    yaml_path = out_root / args.yaml_name
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {base_names}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "base_root": str(base_root),
        "extra_root": str(extra_root),
        "base_yaml": str(base_yaml),
        "extra_yaml": str(extra_yaml),
        "out_root": str(out_root),
        "yaml": str(yaml_path),
        "names": base_names,
        "split_stats": split_stats,
        "totals": {"images": total_img, "labels": total_lab},
    }
    (out_root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Merged {total_img} images ({total_lab} label files written). YAML: {yaml_path}")


if __name__ == "__main__":
    main()
