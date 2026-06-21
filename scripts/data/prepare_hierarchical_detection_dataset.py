"""Prepare the hierarchical detection dataset from a four-class YOLO dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


TARGET_CLASS_NAMES = ["caries_family", "periapical_lesion", "impacted_tooth"]
CLASS_REMAP = {
    "caries": "caries_family",
    "deep_caries": "caries_family",
    "periapical_lesion": "periapical_lesion",
    "impacted_tooth": "impacted_tooth",
}
TARGET_CLASS_TO_ID = {name: idx for idx, name in enumerate(TARGET_CLASS_NAMES)}


def load_data_config(data_path: Path) -> dict:
    return yaml.safe_load(data_path.read_text(encoding="utf-8"))


def resolve_dataset_root(data_path: Path, config: dict) -> Path:
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def get_class_names(config: dict) -> list[str]:
    raw = config.get("names", [])
    if isinstance(raw, dict):
        return [raw[idx] for idx in sorted(raw)]
    return list(raw)


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


def remap_label_file(src_label: Path, src_names: list[str]) -> list[str]:
    lines: list[str] = []
    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        src_class_id = int(parts[0])
        src_class_name = src_names[src_class_id]
        target_class_name = CLASS_REMAP.get(src_class_name)
        if target_class_name is None:
            continue
        target_class_id = TARGET_CLASS_TO_ID[target_class_name]
        lines.append(" ".join([str(target_class_id), *parts[1:]]))
    return lines


def copy_split(split: str, dataset_root: Path, config: dict, out_root: Path, src_names: list[str]) -> Counter:
    stats: Counter = Counter()
    split_rel = Path(config[split])
    if split_rel.suffix.lower() == ".txt":
        raise ValueError("Manifest-based train splits are not supported here. Use the original YAML directories.")

    src_image_dir = dataset_root / split_rel
    src_label_dir = dataset_root / "labels" / split_rel.name
    dst_image_dir = out_root / "images" / split
    dst_label_dir = out_root / "labels" / split
    dst_image_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(src_image_dir.iterdir()):
        if not image_path.is_file():
            continue
        src_label = src_label_dir / f"{image_path.stem}.txt"
        if not src_label.exists():
            continue
        remapped_lines = remap_label_file(src_label, src_names)
        if not remapped_lines:
            continue
        _link_or_copy_image(image_path, dst_image_dir / image_path.name)
        (dst_label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(remapped_lines) + "\n",
            encoding="utf-8",
        )
        stats["images"] += 1
        for line in remapped_lines:
            stats[int(line.split()[0])] += 1
    return stats


def prepare_hierarchical_dataset(data_path: Path, out_root: Path) -> Path:
    data_path = data_path.resolve()
    config = load_data_config(data_path)
    dataset_root = resolve_dataset_root(data_path, config)
    src_names = get_class_names(config)

    out_root = out_root.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    out_root.mkdir(parents=True, exist_ok=True)

    total = Counter()
    for split in ("train", "val", "test"):
        split_stats = copy_split(split, dataset_root, config, out_root, src_names)
        total.update(split_stats)
        print(
            f"{split}: images={split_stats['images']} "
            + " ".join(
                f"{name}={split_stats[idx]}"
                for idx, name in enumerate(TARGET_CLASS_NAMES)
            )
        )

    yaml_path = out_root / "hierarchical_detection.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {TARGET_CLASS_NAMES}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Hierarchical detection dataset prepared at {out_root}")
    print(f"YAML: {yaml_path}")
    print(
        "Totals: "
        + " ".join(f"{name}={total[idx]}" for idx, name in enumerate(TARGET_CLASS_NAMES))
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse caries/deep_caries into caries_family for hierarchical detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/detection_merged/merged_detection.yaml"),
        help="Source YOLO YAML path with the original 4-class layout.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/detection_hierarchical"),
        help="Output YOLO dataset root for hierarchical detection.",
    )
    args = parser.parse_args()

    prepare_hierarchical_dataset(data_path=args.data, out_root=args.out)


if __name__ == "__main__":
    main()
