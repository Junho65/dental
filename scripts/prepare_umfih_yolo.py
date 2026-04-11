"""
Convert UMFIH 14-class YOLO annotations into the project's 4-class detection schema.

Default mapping keeps only semantically safe classes:
  4  Carious lesion       -> caries
  6  Impacted tooth       -> impacted_tooth
  7  Apical periodontitis -> periapical_lesion

Everything else is dropped to avoid introducing label noise.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]
SOURCE_TO_TARGET = {
    4: 0,  # Carious lesion -> caries
    6: 3,  # Impacted tooth -> impacted_tooth
    7: 2,  # Apical periodontitis -> periapical_lesion
}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


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


def remap_label_file(src_label: Path) -> list[str]:
    remapped: list[str] = []
    for raw_line in src_label.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) != 5:
            continue
        source_id = int(parts[0])
        target_id = SOURCE_TO_TARGET.get(source_id)
        if target_id is None:
            continue
        remapped.append(" ".join([str(target_id), *parts[1:]]))
    return remapped


def resolve_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def ensure_dirs(root: Path) -> None:
    for split in SPLIT_MAP.values():
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def prepare_split(source_root: Path, source_split: str, target_split: str, out_root: Path, stem_prefix: str) -> Counter:
    stats: Counter = Counter()
    labels_dir = source_root / source_split / "labels"
    images_dir = source_root / source_split / "images"

    for label_path in sorted(labels_dir.glob("*.txt")):
        remapped_lines = remap_label_file(label_path)
        if not remapped_lines:
            continue
        image_path = resolve_image(images_dir, label_path.stem)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        out_stem = f"{stem_prefix}{label_path.stem}"
        out_image = out_root / "images" / target_split / f"{out_stem}{image_path.suffix.lower()}"
        out_label = out_root / "labels" / target_split / f"{out_stem}.txt"

        _link_or_copy(image_path, out_image)
        out_label.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")

        stats["images"] += 1
        for line in remapped_lines:
            stats[int(line.split()[0])] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 4-class YOLO dataset from UMFIH.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/raw/umfih/extracted"),
        help="Extracted UMFIH root containing train/valid/test.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/detection_umfih"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--stem-prefix",
        default="umfih_",
        help="Prefix added to image/label stems to avoid filename collisions when merging.",
    )
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()

    ensure_dirs(out_root)

    total = Counter()
    for source_split, target_split in SPLIT_MAP.items():
        split_stats = prepare_split(raw_root, source_split, target_split, out_root, args.stem_prefix)
        total.update(split_stats)
        print(
            f"{target_split}: images={split_stats['images']} "
            f"caries={split_stats[0]} deep_caries={split_stats[1]} "
            f"periapical_lesion={split_stats[2]} impacted_tooth={split_stats[3]}"
        )

    yaml_path = out_root / "umfih_detection.yaml"
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

    print(f"UMFIH YOLO dataset prepared at {out_root}")
    print(f"YAML: {yaml_path}")
    print(
        "Totals: "
        f"images={total['images']} "
        f"caries={total[0]} deep_caries={total[1]} "
        f"periapical_lesion={total[2]} impacted_tooth={total[3]} "
        f"missing_images={total['missing_images']}"
    )


if __name__ == "__main__":
    main()
