from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "caries_family",
    "periapical_lesion",
    "impacted_tooth",
    "bone_loss",
    "retained_root",
]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


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


def _copy_split(src_root: Path, out_root: Path, split: str, stem_prefix: str) -> Counter:
    stats: Counter = Counter()
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    if not src_images.is_dir() or not src_labels.is_dir():
        return stats

    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for label_path in sorted(src_labels.glob("*.txt")):
        image_path = None
        for suffix in IMAGE_SUFFIXES:
            candidate = src_images / f"{label_path.stem}{suffix}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            stats["missing_images"] += 1
            continue

        out_stem = f"{stem_prefix}{label_path.stem}"
        _link_or_copy(image_path, out_images / f"{out_stem}{image_path.suffix.lower()}")
        shutil.copy2(label_path, out_labels / f"{out_stem}.txt")
        stats["images"] += 1
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().lstrip("\ufeff").split()
            if len(parts) >= 5:
                try:
                    stats[CLASS_NAMES[int(float(parts[0]))]] += 1
                except (IndexError, ValueError):
                    stats["invalid_labels"] += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing 3-class hierarchical data with a 5-class extension dataset.")
    parser.add_argument("--base", type=Path, default=Path("data/detection_hierarchical"))
    parser.add_argument("--extra", type=Path, default=Path("data/detection_zenodo_5class"))
    parser.add_argument("--out", type=Path, default=Path("data/detection_hierarchical_zenodo_5class"))
    args = parser.parse_args()

    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    total: Counter = Counter()
    for split in ("train", "val", "test"):
        base_stats = _copy_split(args.base.resolve(), out_root, split, "base_")
        extra_stats = _copy_split(args.extra.resolve(), out_root, split, "zenodo_")
        total.update(base_stats)
        total.update(extra_stats)
        print(
            f"{split}: base_images={base_stats['images']} extra_images={extra_stats['images']} "
            + " ".join(f"{name}={base_stats[name] + extra_stats[name]}" for name in CLASS_NAMES)
        )

    yaml_path = out_root / "hierarchical_zenodo_5class.yaml"
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
    print(f"Merged dataset prepared at {out_root}")
    print(f"YAML: {yaml_path}")
    print("Totals: " + " ".join(f"{name}={total[name]}" for name in CLASS_NAMES))


if __name__ == "__main__":
    main()
