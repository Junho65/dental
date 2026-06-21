"""Prepare a normalized six-class dataset from a Roboflow export."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


CLASS_NAMES = [
    "caries_family",
    "periapical_lesion",
    "impacted_tooth",
    "bone_loss",
    "cyst",
    "retained_root",
]
NAME_REMAP = {
    "car": "caries_family",
    "caries": "caries_family",
    "carious lesion": "caries_family",
    "cavity": "caries_family",
    "decay": "caries_family",
    "api": "periapical_lesion",
    "periapical lesion": "periapical_lesion",
    "apical periodontitis": "periapical_lesion",
    "periapical radiolucency": "periapical_lesion",
    "imt": "impacted_tooth",
    "impacted tooth": "impacted_tooth",
    "impaction": "impacted_tooth",
    "bon": "bone_loss",
    "bone loss": "bone_loss",
    "bone resorbtion": "bone_loss",
    "bone resorption": "bone_loss",
    "bone defect": "bone_loss",
    "periodontal bone loss": "bone_loss",
    "cyst": "cyst",
    "radicular cyst": "cyst",
    "rot": "retained_root",
    "retained root": "retained_root",
    "root piece": "retained_root",
    "root fragment": "retained_root",
    "root_fragment_retained_root": "retained_root",
}

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _normalize_name(name: str) -> str:
    return " ".join(name.replace("-", " ").replace("_", " ").strip().lower().split())


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_legend_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    pairs = []
    for raw_part in text.replace("[", "").replace("]", "").split("),"):
        if "(" not in raw_part:
            continue
        part = raw_part.split("(", 1)[1].strip().rstrip(")")
        pieces = part.split(",")
        if len(pieces) < 2:
            continue
        name = pieces[0].strip().strip("'\"")
        try:
            class_id = int(pieces[1].strip())
        except ValueError:
            continue
        pairs.append((class_id, name))
    if not pairs:
        raise SystemExit(f"Could not parse class names from legend: {path}")
    max_id = max(class_id for class_id, _ in pairs)
    names = ["" for _ in range(max_id + 1)]
    for class_id, name in pairs:
        names[class_id] = name
    return names


def _find_metadata(raw_root: Path) -> tuple[Path | None, Path | None]:
    data_yaml = raw_root / "data.yaml"
    legend_txt = raw_root / "legend.txt"
    if data_yaml.exists() or legend_txt.exists():
        return data_yaml if data_yaml.exists() else None, legend_txt if legend_txt.exists() else None

    yaml_candidates = sorted(raw_root.rglob("data.yaml"))
    if yaml_candidates:
        return yaml_candidates[0], None
    legend_candidates = sorted(raw_root.rglob("legend.txt"))
    if legend_candidates:
        return None, legend_candidates[0]
    return None, None


def _get_names(config: dict) -> list[str]:
    raw = config.get("names", [])
    if isinstance(raw, dict):
        return [raw[idx] for idx in sorted(raw)]
    return list(raw)


def _resolve_root(raw_root: Path, config: dict) -> Path:
    root = Path(config.get("path", raw_root))
    if not root.is_absolute():
        root = (raw_root / root).resolve()
    return root


def _resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
    for item in values:
        path = Path(item)
        if not path.exists():
            raw_parts = [part for part in item.replace("\\", "/").split("/") if part]
            parts = [part.lower() for part in raw_parts]
            split_idx = next(
                (idx for idx, part in enumerate(parts) if part in {"train", "valid", "val", "test"}),
                None,
            )
            if split_idx is not None:
                path = root.joinpath(*raw_parts[split_idx:])
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


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


def _label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels"


def _resolve_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _load_keep_paths(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    keep: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("path") or row.get("roboflow_path")
            if value:
                keep.add(str(Path(value).resolve()))
    return keep


def _remap_line(
    raw_line: str,
    source_names: list[str],
    target_to_id: dict[str, int],
    stats: Counter,
) -> str | None:
    parts = raw_line.strip().lstrip("\ufeff").split()
    if len(parts) < 5:
        stats["invalid_label_lines"] += 1
        return None

    try:
        source_id = int(float(parts[0]))
    except ValueError:
        stats["invalid_label_lines"] += 1
        return None

    if source_id < 0 or source_id >= len(source_names):
        stats["unknown_source_ids"] += 1
        return None

    target_name = NAME_REMAP.get(_normalize_name(source_names[source_id]))
    if target_name is None:
        stats["dropped_boxes"] += 1
        return None
    if target_name not in target_to_id:
        stats["dropped_boxes_outside_target_schema"] += 1
        return None

    target_id = target_to_id[target_name]
    coords = parts[1:]
    if len(coords) == 4:
        return " ".join([str(target_id), *coords])

    # YOLO segmentation polygons can be converted to their enclosing box for detection.
    try:
        numbers = [float(value) for value in coords]
    except ValueError:
        stats["invalid_label_lines"] += 1
        return None
    if len(numbers) < 6 or len(numbers) % 2:
        stats["invalid_label_lines"] += 1
        return None
    xs = numbers[0::2]
    ys = numbers[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    stats["polygon_boxes_converted"] += 1
    return f"{target_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def _prepare_split(
    split: str,
    image_dirs: list[Path],
    source_names: list[str],
    target_names: list[str],
    target_to_id: dict[str, int],
    out_root: Path,
    keep_paths: set[str] | None,
    stem_prefix: str,
) -> Counter:
    stats: Counter = Counter()
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_dir in image_dirs:
        label_dir = _label_dir_for_image_dir(image_dir)
        if not image_dir.is_dir() or not label_dir.is_dir():
            stats["missing_split_dirs"] += 1
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _resolve_image(image_dir, label_path.stem)
            if image_path is None:
                stats["missing_images"] += 1
                continue
            if keep_paths is not None and str(image_path.resolve()) not in keep_paths:
                stats["excluded_by_keep_csv"] += 1
                continue

            remapped = [
                line
                for raw_line in label_path.read_text(encoding="utf-8").splitlines()
                if (line := _remap_line(raw_line, source_names, target_to_id, stats)) is not None
            ]
            if not remapped:
                stats["images_without_target_boxes"] += 1
                continue

            out_stem = f"{stem_prefix}{split}_{label_path.stem}"
            out_image = out_images / f"{out_stem}{image_path.suffix.lower()}"
            out_label = out_labels / f"{out_stem}.txt"
            _link_or_copy(image_path, out_image)
            out_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")

            stats["images"] += 1
            for line in remapped:
                stats[target_names[int(line.split()[0])]] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap Roboflow dental X-ray YOLO labels to a 6-class lesion schema.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/roboflow/dental-x-ray-panoramic"))
    parser.add_argument("--out", type=Path, default=Path("data/detection_roboflow_6class"))
    parser.add_argument("--keep-csv", type=Path, default=None, help="Optional roboflow_keep.csv from duplicate audit.")
    parser.add_argument("--stem-prefix", default="rf_")
    parser.add_argument(
        "--target-names",
        default=",".join(CLASS_NAMES),
        help="Comma-separated target class names. Defaults to the full 6-class schema.",
    )
    args = parser.parse_args()

    raw_root = args.raw.resolve()
    out_root = args.out.resolve()
    data_yaml, legend_txt = _find_metadata(raw_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    if data_yaml is not None and data_yaml.exists():
        config = _load_yaml(data_yaml)
        source_root = _resolve_root(data_yaml.parent, config)
        source_names = _get_names(config)
    elif legend_txt is not None and legend_txt.exists():
        config = {
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        }
        source_root = legend_txt.parent
        source_names = _load_legend_names(legend_txt)
    else:
        raise SystemExit(f"Neither data.yaml nor legend.txt was found under: {raw_root}")
    target_names = [name.strip() for name in args.target_names.split(",") if name.strip()]
    unknown_targets = sorted(set(target_names) - set(CLASS_NAMES))
    if unknown_targets:
        raise SystemExit(f"Unknown target names: {unknown_targets}. Allowed: {CLASS_NAMES}")
    target_to_id = {name: idx for idx, name in enumerate(target_names)}
    keep_paths = _load_keep_paths(args.keep_csv.resolve() if args.keep_csv else None)

    out_root.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    split_summary: dict[str, dict[str, int]] = {}
    for source_split, target_split in SPLIT_MAP.items():
        if source_split not in config:
            continue
        image_dirs = _resolve_split(source_root, config[source_split])
        if source_split == "test":
            actual_test_dir = source_root / "test" / "images"
            if actual_test_dir.is_dir():
                image_dirs = [actual_test_dir.resolve()]
        stats = _prepare_split(
            target_split,
            image_dirs,
            source_names,
            target_names,
            target_to_id,
            out_root,
            keep_paths,
            args.stem_prefix,
        )
        total.update(stats)
        split_summary[target_split] = dict(stats)
        print(
            f"{target_split}: images={stats['images']} "
            + " ".join(f"{name}={stats[name]}" for name in target_names)
            + f" dropped_boxes={stats['dropped_boxes']}"
        )

    yaml_path = out_root / "roboflow_6class.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {target_names}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "source_names": source_names,
        "target_names": target_names,
        "splits": split_summary,
        "totals": dict(total),
        "yaml": str(yaml_path),
    }
    (out_root / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Roboflow 6-class dataset prepared at {out_root}")
    print(f"YAML: {yaml_path}")
    print("Totals: " + " ".join(f"{name}={total[name]}" for name in target_names))
    print(
        "Dropped/excluded: "
        f"dropped_boxes={total['dropped_boxes']} "
        f"dropped_boxes_outside_target_schema={total['dropped_boxes_outside_target_schema']} "
        f"images_without_target_boxes={total['images_without_target_boxes']} "
        f"excluded_by_keep_csv={total['excluded_by_keep_csv']}"
    )


if __name__ == "__main__":
    main()
