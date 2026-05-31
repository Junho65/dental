from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _class_names(config: dict) -> list[str]:
    raw = config.get("names", [])
    if isinstance(raw, dict):
        return [raw[idx] for idx in sorted(raw)]
    return list(raw)


def _dataset_root(yaml_path: Path, config: dict) -> Path:
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _split_image_dirs(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = []
    for raw in values:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        paths.append(path.resolve())
    return paths


def _label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels"


def _image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _read_label(path: Path, class_count: int) -> tuple[list[int], list[dict]]:
    class_ids: list[int] = []
    problems: list[dict] = []
    if not path.exists():
        return class_ids, [{"issue": "missing_label"}]
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = raw_line.strip().lstrip("\ufeff").split()
        if not parts:
            continue
        if len(parts) < 5:
            problems.append({"issue": "short_label_line", "line": line_number, "value": raw_line})
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append({"issue": "non_numeric_label", "line": line_number, "value": raw_line})
            continue
        if class_id < 0 or class_id >= class_count:
            problems.append({"issue": "class_id_out_of_range", "line": line_number, "class_id": class_id})
        if len(coords) == 4:
            x_center, y_center, width, height = coords
            if not all(0.0 <= value <= 1.0 for value in coords):
                problems.append({"issue": "bbox_out_of_normalized_range", "line": line_number, "coords": coords})
            if width <= 0 or height <= 0:
                problems.append({"issue": "bbox_non_positive_size", "line": line_number, "coords": coords})
            if width < 0.001 or height < 0.001:
                problems.append({"issue": "bbox_tiny", "line": line_number, "coords": coords})
            if x_center - width / 2 < 0 or x_center + width / 2 > 1 or y_center - height / 2 < 0 or y_center + height / 2 > 1:
                problems.append({"issue": "bbox_extends_outside_image", "line": line_number, "coords": coords})
        elif len(coords) >= 6 and len(coords) % 2 == 0:
            if not all(0.0 <= value <= 1.0 for value in coords):
                problems.append({"issue": "polygon_out_of_normalized_range", "line": line_number})
        else:
            problems.append({"issue": "unsupported_coordinate_count", "line": line_number, "coord_count": len(coords)})
        class_ids.append(class_id)
    return class_ids, problems


def _summarize_sizes(sizes: list[tuple[int, int]]) -> dict:
    if not sizes:
        return {"count": 0}
    widths = [item[0] for item in sizes]
    heights = [item[1] for item in sizes]
    aspects = [round(w / h, 4) for w, h in sizes if h]
    return {
        "count": len(sizes),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
        "aspect_min": min(aspects) if aspects else None,
        "aspect_max": max(aspects) if aspects else None,
        "unique_size_count": len(set(sizes)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a YOLO detection dataset for image and label readiness.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--require-all-classes-all-splits", action="store_true")
    args = parser.parse_args()

    data_path = args.data.resolve()
    config = _load_yaml(data_path)
    names = _class_names(config)
    root = _dataset_root(data_path, config)
    out_dir = (args.out or data_path.parent / "audit").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_rows: list[dict] = []
    split_counts: dict[str, dict[str, int]] = {}
    image_sizes_by_split: dict[str, list[tuple[int, int]]] = defaultdict(list)
    label_issue_counts: Counter = Counter()
    image_counts: Counter = Counter()
    label_counts: Counter = Counter()

    for split in ("train", "val", "test"):
        if split not in config:
            continue
        counts = Counter()
        for image_dir in _split_image_dirs(root, config[split]):
            label_dir = _label_dir_for_image_dir(image_dir)
            seen_images = set()
            for image_path in _image_paths(image_dir):
                seen_images.add(image_path.stem)
                image_counts[split] += 1
                try:
                    with Image.open(image_path) as image:
                        image = ImageOps.exif_transpose(image)
                        image_sizes_by_split[split].append(image.size)
                except Exception as exc:  # noqa: BLE001
                    label_issue_counts["unreadable_image"] += 1
                    bad_rows.append({"split": split, "path": str(image_path), "issue": "unreadable_image", "detail": repr(exc)})
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                class_ids, problems = _read_label(label_path, len(names))
                for class_id in class_ids:
                    if 0 <= class_id < len(names):
                        counts[names[class_id]] += 1
                        label_counts[split] += 1
                for problem in problems:
                    label_issue_counts[problem["issue"]] += 1
                    bad_rows.append({"split": split, "path": str(label_path), **problem})

            if label_dir.is_dir():
                for label_path in sorted(label_dir.glob("*.txt")):
                    if label_path.stem not in seen_images:
                        label_issue_counts["label_without_image"] += 1
                        bad_rows.append({"split": split, "path": str(label_path), "issue": "label_without_image"})
        split_counts[split] = {name: counts[name] for name in names}

    missing_required: dict[str, list[str]] = {}
    if args.require_all_classes_all_splits:
        for split, counts in split_counts.items():
            missing = [name for name in names if counts.get(name, 0) < 1]
            if missing:
                missing_required[split] = missing

    image_size_summary = {split: _summarize_sizes(sizes) for split, sizes in image_sizes_by_split.items()}
    label_audit_summary = {
        "data": str(data_path),
        "root": str(root),
        "names": names,
        "image_counts": dict(image_counts),
        "label_counts": dict(label_counts),
        "label_issue_counts": dict(label_issue_counts),
        "missing_required_classes": missing_required,
    }

    (out_dir / "image_size_summary.json").write_text(json.dumps(image_size_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "label_audit_summary.json").write_text(json.dumps(label_audit_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "split_class_counts.json").write_text(json.dumps(split_counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "bad_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in bad_rows for key in row} | {"split", "path", "issue"})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_rows)

    print(json.dumps(label_audit_summary, indent=2, ensure_ascii=False))
    if missing_required:
        raise SystemExit(f"Required class coverage failed: {missing_required}")


if __name__ == "__main__":
    main()
