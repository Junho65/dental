from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


TARGET_CLASS_NAMES = [
    "caries_family",
    "periapical_lesion",
    "impacted_tooth",
    "bone_loss",
    "cyst",
    "retained_root",
]
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class SourceDataset:
    root: Path
    yaml_path: Path
    names: list[str]
    prefix: str


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    source: str
    original_split: str
    classes: frozenset[int]
    digest: str


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


def _resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        path = Path(item)
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


def _image_for_label(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_remapped_label(label_path: Path, source_names: list[str]) -> tuple[list[str], frozenset[int]]:
    lines: list[str] = []
    classes: set[int] = set()
    name_to_target = {name: idx for idx, name in enumerate(TARGET_CLASS_NAMES)}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().lstrip("\ufeff").split()
        if len(parts) < 5:
            continue
        try:
            source_id = int(float(parts[0]))
        except ValueError:
            continue
        if source_id < 0 or source_id >= len(source_names):
            continue
        source_name = source_names[source_id]
        if source_name not in name_to_target:
            continue
        target_id = name_to_target[source_name]
        lines.append(" ".join([str(target_id), *parts[1:]]))
        classes.add(target_id)
    return lines, frozenset(classes)


def _load_dataset(yaml_path: Path, prefix: str) -> SourceDataset:
    config = _load_yaml(yaml_path)
    return SourceDataset(root=_dataset_root(yaml_path, config), yaml_path=yaml_path, names=_class_names(config), prefix=prefix)


def _collect_records(dataset: SourceDataset, scratch_labels: Path) -> list[ImageRecord]:
    config = _load_yaml(dataset.yaml_path)
    records: list[ImageRecord] = []
    for split in ("train", "val", "test"):
        if split not in config:
            continue
        for image_dir in _resolve_split(dataset.root, config[split]):
            label_dir = _label_dir_for_image_dir(image_dir)
            if not image_dir.is_dir() or not label_dir.is_dir():
                continue
            for label_path in sorted(label_dir.glob("*.txt")):
                image_path = _image_for_label(image_dir, label_path.stem)
                if image_path is None:
                    continue
                remapped, classes = _read_remapped_label(label_path, dataset.names)
                if not remapped:
                    continue
                safe_stem = f"{dataset.prefix}_{split}_{label_path.stem}"
                scratch_path = scratch_labels / f"{safe_stem}.txt"
                scratch_path.parent.mkdir(parents=True, exist_ok=True)
                scratch_path.write_text("\n".join(remapped) + "\n", encoding="utf-8")
                records.append(
                    ImageRecord(
                        image_path=image_path,
                        label_path=scratch_path,
                        source=dataset.prefix,
                        original_split=split,
                        classes=classes,
                        digest=_sha256(image_path),
                    )
                )
    return records


def _group_by_digest(records: list[ImageRecord]) -> list[list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.digest].append(record)
    return list(grouped.values())


def _group_classes(group: list[ImageRecord]) -> set[int]:
    result: set[int] = set()
    for record in group:
        result.update(record.classes)
    return result


def _assign_groups(groups: list[list[ImageRecord]], seed: int) -> dict[int, str]:
    rng = random.Random(seed)
    indexed = list(enumerate(groups))
    class_to_groups: dict[int, list[int]] = defaultdict(list)
    for idx, group in indexed:
        for class_id in _group_classes(group):
            class_to_groups[class_id].append(idx)

    missing_source = [TARGET_CLASS_NAMES[class_id] for class_id in range(len(TARGET_CLASS_NAMES)) if len(class_to_groups[class_id]) < 3]
    if missing_source:
        raise SystemExit(
            "Not enough usable image groups to cover train/val/test for classes: "
            + ", ".join(missing_source)
        )

    assignments: dict[int, str] = {}
    split_order = ("val", "test", "train")
    for class_id in sorted(class_to_groups, key=lambda item: len(class_to_groups[item])):
        candidates = [idx for idx in class_to_groups[class_id] if idx not in assignments]
        rng.shuffle(candidates)
        for split in split_order:
            if any(assignments.get(idx) == split and class_id in _group_classes(groups[idx]) for idx in assignments):
                continue
            if not candidates:
                raise SystemExit(f"Could not assign class {TARGET_CLASS_NAMES[class_id]} to split {split}.")
            assignments[candidates.pop()] = split

    target_fraction = {"train": 0.8, "val": 0.1, "test": 0.1}
    target_counts = {split: max(1, round(len(groups) * fraction)) for split, fraction in target_fraction.items()}
    remaining = [idx for idx, _ in indexed if idx not in assignments]
    rng.shuffle(remaining)
    for idx in remaining:
        current_counts = Counter(assignments.values())
        split = min(target_fraction, key=lambda item: current_counts[item] / target_counts[item])
        assignments[idx] = split
    return assignments


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


def _write_group(group: list[ImageRecord], split: str, out_root: Path, used_stems: Counter) -> Counter:
    counts: Counter = Counter()
    for record in group:
        base_stem = f"{record.source}_{record.original_split}_{record.image_path.stem}"
        used_stems[base_stem] += 1
        out_stem = base_stem if used_stems[base_stem] == 1 else f"{base_stem}_{used_stems[base_stem]}"
        out_image = out_root / "images" / split / f"{out_stem}{record.image_path.suffix.lower()}"
        out_label = out_root / "labels" / split / f"{out_stem}.txt"
        _link_or_copy(record.image_path, out_image)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record.label_path, out_label)
        counts["images"] += 1
        for raw_line in record.label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if parts:
                class_id = int(parts[0])
                counts[TARGET_CLASS_NAMES[class_id]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into a 6-class class-aware train/val/test split.")
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--extra-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/detection_hierarchical_zenodo_kaggle_6class"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")
    scratch_labels = out_root / "_scratch_labels"

    base = _load_dataset(args.base_data.resolve(), "base")
    extra = _load_dataset(args.extra_data.resolve(), "kaggle")
    records = _collect_records(base, scratch_labels) + _collect_records(extra, scratch_labels)
    if not records:
        raise SystemExit("No usable records were found.")

    groups = _group_by_digest(records)
    assignments = _assign_groups(groups, seed=args.seed)

    total_counts: Counter = Counter()
    split_counts: dict[str, dict[str, int]] = {}
    used_stems: Counter = Counter()
    for split in ("train", "val", "test"):
        split_total: Counter = Counter()
        for idx, group in enumerate(groups):
            if assignments[idx] != split:
                continue
            split_total.update(_write_group(group, split, out_root, used_stems))
        split_counts[split] = {name: split_total[name] for name in TARGET_CLASS_NAMES}
        split_counts[split]["images"] = split_total["images"]
        total_counts.update(split_total)

    missing = {
        split: [name for name in TARGET_CLASS_NAMES if split_counts[split].get(name, 0) < 1]
        for split in ("train", "val", "test")
    }
    missing = {split: names for split, names in missing.items() if names}
    if missing:
        raise SystemExit(f"Class coverage failed after split assignment: {missing}")

    yaml_path = out_root / "hierarchical_zenodo_kaggle_6class.yaml"
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
    if scratch_labels.exists():
        shutil.rmtree(scratch_labels, ignore_errors=True)

    summary = {
        "base_data": str(args.base_data.resolve()),
        "extra_data": str(args.extra_data.resolve()),
        "out_root": str(out_root),
        "yaml": str(yaml_path),
        "target_names": TARGET_CLASS_NAMES,
        "source_image_records": len(records),
        "duplicate_digest_groups": len(groups),
        "split_class_counts": split_counts,
        "totals": {name: total_counts[name] for name in TARGET_CLASS_NAMES} | {"images": total_counts["images"]},
    }
    (out_root / "split_class_counts.json").write_text(json.dumps(split_counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_root / "merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
