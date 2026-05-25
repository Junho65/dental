from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


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


def count_split_labels(label_dir: Path, class_names: list[str]) -> tuple[Counter, int]:
    counts: Counter = Counter()
    image_count = 0
    if not label_dir.exists():
        return counts, image_count

    for label_path in sorted(label_dir.glob("*.txt")):
        image_count += 1
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            if 0 <= class_id < len(class_names):
                counts[class_names[class_id]] += 1
    return counts, image_count


def count_split_labels_with_remap(
    label_dir: Path,
    source_class_names: list[str],
    remap: dict[str, str],
    target_class_names: list[str],
) -> tuple[Counter, int]:
    counts: Counter = Counter()
    image_count = 0
    if not label_dir.exists():
        return counts, image_count

    for label_path in sorted(label_dir.glob("*.txt")):
        image_count += 1
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            if not (0 <= class_id < len(source_class_names)):
                continue
            src_name = source_class_names[class_id]
            dst_name = remap.get(src_name)
            if dst_name in target_class_names:
                counts[dst_name] += 1
    return counts, image_count


def build_rows(counts: Counter, class_names: list[str]) -> list[dict]:
    total = sum(counts.values())
    rows: list[dict] = []
    for name in class_names:
        freq = int(counts.get(name, 0))
        rel = round((freq / total * 100.0), 4) if total else 0.0
        rows.append(
            {
                "label": name,
                "frequency_label": freq,
                "relative_frequency_percent": rel,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-frequency JSON from a YOLO dataset YAML.")
    parser.add_argument("--data", type=Path, help="YOLO dataset YAML path.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Direct labels directory to summarize when a YAML file is unavailable.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        help="Class names to use with --labels-dir mode.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Only summarize a single split when --data is used.",
    )
    parser.add_argument(
        "--remap-json",
        help='Optional JSON object for class remapping, e.g. {"caries":"caries_family","deep_caries":"caries_family"}',
    )
    parser.add_argument(
        "--target-class-names",
        nargs="+",
        help="Target class names to use with --remap-json.",
    )
    args = parser.parse_args()

    if args.data:
        data_path = args.data.resolve()
        config = load_data_config(data_path)
        dataset_root = resolve_dataset_root(data_path, config)
        class_names = get_class_names(config)

        remap = json.loads(args.remap_json) if args.remap_json else None
        target_class_names = list(args.target_class_names) if args.target_class_names else class_names
        split_names = [args.split] if args.split else ["train", "val", "test"]

        splits: dict[str, dict] = {}
        total_counts: Counter = Counter()
        total_images = 0

        for split in split_names:
            split_rel = Path(config[split])
            label_dir = dataset_root / "labels" / split_rel.name
            if remap:
                counts, image_count = count_split_labels_with_remap(
                    label_dir,
                    class_names,
                    remap,
                    target_class_names,
                )
                row_class_names = target_class_names
            else:
                counts, image_count = count_split_labels(label_dir, class_names)
                row_class_names = class_names
            total_counts.update(counts)
            total_images += image_count
            splits[split] = {
                "image_count": image_count,
                "annotation_count": int(sum(counts.values())),
                "rows": build_rows(counts, row_class_names),
            }

        payload = {
            "dataset_yaml": str(data_path),
            "dataset_root": str(dataset_root),
            "class_names": target_class_names if remap else class_names,
            "source_class_names": class_names,
            "applied_remap": remap,
            "metric_definition": {
                "frequency_label": "Number of annotations for each label.",
                "relative_frequency_percent": "Frequency(Label) divided by total annotations in the same scope, multiplied by 100.",
            },
            "overall": {
                "image_count": total_images,
                "annotation_count": int(sum(total_counts.values())),
                "rows": build_rows(total_counts, target_class_names if remap else class_names),
            },
            "splits": splits,
        }
    else:
        if not args.labels_dir or not args.class_names:
            raise SystemExit("Use either --data or both --labels-dir and --class-names.")
        labels_dir = args.labels_dir.resolve()
        class_names = list(args.class_names)
        counts, image_count = count_split_labels(labels_dir, class_names)
        payload = {
            "labels_dir": str(labels_dir),
            "class_names": class_names,
            "metric_definition": {
                "frequency_label": "Number of annotations for each label.",
                "relative_frequency_percent": "Frequency(Label) divided by total annotations in the same scope, multiplied by 100.",
            },
            "scope": {
                "image_count": image_count,
                "annotation_count": int(sum(counts.values())),
                "rows": build_rows(counts, class_names),
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
