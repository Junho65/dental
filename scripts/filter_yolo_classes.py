from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _class_names(config: dict) -> list[str]:
    names = config.get("names", [])
    if isinstance(names, dict):
        return [names[idx] for idx in sorted(names)]
    return list(names)


def _dataset_root(yaml_path: Path, config: dict) -> Path:
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = yaml_path.parent / root
    return root.resolve()


def _resolve_split(root: Path, value: str | list[str]) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    paths: list[Path] = []
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
        parts[lowered.index("images")] = "labels"
        return Path(*parts)
    return image_dir.parent / "labels"


def _image_for_label(image_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter/remap YOLO classes into a new dataset.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-names", required=True, help="Comma-separated class names to keep in output order.")
    parser.add_argument("--yaml-name", default="filtered_detection.yaml")
    args = parser.parse_args()

    config = _load_yaml(args.data.resolve())
    source_root = _dataset_root(args.data.resolve(), config)
    source_names = _class_names(config)
    keep_names = [name.strip() for name in args.keep_names.split(",") if name.strip()]
    unknown = sorted(set(keep_names) - set(source_names))
    if unknown:
        raise SystemExit(f"Unknown keep names {unknown}; source names are {source_names}")

    out_root = args.out.resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise SystemExit(f"Output {out_root} is not empty. Choose a new folder or delete it first.")

    old_to_new = {source_names.index(name): idx for idx, name in enumerate(keep_names)}
    split_counts: dict[str, dict[str, int]] = {}
    totals: Counter = Counter()

    for split in ("train", "val", "test"):
        counts: Counter = Counter()
        if split not in config:
            continue
        for image_dir in _resolve_split(source_root, config[split]):
            label_dir = _label_dir_for_image_dir(image_dir)
            if not image_dir.is_dir() or not label_dir.is_dir():
                counts["missing_split_dirs"] += 1
                continue
            for label_path in sorted(label_dir.glob("*.txt")):
                image_path = _image_for_label(image_dir, label_path.stem)
                if image_path is None:
                    counts["missing_images"] += 1
                    continue

                remapped: list[str] = []
                for raw_line in label_path.read_text(encoding="utf-8").splitlines():
                    parts = raw_line.strip().lstrip("\ufeff").split()
                    if len(parts) < 5:
                        counts["invalid_label_lines"] += 1
                        continue
                    try:
                        old_id = int(float(parts[0]))
                    except ValueError:
                        counts["invalid_label_lines"] += 1
                        continue
                    if old_id not in old_to_new:
                        counts[f"dropped_{source_names[old_id] if 0 <= old_id < len(source_names) else old_id}"] += 1
                        continue
                    new_id = old_to_new[old_id]
                    remapped.append(" ".join([str(new_id), *parts[1:]]))
                    counts[keep_names[new_id]] += 1

                if not remapped:
                    counts["images_without_kept_boxes"] += 1
                    continue

                out_image = out_root / "images" / split / f"{label_path.stem}{image_path.suffix.lower()}"
                out_label = out_root / "labels" / split / f"{label_path.stem}.txt"
                _link_or_copy(image_path, out_image)
                out_label.parent.mkdir(parents=True, exist_ok=True)
                out_label.write_text("\n".join(remapped) + "\n", encoding="utf-8")
                counts["images"] += 1

        split_counts[split] = dict(counts)
        totals.update(counts)

    yaml_path = out_root / args.yaml_name
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                f"names: {keep_names}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = {
        "source_data": str(args.data.resolve()),
        "source_names": source_names,
        "out_root": str(out_root),
        "yaml": str(yaml_path),
        "target_names": keep_names,
        "old_to_new": {source_names[old_id]: new_id for old_id, new_id in old_to_new.items()},
        "splits": split_counts,
        "totals": dict(totals),
    }
    (out_root / "filter_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
