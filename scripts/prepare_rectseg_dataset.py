import argparse
import os
import shutil
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_root(data_yaml: Path, config: dict) -> Path:
    root = Path(config.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def bbox_to_rect_segment(line: str) -> str:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected YOLO bbox row with 5 fields, got {len(parts)}: {line!r}")

    cls, x, y, w, h = parts
    x = float(x)
    y = float(y)
    w = float(w)
    h = float(h)

    x1 = max(0.0, x - w / 2)
    y1 = max(0.0, y - h / 2)
    x2 = min(1.0, x + w / 2)
    y2 = min(1.0, y + h / 2)

    return f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert_split(src_root: Path, dst_root: Path, split_rel: str) -> tuple[int, int]:
    image_src_dir = src_root / split_rel
    image_dst_dir = dst_root / split_rel
    label_src_dir = src_root / "labels" / Path(split_rel).name
    label_dst_dir = dst_root / "labels" / Path(split_rel).name

    image_count = 0
    label_count = 0

    for image_path in sorted(image_src_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        link_or_copy(image_path, image_dst_dir / image_path.name)
        image_count += 1

        src_label = label_src_dir / f"{image_path.stem}.txt"
        dst_label = label_dst_dir / src_label.name
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        if not src_label.exists():
            dst_label.write_text("", encoding="utf-8")
            label_count += 1
            continue

        converted = []
        for line in src_label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                converted.append(bbox_to_rect_segment(line))
        dst_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
        label_count += 1

    return image_count, label_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert YOLO bbox labels into rectangle-polygon YOLO segment labels."
    )
    parser.add_argument("--data", default="data/detection_hierarchical/hierarchical_detection.yaml")
    parser.add_argument("--out", default="data/detection_hierarchical_rectseg")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    config = load_yaml(data_yaml)
    src_root = resolve_root(data_yaml, config)
    dst_root = Path(args.out).resolve()

    dst_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split_key in ("train", "val", "test"):
        split_rel = config.get(split_key)
        if split_rel:
            summary[split_key] = convert_split(src_root, dst_root, split_rel)

    out_config = dict(config)
    out_config["path"] = dst_root.as_posix()
    out_yaml = dst_root / f"{data_yaml.stem}_rectseg.yaml"
    out_yaml.write_text(yaml.safe_dump(out_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"Rectangle segment dataset: {dst_root}")
    print(f"YAML: {out_yaml}")
    for split_key, (images, labels) in summary.items():
        print(f"{split_key}: images={images} labels={labels}")


if __name__ == "__main__":
    main()
