from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


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
    resolved = []
    for item in values:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        resolved.append(path.resolve())
    return resolved


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


def _yolo_bbox_to_xyxy(parts: list[str], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    width, height = image_size
    _, cx, cy, bw, bh = [float(value) for value in parts[:5]]
    box_w = bw * width
    box_h = bh * height
    center_x = cx * width
    center_y = cy * height
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return x1, y1, x2, y2


def _expand_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(width, int(round(x2 + margin_x)))
    bottom = min(height, int(round(y2 + margin_y)))
    return left, top, right, bottom


def _crop_and_save(
    image: Image.Image,
    bbox_xyxy: tuple[float, float, float, float],
    out_path: Path,
    margin_ratio: float,
) -> None:
    left, top, right, bottom = _expand_bbox(*bbox_xyxy, *image.size, margin_ratio)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop((left, top, right, bottom)).save(out_path)


def _prepare_split(
    split: str,
    image_dirs: list[Path],
    source_names: list[str],
    target_names: set[str],
    out_root: Path,
    margin_ratio: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for image_dir in image_dirs:
        label_dir = _label_dir_for_image_dir(image_dir)
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            image_path = _image_for_label(image_dir, label_path.stem)
            if image_path is None:
                continue

            image = Image.open(image_path).convert("RGB")
            for box_idx, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                parts = raw_line.strip().lstrip("\ufeff").split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                if class_id < 0 or class_id >= len(source_names):
                    continue

                class_name = source_names[class_id]
                if class_name not in target_names:
                    continue

                bbox_xyxy = _yolo_bbox_to_xyxy(parts, image.size)
                out_path = (
                    out_root
                    / "images"
                    / split
                    / class_name
                    / f"{image_path.stem}_box{box_idx}.png"
                )
                _crop_and_save(image, bbox_xyxy, out_path, margin_ratio)
                rows.append(
                    {
                        "image_path": str(out_path.resolve()),
                        "label": class_name,
                        "source": "labeled",
                        "weight": 1.0,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"{split}.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare crop-classification CSVs from selected YOLO detection classes."
    )
    parser.add_argument("--data", type=Path, required=True, help="YOLO dataset YAML.")
    parser.add_argument(
        "--class-names",
        nargs="+",
        required=True,
        help="Class names to export as crop labels.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory for CSVs and crops.")
    parser.add_argument("--margin-ratio", type=float, default=0.15)
    args = parser.parse_args()

    yaml_path = args.data.resolve()
    config = _load_yaml(yaml_path)
    source_names = _class_names(config)
    target_names = set(args.class_names)
    unknown = sorted(target_names - set(source_names))
    if unknown:
        raise SystemExit(f"Unknown class names {unknown}; source names are {source_names}")

    dataset_root = _dataset_root(yaml_path, config)
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        if split not in config:
            continue
        df = _prepare_split(
            split=split,
            image_dirs=_resolve_split(dataset_root, config[split]),
            source_names=source_names,
            target_names=target_names,
            out_root=out_root,
            margin_ratio=args.margin_ratio,
        )
        stats[split] = {
            "total": len(df),
            "counts": df["label"].value_counts().to_dict() if not df.empty else {},
        }

    summary = {
        "source_data": str(yaml_path),
        "source_names": source_names,
        "class_names": list(args.class_names),
        "out_root": str(out_root),
        "margin_ratio": args.margin_ratio,
        "splits": stats,
    }
    (out_root / "stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
