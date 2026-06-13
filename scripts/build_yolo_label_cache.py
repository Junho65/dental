from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import yaml
from ultralytics.data.dataset import DATASET_CACHE_VERSION, img2label_paths, verify_image_label
from ultralytics.data.utils import IMG_FORMATS, get_hash


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


def _image_files(image_dir: Path) -> list[str]:
    files = glob.glob(str(image_dir / "**" / "*.*"), recursive=True)
    return sorted(x.replace("/", os.sep) for x in files if x.rpartition(".")[-1].lower() in IMG_FORMATS)


def _write_cache_file(path: Path, cache: dict) -> None:
    cache["version"] = DATASET_CACHE_VERSION
    with open(path, "wb") as file:
        np.save(file, cache)


def _build_cache_for_split(split: str, image_dir: Path, label_dir: Path, class_count: int, refresh: bool) -> dict:
    im_files = _image_files(image_dir)
    label_files = img2label_paths(im_files)
    cache_path = label_dir.with_suffix(".cache")
    if cache_path.exists() and not refresh:
        return {
            "split": split,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "cache_path": str(cache_path),
            "images": len(im_files),
            "skipped_existing": True,
        }

    x = {"labels": []}
    nm = nf = ne = nc = 0
    msgs: list[str] = []

    for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in (
        verify_image_label((im_file, lb_file, "", False, class_count, 0, 0, False))
        for im_file, lb_file in zip(im_files, label_files)
    ):
        nm += nm_f
        nf += nf_f
        ne += ne_f
        nc += nc_f
        if im_file:
            x["labels"].append(
                {
                    "im_file": im_file,
                    "shape": shape,
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": segments,
                    "keypoints": keypoint,
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )
        if msg:
            msgs.append(msg)

    x["hash"] = get_hash(label_files + im_files)
    x["results"] = (nf, nm, ne, nc, len(im_files))
    x["msgs"] = msgs
    _write_cache_file(cache_path, x)
    return {
        "split": split,
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "cache_path": str(cache_path),
        "images": len(im_files),
        "found": nf,
        "missing": nm,
        "empty": ne,
        "corrupt": nc,
        "messages": len(msgs),
        "skipped_existing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Ultralytics label cache files sequentially.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated dataset splits to cache. Missing splits are skipped.",
    )
    parser.add_argument("--refresh", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    yaml_path = args.data.resolve()
    config = _load_yaml(yaml_path)
    root = _dataset_root(yaml_path, config)
    names = config.get("names", [])
    class_count = len(names if isinstance(names, list) else names.keys())
    requested_splits = [split.strip() for split in args.splits.split(",") if split.strip()]

    summary = {
        "data": str(yaml_path),
        "root": str(root),
        "splits": [],
    }
    for split in requested_splits:
        if split not in config:
            continue
        for image_dir in _resolve_split(root, config[split]):
            label_dir = image_dir.parent.parent / "labels" / image_dir.name
            if not image_dir.is_dir() or not label_dir.is_dir():
                continue
            summary["splits"].append(_build_cache_for_split(split, image_dir, label_dir, class_count, args.refresh))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
