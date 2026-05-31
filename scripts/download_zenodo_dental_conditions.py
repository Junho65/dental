from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = (
    "https://zenodo.org/records/15487430/files/"
    "panoramic_radiography_yolo_dataset_14_classes.zip?download=1"
)
DEFAULT_OUT = Path("data/raw/zenodo/panoramic_radiography_yolo_dataset_14_classes")


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dst.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _extract(zip_path: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the open Zenodo panoramic radiography YOLO dataset."
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--zip", type=Path, default=None, help="Optional zip cache path.")
    parser.add_argument("--keep-zip", action="store_true")
    args = parser.parse_args()

    out = args.out.resolve()
    zip_path = (args.zip or out.with_suffix(".zip")).resolve()
    if out.exists() and any(out.iterdir()):
        print(f"Dataset output already exists, skipping download/extract: {out}")
        return

    print(f"Downloading Zenodo dataset to: {zip_path}")
    _download(args.url, zip_path)
    print(f"Extracting to: {out}")
    _extract(zip_path, out)
    if not args.keep_zip:
        zip_path.unlink()
    print(f"Downloaded and extracted dataset to: {out}")


if __name__ == "__main__":
    main()
