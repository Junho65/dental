from __future__ import annotations

import argparse
import shutil
import time
import urllib.request
from pathlib import Path


UMFIH_ZIP_URL = (
    "https://zenodo.org/records/15487430/files/"
    "panoramic_radiography_yolo_dataset_14_classes.zip?download=1"
)


def download_with_resume(url: str, output_path: Path, retries: int = 8, chunk_size: int = 1024 * 1024) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        existing_size = output_path.stat().st_size if output_path.exists() else 0
        headers = {}
        mode = "ab" if existing_size > 0 else "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", None)
                if status == 416:
                    return
                if existing_size > 0 and status == 200:
                    output_path.unlink(missing_ok=True)
                    existing_size = 0
                    mode = "wb"

                with output_path.open(mode) as handle:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
            return
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Download failed after {retries} attempts: {exc}") from exc
            wait_seconds = min(5 * attempt, 30)
            print(f"Download attempt {attempt}/{retries} failed: {exc}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Extraction skipped; directory already populated: {extract_dir}")
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(extract_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the UMFIH Dental Pathology YOLO dataset from Zenodo.")
    parser.add_argument(
        "--out-zip",
        type=Path,
        default=Path("data/raw/umfih/panoramic_radiography_yolo_dataset_14_classes.zip"),
        help="Where to store the downloaded zip file.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data/raw/umfih/extracted"),
        help="Where to extract the dataset.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download only; do not extract the zip.",
    )
    args = parser.parse_args()

    print(f"Downloading UMFIH dataset to {args.out_zip} ...")
    download_with_resume(UMFIH_ZIP_URL, args.out_zip)
    print(f"Saved zip: {args.out_zip}")
    if not args.skip_extract:
        extract_zip(args.out_zip, args.extract_dir)
        print(f"Extracted dataset to {args.extract_dir}")


if __name__ == "__main__":
    main()
