"""Download and extract the DENTEX source files used by this project."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "LUNA0206/DENTEX"
REQUIRED_FILES = (
    "README.md",
    "DENTEX/validation_triple.json",
    "DENTEX/validation_data.zip",
    "DENTEX/test_data.zip",
)
EXPECTED_EXTRACTED_PATHS = (
    Path("DENTEX/validation_data/validation_data/quadrant_enumeration_disease/xrays"),
    Path("DENTEX/test_data/disease/input"),
    Path("DENTEX/test_data/disease/label"),
)


def extract_if_needed(zip_path: Path, expected_paths: tuple[Path, ...], force: bool) -> None:
    if all(path.exists() for path in expected_paths) and not force:
        print(f"Already extracted: {zip_path.parent}")
        return
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(zip_path.parent)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the DENTEX files used by this project."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/dentex"),
        help="Local Hugging Face download directory.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download the archives without extracting them.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the archives even when the expected directories already exist.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: dict[str, Path] = {}
    for filename in REQUIRED_FILES:
        local_path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                local_dir=str(args.output_dir),
            )
        )
        downloaded[filename] = local_path
        print(f"Downloaded: {filename} -> {local_path}")

    if not args.skip_extract:
        extract_if_needed(
            downloaded["DENTEX/validation_data.zip"],
            (args.output_dir / EXPECTED_EXTRACTED_PATHS[0],),
            args.force_extract,
        )
        extract_if_needed(
            downloaded["DENTEX/test_data.zip"],
            tuple(args.output_dir / path for path in EXPECTED_EXTRACTED_PATHS[1:]),
            args.force_extract,
        )
        missing = [path for path in EXPECTED_EXTRACTED_PATHS if not (args.output_dir / path).exists()]
        if missing:
            formatted = "\n".join(f"- {args.output_dir / path}" for path in missing)
            raise RuntimeError(f"DENTEX extraction is incomplete; missing:\n{formatted}")

    status = "archives downloaded" if args.skip_extract else "ready"
    print(f"DENTEX is {status} under {args.output_dir}.")
    print("License: CC-BY-NC-SA-4.0 (confirm the dataset card before redistribution).")


if __name__ == "__main__":
    main()
