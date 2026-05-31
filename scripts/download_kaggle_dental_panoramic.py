from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_DATASET = "lokisilvres/dental-disease-panoramic-detection-dataset"
DEFAULT_OUT = Path("data/raw/kaggle/dental_disease_panoramic_detection")


def _kaggle_token_exists() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    return token_path.is_file()


def _require_kaggle() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit(
            "Kaggle CLI was not found on PATH. Install it with `pip install kaggle` "
            "and configure Kaggle API credentials before running this pipeline."
        )
    if not _kaggle_token_exists():
        raise SystemExit(
            "Kaggle API credentials were not found. Create an API token on Kaggle and place it at "
            "%USERPROFILE%\\.kaggle\\kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY."
        )


def _run_kaggle(args: list[str]) -> None:
    command = ["kaggle", *args]
    print("Running: " + " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(f"Kaggle command failed with exit code {completed.returncode}: {' '.join(command)}")


def _extract_archives(out_root: Path) -> None:
    for archive in sorted(out_root.glob("*.zip")):
        marker = out_root / f".extracted_{archive.stem}"
        if marker.exists():
            continue
        print(f"Extracting {archive} -> {out_root}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_root)
        marker.write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle dental panoramic detection dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="Pass --force to Kaggle and re-extract archives.")
    args = parser.parse_args()

    _require_kaggle()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    command = ["datasets", "download", "-d", args.dataset, "-p", str(out_root)]
    if args.force:
        command.append("--force")
        for marker in out_root.glob(".extracted_*"):
            marker.unlink()
    _run_kaggle(command)
    _extract_archives(out_root)

    print(f"Kaggle dataset ready: {out_root}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
