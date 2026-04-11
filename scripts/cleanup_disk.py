"""
Free disk space for this repo (Windows-friendly).

1) Removes Hugging Face download cache under data/raw/dentex/.cache (~5GB duplicate
   after DENTEX is already unzipped in data/raw/dentex/DENTEX/).
2) Optional: delete old YOLO run folders under runs/detect/artifacts/detection/.
3) Optional: remove _tmp_cariesxrays_sample.

After (1), re-download via scripts/download_dataset.py only if you delete DENTEX/ too.

To rebuild data/detection images as hardlinks (saves ~1GB vs copies), run:
  python scripts/prepare_detection_dataset.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def rm_tree(path: Path, dry: bool, *, ignore_errors: bool = False) -> int:
    if not path.exists():
        return 0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    if dry:
        print(f"[dry-run] would remove {path} (~{total / (1024**2):.1f} MB)")
        return total
    shutil.rmtree(path, ignore_errors=ignore_errors)
    print(f"Removed {path} (~{total / (1024**2):.1f} MB)" + (" (some locked files skipped)" if ignore_errors else ""))
    return total


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Remove HF cache and optional bulky artifacts")
    parser.add_argument("--yes", "-y", action="store_true", help="No confirmation prompt")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--remove-tmp-sample",
        action="store_true",
        help="Remove _tmp_cariesxrays_sample/",
    )
    parser.add_argument(
        "--prune-old-runs",
        action="store_true",
        help="Delete legacy run dirs: epoch20_default, retry, retry_clean, smoke, yolov8n_dentex (not dentex2/rerun)",
    )
    args = parser.parse_args()

    hf_cache = root / "data" / "raw" / "dentex" / ".cache"
    tmp_sample = root / "_tmp_cariesxrays_sample"
    runs_root = root / "runs" / "detect" / "artifacts" / "detection"
    prune_names = ("epoch20_default", "retry", "retry_clean", "smoke")

    freed = 0
    if not args.yes and not args.dry_run:
        print("This will remove cached HF blobs under data/raw/dentex/.cache (dataset stays in DENTEX/).")
        if input("Continue? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)

    freed += rm_tree(hf_cache, args.dry_run)

    if args.remove_tmp_sample:
        freed += rm_tree(tmp_sample, args.dry_run, ignore_errors=True)

    if args.prune_old_runs and runs_root.is_dir():
        for name in prune_names:
            freed += rm_tree(runs_root / name, args.dry_run)

    print(f"Done. Approx freed (if not dry-run): {freed / (1024**2):.1f} MB")
    if not args.dry_run and hf_cache.parent.exists():
        print(
            "\nIf data/detection/images are full file copies, reclaim ~1GB more:\n"
            "  Remove-Item -Recurse data\\detection\\images\\train,val,test  # or delete data/detection/images\n"
            "  python scripts/prepare_detection_dataset.py"
        )


if __name__ == "__main__":
    main()
