"""Audit source datasets for duplicate images before merging."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    rel_path: str
    split: str
    sha256: str
    dhash: int
    width: int
    height: int


@dataclass(frozen=True)
class DuplicateMatch:
    roboflow_path: str
    roboflow_rel_path: str
    roboflow_split: str
    match_path: str
    match_rel_path: str
    match_split: str
    source: str
    method: str
    distance: int | str
    decision: str


def iter_image_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dhash_image(path: Path, hash_size: int = 8) -> tuple[int, int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(image.getdata())

    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                value |= 1
    return value, width, height


def hamming_distance(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def infer_split(path: Path, root: Path, default: str) -> str:
    rel_parts = path.relative_to(root).parts
    lowered = [part.lower() for part in rel_parts]
    for split in ("train", "val", "valid", "test"):
        if split in lowered:
            return "val" if split == "valid" else split
    return default


def build_records(root: Path, default_split: str) -> tuple[list[ImageRecord], list[dict]]:
    records: list[ImageRecord] = []
    skipped: list[dict] = []
    for path in iter_image_paths(root):
        try:
            dhash, width, height = dhash_image(path)
            digest = sha256_file(path)
        except Exception as exc:  # noqa: BLE001 - audit report should keep moving.
            skipped.append({"path": str(path), "error": repr(exc)})
            continue

        records.append(
            ImageRecord(
                path=str(path.resolve()),
                rel_path=path.relative_to(root).as_posix(),
                split=infer_split(path, root, default_split),
                sha256=digest,
                dhash=dhash,
                width=width,
                height=height,
            )
        )
    return records, skipped


def build_existing_records(existing_root: Path) -> tuple[list[ImageRecord], list[dict]]:
    records: list[ImageRecord] = []
    skipped: list[dict] = []
    for split in ("train", "val", "test"):
        split_root = existing_root / "images" / split
        if not split_root.is_dir():
            continue
        split_records, split_skipped = build_records(split_root, default_split=split)
        records.extend(split_records)
        skipped.extend(split_skipped)
    return records, skipped


def records_by_sha(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = {}
    for record in records:
        grouped.setdefault(record.sha256, []).append(record)
    return grouped


def match_existing(
    existing: list[ImageRecord],
    roboflow: list[ImageRecord],
    near_threshold: int,
    suspect_threshold: int,
) -> tuple[list[DuplicateMatch], list[DuplicateMatch], list[DuplicateMatch]]:
    exact: list[DuplicateMatch] = []
    near: list[DuplicateMatch] = []
    suspect: list[DuplicateMatch] = []
    existing_sha = records_by_sha(existing)

    for rf in roboflow:
        for old in existing_sha.get(rf.sha256, []):
            exact.append(
                DuplicateMatch(
                    roboflow_path=rf.path,
                    roboflow_rel_path=rf.rel_path,
                    roboflow_split=rf.split,
                    match_path=old.path,
                    match_rel_path=old.rel_path,
                    match_split=old.split,
                    source="existing",
                    method="sha256",
                    distance="exact",
                    decision="exclude",
                )
            )

        nearest: tuple[int, ImageRecord] | None = None
        for old in existing:
            distance = hamming_distance(rf.dhash, old.dhash)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, old)

        if nearest is None or nearest[0] > suspect_threshold:
            continue

        distance, old = nearest
        match = DuplicateMatch(
            roboflow_path=rf.path,
            roboflow_rel_path=rf.rel_path,
            roboflow_split=rf.split,
            match_path=old.path,
            match_rel_path=old.rel_path,
            match_split=old.split,
            source="existing",
            method="dhash",
            distance=distance,
            decision="exclude",
        )
        if distance <= near_threshold:
            near.append(match)
        else:
            suspect.append(match)

    return exact, near, suspect


def match_roboflow_internal(
    roboflow: list[ImageRecord],
    near_threshold: int,
    suspect_threshold: int,
) -> tuple[list[DuplicateMatch], list[DuplicateMatch], list[DuplicateMatch]]:
    exact: list[DuplicateMatch] = []
    near: list[DuplicateMatch] = []
    suspect: list[DuplicateMatch] = []

    grouped = records_by_sha(roboflow)
    for group in grouped.values():
        if len(group) <= 1:
            continue
        keeper = sorted(group, key=lambda record: record.rel_path)[0]
        for duplicate in sorted(group, key=lambda record: record.rel_path)[1:]:
            exact.append(
                DuplicateMatch(
                    roboflow_path=duplicate.path,
                    roboflow_rel_path=duplicate.rel_path,
                    roboflow_split=duplicate.split,
                    match_path=keeper.path,
                    match_rel_path=keeper.rel_path,
                    match_split=keeper.split,
                    source="roboflow",
                    method="sha256",
                    distance="exact",
                    decision="exclude",
                )
            )

    for idx, left in enumerate(roboflow):
        nearest: tuple[int, ImageRecord] | None = None
        for right in roboflow[:idx]:
            distance = hamming_distance(left.dhash, right.dhash)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, right)

        if nearest is None or nearest[0] > suspect_threshold:
            continue

        distance, right = nearest
        match = DuplicateMatch(
            roboflow_path=left.path,
            roboflow_rel_path=left.rel_path,
            roboflow_split=left.split,
            match_path=right.path,
            match_rel_path=right.rel_path,
            match_split=right.split,
            source="roboflow",
            method="dhash",
            distance=distance,
            decision="exclude",
        )
        if distance <= near_threshold:
            near.append(match)
        else:
            suspect.append(match)

    return exact, near, suspect


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_montage(matches: list[DuplicateMatch], out_path: Path, max_pairs: int = 24) -> None:
    if not matches:
        return

    thumb_w, thumb_h = 220, 160
    label_h = 54
    gap = 12
    rows = min(len(matches), max_pairs)
    canvas = Image.new("RGB", (thumb_w * 2 + gap, rows * (thumb_h + label_h + gap)), "white")
    draw = ImageDraw.Draw(canvas)

    for row, match in enumerate(matches[:max_pairs]):
        y = row * (thumb_h + label_h + gap)
        for col, image_path in enumerate((match.roboflow_path, match.match_path)):
            x = col * (thumb_w + gap)
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    canvas.paste(image, (x + (thumb_w - image.width) // 2, y))
            except Exception:  # noqa: BLE001 - montage is best-effort.
                draw.rectangle([x, y, x + thumb_w, y + thumb_h], outline="red", width=2)
                draw.text((x + 8, y + 8), "failed to load", fill="red")

        label = (
            f"{match.method}={match.distance} source={match.source}\n"
            f"RF: {match.roboflow_rel_path}\n"
            f"Match: {match.match_split}/{match.match_rel_path}"
        )
        draw.text((0, y + thumb_h + 2), label[:220], fill="black")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def unique_paths(matches: list[DuplicateMatch]) -> set[str]:
    return {match.roboflow_path for match in matches}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Roboflow images against existing YOLO datasets using SHA256 and dHash."
    )
    parser.add_argument("--existing-root", type=Path, default=Path("data/detection_hierarchical"))
    parser.add_argument("--roboflow-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--suspect-threshold", type=int, default=8)
    parser.add_argument("--montage-max-pairs", type=int, default=24)
    args = parser.parse_args()

    if args.near_threshold > args.suspect_threshold:
        raise SystemExit("--near-threshold must be <= --suspect-threshold")

    existing_root = args.existing_root.resolve()
    roboflow_root = args.roboflow_root.resolve()
    if not existing_root.is_dir():
        raise SystemExit(f"Existing dataset root does not exist: {existing_root}")
    if not roboflow_root.is_dir():
        raise SystemExit(f"Roboflow dataset root does not exist: {roboflow_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (args.out or Path("reports/roboflow_audit") / timestamp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_records, existing_skipped = build_existing_records(existing_root)
    roboflow_records, roboflow_skipped = build_records(roboflow_root, default_split="roboflow")

    existing_exact, existing_near, existing_suspect = match_existing(
        existing_records,
        roboflow_records,
        near_threshold=args.near_threshold,
        suspect_threshold=args.suspect_threshold,
    )
    rf_exact, rf_near, rf_suspect = match_roboflow_internal(
        roboflow_records,
        near_threshold=args.near_threshold,
        suspect_threshold=args.suspect_threshold,
    )

    exact_matches = existing_exact + rf_exact
    near_matches = existing_near + rf_near
    suspect_matches = existing_suspect + rf_suspect
    all_matches = exact_matches + near_matches + suspect_matches

    excluded_paths = unique_paths(all_matches)
    kept_records = [record for record in roboflow_records if record.path not in excluded_paths]
    excluded_val_test = {
        match.roboflow_path
        for match in all_matches
        if match.source == "existing" and match.match_split in {"val", "test"}
    }

    write_csv(out_dir / "duplicate_exact.csv", [asdict(match) for match in exact_matches])
    write_csv(out_dir / "duplicate_near.csv", [asdict(match) for match in near_matches])
    write_csv(out_dir / "duplicate_suspect.csv", [asdict(match) for match in suspect_matches])
    write_csv(out_dir / "roboflow_keep.csv", [asdict(record) for record in kept_records])
    write_csv(out_dir / "skipped_images.csv", existing_skipped + roboflow_skipped)

    make_montage(suspect_matches, out_dir / "duplicate_suspect_montage.png", args.montage_max_pairs)
    make_montage(near_matches, out_dir / "duplicate_near_montage.png", args.montage_max_pairs)

    existing_by_split: dict[str, int] = {}
    for record in existing_records:
        existing_by_split[record.split] = existing_by_split.get(record.split, 0) + 1
    roboflow_by_split: dict[str, int] = {}
    for record in roboflow_records:
        roboflow_by_split[record.split] = roboflow_by_split.get(record.split, 0) + 1

    summary = {
        "existing_root": str(existing_root),
        "roboflow_root": str(roboflow_root),
        "output_dir": str(out_dir),
        "thresholds": {
            "near_duplicate_distance_lte": args.near_threshold,
            "suspect_duplicate_distance_lte": args.suspect_threshold,
        },
        "existing_image_count": len(existing_records),
        "existing_image_count_by_split": existing_by_split,
        "roboflow_original_image_count": len(roboflow_records),
        "roboflow_original_image_count_by_split": roboflow_by_split,
        "exact_duplicate_count": len(exact_matches),
        "near_duplicate_count": len(near_matches),
        "suspect_duplicate_count": len(suspect_matches),
        "excluded_roboflow_image_count": len(excluded_paths),
        "final_usable_roboflow_image_count": len(kept_records),
        "existing_val_test_duplicate_roboflow_image_count": len(excluded_val_test),
        "skipped_image_count": len(existing_skipped) + len(roboflow_skipped),
        "outputs": {
            "duplicate_exact_csv": str(out_dir / "duplicate_exact.csv"),
            "duplicate_near_csv": str(out_dir / "duplicate_near.csv"),
            "duplicate_suspect_csv": str(out_dir / "duplicate_suspect.csv"),
            "roboflow_keep_csv": str(out_dir / "roboflow_keep.csv"),
            "skipped_images_csv": str(out_dir / "skipped_images.csv"),
            "duplicate_suspect_montage_png": str(out_dir / "duplicate_suspect_montage.png"),
            "duplicate_near_montage_png": str(out_dir / "duplicate_near_montage.png"),
        },
    }
    (out_dir / "dedupe_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
