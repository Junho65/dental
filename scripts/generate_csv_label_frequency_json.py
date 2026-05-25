from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def build_rows(counts: Counter, class_names: list[str]) -> list[dict]:
    total = sum(counts.values())
    rows: list[dict] = []
    for name in class_names:
        freq = int(counts.get(name, 0))
        rel = round((freq / total * 100.0), 4) if total else 0.0
        rows.append(
            {
                "label": name,
                "frequency_label": freq,
                "relative_frequency_percent": rel,
            }
        )
    return rows


def load_counts(csv_path: Path, label_column: str, class_names: list[str]) -> tuple[Counter, int]:
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        raise ValueError(f"{csv_path} must contain '{label_column}' column.")
    counts = Counter(df[label_column].astype(str).tolist())
    for name in class_names:
        counts.setdefault(name, 0)
    return counts, len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label-frequency JSON from classification CSV splits.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    class_names = list(args.class_names)
    splits = {}
    total_counts: Counter = Counter()
    total_samples = 0

    for split_name, csv_path in {
        "train": args.train_csv.resolve(),
        "val": args.val_csv.resolve(),
        "test": args.test_csv.resolve(),
    }.items():
        counts, sample_count = load_counts(csv_path, args.label_column, class_names)
        total_counts.update(counts)
        total_samples += sample_count
        splits[split_name] = {
            "csv_path": str(csv_path),
            "sample_count": sample_count,
            "rows": build_rows(counts, class_names),
        }

    payload = {
        "class_names": class_names,
        "label_column": args.label_column,
        "metric_definition": {
            "frequency_label": "Number of samples for each label.",
            "relative_frequency_percent": "Frequency(Label) divided by total samples in the same scope, multiplied by 100.",
        },
        "overall": {
            "sample_count": total_samples,
            "rows": build_rows(total_counts, class_names),
        },
        "splits": splits,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
