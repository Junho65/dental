"""
Write YOLO results.csv rows into TensorBoard scalar logs.

Use when a run folder has no events.out.tfevents* (TensorBoard was off during training).
Then: tensorboard --logdir <logdir>
"""

from __future__ import annotations

import argparse
import csv
import re
import tempfile
from pathlib import Path


def tb_scalar_tag(name: str) -> str:
    return name.replace("(", "_").replace(")", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="results.csv -> TensorBoard event files")
    parser.add_argument(
        "results_csv",
        type=Path,
        help="Path to runs/.../results.csv",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help=(
            "Output directory for event files. "
            "Default: %%TEMP%%/dental_tensorboard/<run_name> (ASCII path; avoids TF bugs on non-ASCII project paths)."
        ),
    )
    args = parser.parse_args()

    csv_path = args.results_csv.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"Not found: {csv_path}")

    if args.logdir is not None:
        logdir = args.logdir.resolve()
    else:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", csv_path.parent.name) or "run"
        logdir = Path(tempfile.gettempdir()) / "dental_tensorboard" / safe_name
    logdir.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(str(logdir), flush_secs=1)
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row["epoch"]))
            for key, raw in row.items():
                if key in ("epoch", "time"):
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                writer.add_scalar(tb_scalar_tag(key), val, epoch)
    writer.close()
    print(f"Wrote TensorBoard logs to: {logdir}")
    print(f"Run: tensorboard --logdir \"{logdir}\"")


if __name__ == "__main__":
    main()
