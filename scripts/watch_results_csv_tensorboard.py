"""
Stream YOLO results.csv into TensorBoard while training (polls file each interval).

Ultralytics appends one row per epoch to results.csv. Log dir should be ASCII-only
on Windows (TensorFlow file I/O can fail on non-ASCII paths).

Terminal 1: python scripts/train_detection.py --name my_run ...
Terminal 2: python scripts/watch_results_csv_tensorboard.py
  (Default --auto-latest follows the run whose epoch count is actually increasing.)
Terminal 2 (explicit): python scripts/watch_results_csv_tensorboard.py --run-name my_run
Terminal 3 (cmd.exe): tensorboard --logdir "%TEMP%\\dental_tensorboard_live" --reload_interval 5
Terminal 3 (PowerShell): tensorboard --logdir "$env:TEMP\\dental_tensorboard_live" --reload_interval 5
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path


def _safe_name(part: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", part) or "run"


def tb_scalar_tag(name: str) -> str:
    """Avoid rare TensorBoard UI issues with '(' / ')' in tag names."""
    return name.replace("(", "_").replace(")", "")


def read_epochs(csv_path: Path) -> dict[int, dict[str, float]]:
    """Return {epoch: {metric_name: value}} from full CSV."""
    if not csv_path.is_file():
        return {}
    import csv

    rows: dict[int, dict[str, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ep = int(float(row["epoch"]))
            except (KeyError, ValueError):
                continue
            metrics: dict[str, float] = {}
            for key, raw in row.items():
                if key in ("epoch", "time") or raw is None or raw == "":
                    continue
                try:
                    metrics[key] = float(raw)
                except ValueError:
                    continue
            rows[ep] = metrics
    return rows


def _scan_results_rows(
    root: Path,
    mtime_cache: dict[str, float],
    max_ep_cache: dict[str, int],
) -> list[tuple[Path, float, int]]:
    """(path, mtime, max_epoch). Skips files with no epoch rows. Cached on mtime."""
    out: list[tuple[Path, float, int]] = []
    for p in root.rglob("results.csv"):
        key = str(p.resolve())
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        if mtime_cache.get(key) == mt and key in max_ep_cache:
            mx = max_ep_cache[key]
        else:
            epochs = read_epochs(p)
            mx = max(epochs) if epochs else 0
            mtime_cache[key] = mt
            max_ep_cache[key] = mx
        if mx <= 0:
            continue
        out.append((p, mt, mx))
    return out


def _select_csv_auto_latest(
    root: Path,
    current: Path | None,
    prev_max_epoch: dict[str, int],
    seeded: bool,
    mtime_cache: dict[str, float],
    max_ep_cache: dict[str, int],
) -> tuple[Path | None, bool, bool]:
    """
    Returns (chosen_path, path_changed, need_seed_only).

    need_seed_only: caller should update prev_max_epoch and skip writing this tick (first poll).
    """
    rows = _scan_results_rows(root, mtime_cache, max_ep_cache)
    if not rows:
        return current, False, False

    if not seeded:
        return current, False, True

    grown: list[tuple[Path, float, int]] = []
    for p, mt, mx in rows:
        key = str(p.resolve())
        old = prev_max_epoch.get(key, -1)
        if mx > old:
            grown.append((p, mt, mx))
        prev_max_epoch[key] = mx

    if grown:
        chosen = max(grown, key=lambda t: t[1])[0]
        changed = current is None or chosen.resolve() != current.resolve()
        return chosen, changed, False

    if current is not None and current.is_file():
        epochs = read_epochs(current)
        if epochs:
            return current, False, False

    chosen = max(rows, key=lambda t: t[1])[0]
    changed = current is None or chosen.resolve() != current.resolve()
    return chosen, changed, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll results.csv -> live TensorBoard scalars")
    parser.add_argument(
        "results_csv",
        type=Path,
        nargs="?",
        default=None,
        help="Explicit path to results.csv",
    )
    parser.add_argument(
        "--auto-latest",
        action="store_true",
        help="Follow runs under --runs-root (default if no path / --run-name).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Watch runs-root/NAME/results.csv (fixed; same as train --name).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/detect/artifacts/detection"),
        help="Root for --auto-latest / --run-name (relative to cwd unless absolute).",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="ASCII-only log directory (default: %%TEMP%%/dental_tensorboard_live)",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between polls")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each poll to stderr.",
    )
    args = parser.parse_args()

    if args.results_csv is not None and (args.auto_latest or args.run_name):
        parser.error("Do not pass results_csv together with --auto-latest or --run-name")
    if args.auto_latest and args.run_name:
        parser.error("Use either --auto-latest or --run-name, not both")

    if args.results_csv is None and args.run_name is None and not args.auto_latest:
        args.auto_latest = True

    root: Path | None = None
    use_auto = False
    auto_seeded = False
    prev_max_epoch: dict[str, int] = {}
    mtime_cache: dict[str, float] = {}
    max_ep_cache: dict[str, int] = {}

    if args.run_name:
        root = args.runs_root.resolve()
        csv_path: Path | None = (root / args.run_name / "results.csv").resolve()
        print(f"[watch_results_csv_tensorboard] --run-name: {csv_path}")
    elif args.auto_latest:
        root = args.runs_root.resolve()
        use_auto = True
        csv_path = None
        print(
            f"[watch_results_csv_tensorboard] --auto-latest under {root} "
            "(picks the run whose epoch count increases; wait ~one interval after start)"
        )
    else:
        csv_path = args.results_csv.resolve()

    import tempfile

    tb_root = (
        args.logdir.resolve()
        if args.logdir
        else Path(tempfile.gettempdir()) / "dental_tensorboard_live"
    )
    tb_root.mkdir(parents=True, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter

    writer: SummaryWriter | None = None
    active_write_dir: Path | None = None
    logged_max = 0
    missing_warned = False
    empty_rows_warned = False

    def attach_writer_for(csv_file: Path) -> None:
        nonlocal writer, active_write_dir, logged_max
        if args.run_name:
            wd = (tb_root / _safe_name(args.run_name)).resolve()
        else:
            wd = (tb_root / _safe_name(csv_file.parent.name)).resolve()
        if active_write_dir == wd and writer is not None:
            return
        if writer is not None:
            writer.close()
        wd.mkdir(parents=True, exist_ok=True)
        active_write_dir = wd
        writer = SummaryWriter(str(wd), flush_secs=1)
        logged_max = 0

    if not use_auto:
        assert csv_path is not None
        attach_writer_for(csv_path)

    if use_auto:
        print("Watching: (auto — locks onto the run whose epoch count increases)")
    else:
        print(f"Watching: {csv_path}")
    print(f"TensorBoard root (use this --logdir): {tb_root}")
    print(f'  tensorboard --logdir "{tb_root}" --reload_interval 2')
    print("Ctrl+C to stop.\n")

    try:
        while True:
            if use_auto and root is not None:
                chosen, changed, need_seed = _select_csv_auto_latest(
                    root, csv_path, prev_max_epoch, auto_seeded, mtime_cache, max_ep_cache
                )
                if need_seed:
                    rows = _scan_results_rows(root, mtime_cache, max_ep_cache)
                    for p, mt, mx in rows:
                        prev_max_epoch[str(p.resolve())] = mx
                    auto_seeded = True
                    if args.verbose:
                        print(
                            "[watch_results_csv_tensorboard] seeded epoch snapshot "
                            f"for {len(rows)} results.csv file(s); next poll uses growth pick",
                            file=sys.stderr,
                            flush=True,
                        )
                    time.sleep(args.interval)
                    continue
                if chosen is None:
                    time.sleep(args.interval)
                    continue
                if changed or writer is None:
                    if csv_path is not None and changed:
                        print(
                            f"[watch_results_csv_tensorboard] following: {chosen}",
                            flush=True,
                        )
                    csv_path = chosen
                    attach_writer_for(chosen)
                    missing_warned = False
                    empty_rows_warned = False

            if csv_path is None or writer is None:
                time.sleep(args.interval)
                continue

            if not csv_path.is_file():
                if not missing_warned:
                    print(
                        f"[watch_results_csv_tensorboard] CSV not found yet: {csv_path}\n"
                        "  (Training will create it after the first epoch — waiting…)",
                        file=sys.stderr,
                    )
                    missing_warned = True
                time.sleep(args.interval)
                continue
            missing_warned = False

            epochs = read_epochs(csv_path)
            if not epochs and logged_max == 0:
                if not empty_rows_warned:
                    print(
                        "[watch_results_csv_tensorboard] CSV exists but has no epoch rows yet — waiting…",
                        file=sys.stderr,
                    )
                    empty_rows_warned = True
            elif epochs:
                empty_rows_warned = False

            if args.verbose and epochs:
                print(
                    f"[watch_results_csv_tensorboard] poll: file={csv_path} "
                    f"max_epoch_in_csv={max(epochs)} logged_max={logged_max}",
                    file=sys.stderr,
                    flush=True,
                )

            prev_logged = logged_max
            for ep in sorted(epochs):
                if ep <= logged_max:
                    continue
                for name, val in epochs[ep].items():
                    writer.add_scalar(tb_scalar_tag(name), val, ep)
                logged_max = max(logged_max, ep)
            if args.verbose and logged_max > prev_logged:
                print(
                    f"[watch_results_csv_tensorboard] wrote new points through epoch {logged_max}",
                    file=sys.stderr,
                    flush=True,
                )
            writer.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
