from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_WORKSPACE = "opg-unbmz"
DEFAULT_PROJECT = "dental-x-ray-panoramic"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"
DEFAULT_OUT = Path("data/raw/roboflow/dental-x-ray-panoramic")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Roboflow Universe dataset export.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--format", default=DEFAULT_FORMAT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
        help="Environment variable that contains the Roboflow API key.",
    )
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"{args.api_key_env} is not set. Set it before downloading, for example:\n"
            f"  $env:{args.api_key_env} = '<your-roboflow-api-key>'"
        )

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise SystemExit(
            "The 'roboflow' package is not installed. Install it with:\n"
            "  python -m pip install roboflow"
        ) from exc

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    dataset = (
        rf.workspace(args.workspace)
        .project(args.project)
        .version(args.version)
        .download(args.format, location=str(out))
    )
    print(f"Downloaded Roboflow dataset to: {dataset.location}")


if __name__ == "__main__":
    main()
