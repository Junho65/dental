import argparse
import json
import os
from pathlib import Path


_yolo_config_root = (Path.cwd() / ".yolo_config").resolve()
_yolo_config_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_yolo_config_root))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detect/segment weights using box metrics.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--task", choices=["detect", "segment"], default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = YOLO(args.weights)
    val_kwargs = {"data": args.data, "imgsz": args.imgsz}
    if args.task is not None:
        val_kwargs["task"] = args.task
    metrics = model.val(**val_kwargs)

    out = {
        "weights": args.weights,
        "data": args.data,
        "imgsz": args.imgsz,
        "task": args.task,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
