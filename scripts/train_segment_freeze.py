import argparse
import os
from pathlib import Path


_yolo_config_root = (Path.cwd() / ".yolo_config").resolve()
_yolo_config_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_yolo_config_root))

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation checkpoint with frozen layers.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--freeze", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="artifacts/segment")
    parser.add_argument("--name", default="nsitnov8024_freeze20_rectseg")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        freeze=args.freeze,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        amp=args.amp,
        task="segment",
        pretrained=True,
    )


if __name__ == "__main__":
    main()
