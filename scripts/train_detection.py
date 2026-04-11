import argparse
import math
import random
import re
import tempfile
from collections import Counter
from pathlib import Path

import yaml
from ultralytics import YOLO, settings as ultralytics_settings


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _safe_run_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name) or "run"


def _tensorboard_tag(key: str) -> str:
    return str(key).replace("(", "_").replace(")", "")


def load_data_config(data_path: Path) -> dict:
    return yaml.safe_load(data_path.read_text(encoding="utf-8"))


def get_class_names(config: dict) -> list[str]:
    raw_names = config.get("names", [])
    if isinstance(raw_names, dict):
        return [raw_names[idx] for idx in sorted(raw_names)]
    return list(raw_names)


def resolve_dataset_root(data_path: Path, config: dict) -> Path:
    root = Path(config.get("path", data_path.parent))
    if not root.is_absolute():
        root = (data_path.parent / root).resolve()
    return root


def read_class_ids(label_path: Path) -> list[int]:
    if not label_path.exists():
        return []
    class_ids = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            class_ids.append(int(parts[0]))
    return class_ids


def derive_repeat_factor(label_dir: Path, target_class_id: int) -> tuple[int, Counter, int]:
    counts: Counter = Counter()
    target_image_count = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        class_ids = read_class_ids(label_path)
        counts.update(class_ids)
        if target_class_id in class_ids:
            target_image_count += 1

    target_count = counts.get(target_class_id, 0)
    non_zero_counts = sorted((count for count in counts.values() if count > 0), reverse=True)
    if target_count == 0 or len(non_zero_counts) < 2:
        return 1, counts, target_image_count

    # Match the second-largest class rather than the dominant one to avoid excessive duplication.
    desired_count = non_zero_counts[1]
    repeat_factor = max(1, math.ceil(desired_count / target_count))
    return repeat_factor, counts, target_image_count


def build_balanced_training_data(data_path: Path, target_class_name: str) -> Path:
    config = load_data_config(data_path)
    dataset_root = resolve_dataset_root(data_path, config)
    class_names = get_class_names(config)
    if target_class_name not in class_names:
        print(f"Class balancing skipped: '{target_class_name}' is not present in dataset names {class_names}.")
        return data_path
    target_class_id = class_names.index(target_class_name)

    train_rel = Path(config["train"])
    if train_rel.suffix.lower() == ".txt":
        print(f"Using existing train manifest without extra balancing: {train_rel}")
        return data_path

    train_image_dir = dataset_root / train_rel
    train_label_dir = dataset_root / "labels" / train_rel.name
    repeat_factor, counts, target_image_count = derive_repeat_factor(train_label_dir, target_class_id)

    if repeat_factor <= 1:
        print(f"Class balancing skipped: '{target_class_name}' train split is already balanced enough.")
        return data_path

    manifest: list[str] = []
    oversampled_images = 0
    for image_path in sorted(train_image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        label_path = train_label_dir / f"{image_path.stem}.txt"
        class_ids = read_class_ids(label_path)
        manifest.append(image_path.resolve().as_posix())
        if target_class_id in class_ids:
            oversampled_images += 1
            manifest.extend([image_path.resolve().as_posix()] * (repeat_factor - 1))

    random.Random(42).shuffle(manifest)

    artifact_dir = Path("artifacts/detection/training_assets")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_class_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", target_class_name)
    manifest_path = artifact_dir / f"{data_path.stem}_{safe_class_name}_x{repeat_factor}_train.txt"
    manifest_path.write_text("\n".join(manifest) + "\n", encoding="utf-8")

    balanced_config = dict(config)
    balanced_config["train"] = manifest_path.resolve().as_posix()
    balanced_yaml = artifact_dir / f"{data_path.stem}_{safe_class_name}_x{repeat_factor}.yaml"
    balanced_yaml.write_text(
        yaml.safe_dump(balanced_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(
        f"Class balancing enabled for '{target_class_name}': "
        f"class_counts={dict(counts)} target_images={target_image_count} "
        f"repeat_factor={repeat_factor} oversampled_images={oversampled_images}"
    )
    print(f"Balanced train manifest: {manifest_path}")
    print(f"Balanced data config: {balanced_yaml}")
    return balanced_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/detection/dentex_detection.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping: epochs with no fitness improvement before stopping (Ultralytics).",
    )
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="artifacts/detection")
    parser.add_argument("--name", default="yolov8n_dentex")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Ultralytics AMP/autocast. Disable on low-VRAM GPUs to skip AMP checks and reduce memory pressure.",
    )
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log scalars each epoch via torch.utils.tensorboard.SummaryWriter to %%TEMP%%/dental_yolo_tb/<run> (ASCII path).",
    )
    parser.add_argument(
        "--deep-caries-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Oversample train images that contain the configured target class to reduce class imbalance.",
    )
    parser.add_argument(
        "--oversample-class",
        default="deep_caries",
        help="Class name to oversample when --deep-caries-balance is enabled. Defaults to deep_caries.",
    )
    args = parser.parse_args()

    # Ultralytics' built-in TB writes under trainer.save_dir (often breaks on non-ASCII Windows paths).
    # We always disable it here and optionally attach PyTorch SummaryWriter to an ASCII-only logdir instead.
    ultralytics_settings["tensorboard"] = False

    data_path = Path(args.data)
    if args.deep_caries_balance:
        data_path = build_balanced_training_data(data_path, target_class_name=args.oversample_class)

    model = YOLO(args.model)

    tb_writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_runs_root = Path(tempfile.gettempdir()) / "dental_yolo_tb"
        run_logdir = tb_runs_root / _safe_run_name(args.name)
        run_logdir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(run_logdir), flush_secs=1)
        batch_step = 0

        def on_train_batch_end(trainer):
            nonlocal batch_step
            batch_step += 1
            for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
                try:
                    tb_writer.add_scalar(_tensorboard_tag(k), float(v), batch_step)
                except (TypeError, ValueError):
                    pass
            if batch_step % 10 == 0:
                tb_writer.flush()

        def on_train_epoch_end(trainer):
            step = trainer.epoch + 1
            for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
                try:
                    tb_writer.add_scalar(_tensorboard_tag(k), float(v), step)
                except (TypeError, ValueError):
                    pass
            for k, v in getattr(trainer, "lr", {}).items():
                try:
                    tb_writer.add_scalar(_tensorboard_tag(k), float(v), step)
                except (TypeError, ValueError):
                    pass
            tb_writer.flush()

        def on_fit_epoch_end(trainer):
            step = trainer.epoch + 1
            for k, v in trainer.metrics.items():
                try:
                    tb_writer.add_scalar(_tensorboard_tag(k), float(v), step)
                except (TypeError, ValueError):
                    pass
            tb_writer.flush()

        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        print(
            "TensorBoard: logging training loss each batch and validation/LR metrics each epoch "
            "(see https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html).\n"
            f'  tensorboard --logdir "{tb_runs_root}" --reload_interval 2\n'
            f"  Open http://localhost:6006/ - run folder: {_safe_run_name(args.name)}\n"
        )

    train_kw = dict(
        data=str(data_path),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        plots=args.plots,
    )
    try:
        model.train(**train_kw)
    finally:
        if tb_writer is not None:
            tb_writer.close()


if __name__ == "__main__":
    main()
