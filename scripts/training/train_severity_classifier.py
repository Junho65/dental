"""Train a lesion-crop severity classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.severity.dataset import SeverityCropDataset
from src.severity.model import (
    configure_head_only_finetuning,
    DEFAULT_SEVERITY_MODEL_NAME,
    DEFAULT_XRV_WEIGHTS,
    build_severity_model,
)


def _load_labeled_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if "label" not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'label' column.")
    if "source" not in df.columns:
        df["source"] = "labeled"
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["image_path"] = df["image_path"].map(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else (csv_path.parent / Path(value)).resolve()
        )
    )
    keep_columns = ["image_path", "label", "source", "weight"]
    return df[keep_columns]


def _load_pseudo_csv(pseudo_csv: Path | None, pseudo_weight_scale: float) -> pd.DataFrame:
    if pseudo_csv is None:
        return pd.DataFrame(columns=["image_path", "label", "source", "weight"])

    pseudo_df = pd.read_csv(pseudo_csv).copy()
    if pseudo_df.empty:
        return pd.DataFrame(columns=["image_path", "label", "source", "weight"])
    if "label" not in pseudo_df.columns:
        raise ValueError(f"{pseudo_csv} must contain a 'label' column.")

    confidence = pseudo_df["confidence"] if "confidence" in pseudo_df.columns else 1.0
    pseudo_df["weight"] = confidence.astype(float) * pseudo_weight_scale
    if "source" not in pseudo_df.columns:
        pseudo_df["source"] = "pseudo"
    pseudo_df["image_path"] = pseudo_df["image_path"].map(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else (pseudo_csv.parent / Path(value)).resolve()
        )
    )
    keep_columns = ["image_path", "label", "source", "weight"]
    return pseudo_df[keep_columns]


def _save_dataframe(df: pd.DataFrame, csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    saved_df = df.copy()
    saved_df["image_path"] = saved_df["image_path"].map(
        lambda value: Path(os.path.relpath(Path(value), csv_path.parent)).as_posix()
        if Path(value).is_absolute()
        else Path(value).as_posix()
    )
    saved_df.to_csv(csv_path, index=False)
    return csv_path


def _build_class_weights(train_df: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    counts = train_df["label"].value_counts()
    total = max(int(counts.sum()), 1)
    weights = []
    for class_name in class_names:
        count = max(int(counts.get(class_name, 0)), 1)
        weights.append(total / (len(class_names) * count))
    return torch.tensor(weights, dtype=torch.float32)


def _metric_mode(metric_name: str) -> str:
    return "min" if metric_name == "val_loss" else "max"


def _is_improvement(metric_name: str, current: float, best: float, min_delta: float) -> bool:
    mode = _metric_mode(metric_name)
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels, sample_weights in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        per_sample_loss = criterion(logits, labels)
        loss = (per_sample_loss * sample_weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for images, labels, sample_weights in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        sample_weights = sample_weights.to(device)

        logits = model(images)
        per_sample_loss = criterion(logits, labels)
        loss = (per_sample_loss * sample_weights).mean()
        total_loss += float(loss.item())

        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    f1_macro = f1_score(all_labels, all_preds, average="macro") if all_labels else 0.0
    return total_loss / max(len(loader), 1), accuracy, f1_macro


def _train_single_run(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    seed: int,
    run_name: str,
    device: str,
    class_names: list[str],
) -> dict:
    _set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = _save_dataframe(train_df, output_dir / "train_combined.csv")
    val_csv = _save_dataframe(val_df, output_dir / "val_fold.csv")
    class_weights = _build_class_weights(train_df, class_names)

    train_ds = SeverityCropDataset(
        train_csv,
        train=True,
        img_size=args.img_size,
        model_name=args.model_name,
        class_names=class_names,
    )
    val_ds = SeverityCropDataset(
        val_csv,
        train=False,
        img_size=args.img_size,
        model_name=args.model_name,
        class_names=class_names,
    )
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    use_pretrained_backbone = args.init_checkpoint is None
    model = build_severity_model(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=use_pretrained_backbone,
        xrv_weights=args.xrv_weights,
    ).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(init_checkpoint["state_dict"])
    configure_head_only_finetuning(args.model_name, model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    print(
        f"Fine-tuning mode: head_only "
        f"(trainable_params={sum(param.numel() for param in trainable_params)})"
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction="none")
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=_metric_mode(args.selection_metric),
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    best_selection_value = float("inf") if _metric_mode(args.selection_metric) == "min" else float("-inf")
    best_stop_value = float("inf") if _metric_mode(args.early_stopping_metric) == "min" else float("-inf")
    best_val_loss_observed = float("inf")
    best_val_f1_observed = float("-inf")
    best_epoch = 0
    stop_wait = 0
    stopped_early = False
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, val_f1 = eval_one_epoch(model, val_loader, criterion, device)
        best_val_loss_observed = min(best_val_loss_observed, val_loss)
        best_val_f1_observed = max(best_val_f1_observed, val_f1)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_f1_macro": val_f1,
                "lr": current_lr,
            }
        )
        print(
            f"{run_name} epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f} lr={current_lr:.6f}"
        )

        ckpt = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "model_name": args.model_name,
            "xrv_weights": args.xrv_weights,
            "fine_tuning_mode": "head_only",
            "img_size": args.img_size,
            "class_names": class_names,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_f1_macro": val_f1,
            "selection_metric": args.selection_metric,
            "early_stopping_metric": args.early_stopping_metric,
            "seed": seed,
            "run_name": run_name,
        }
        torch.save(ckpt, output_dir / "last.pt")

        selection_value = val_loss if args.selection_metric == "val_loss" else val_f1
        if _is_improvement(
            args.selection_metric,
            selection_value,
            best_selection_value,
            args.early_stopping_min_delta,
        ):
            best_selection_value = selection_value
            best_epoch = epoch
            torch.save(ckpt, output_dir / "best.pt")

        stop_value = val_loss if args.early_stopping_metric == "val_loss" else val_f1
        if _is_improvement(
            args.early_stopping_metric,
            stop_value,
            best_stop_value,
            args.early_stopping_min_delta,
        ):
            best_stop_value = stop_value
            stop_wait = 0
        else:
            stop_wait += 1

        scheduler_metric = val_loss if args.selection_metric == "val_loss" else val_f1
        scheduler.step(scheduler_metric)

        if stop_wait >= args.early_stopping_patience:
            stopped_early = True
            print(
                f"{run_name} early stopping at epoch={epoch} "
                f"(metric={args.early_stopping_metric}, patience={args.early_stopping_patience})"
            )
            break

    summary = {
        "run_name": run_name,
        "seed": seed,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_checkpoint_metric": args.selection_metric,
        "best_checkpoint_metric_value": best_selection_value,
        "best_val_loss_observed": best_val_loss_observed,
        "best_val_f1_macro_observed": best_val_f1_observed,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_class_counts": train_df["label"].value_counts().to_dict(),
        "val_class_counts": val_df["label"].value_counts().to_dict(),
        "used_pseudo_labels": bool(args.pseudo_csv),
        "fine_tuning_mode": "head_only",
        "trainable_parameter_count": sum(param.numel() for param in trainable_params),
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a severity classifier on lesion crops.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/severity/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/severity/val.csv"))
    parser.add_argument("--pseudo-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/severity/xrv_densenet121"))
    parser.add_argument(
        "--model-name",
        default=DEFAULT_SEVERITY_MODEL_NAME,
        choices=[DEFAULT_SEVERITY_MODEL_NAME],
    )
    parser.add_argument("--xrv-weights", default=DEFAULT_XRV_WEIGHTS)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional local checkpoint used to initialize the model without downloading pretrained weights.",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pseudo-weight-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        default="val_loss",
        choices=["val_loss", "val_f1_macro"],
        help="Metric used to save best.pt.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        default="val_loss",
        choices=["val_loss", "val_f1_macro"],
        help="Metric monitored for early stopping.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Class names in label order. If omitted, derived from sorted unique labels in train CSV.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labeled_df = _load_labeled_csv(args.train_csv)
    val_labeled_df = _load_labeled_csv(args.val_csv)
    pseudo_df = _load_pseudo_csv(args.pseudo_csv, args.pseudo_weight_scale)
    train_df = train_labeled_df.copy()
    if not pseudo_df.empty:
        train_df = pd.concat([train_df, pseudo_df], ignore_index=True)

    if args.class_names:
        class_names = args.class_names
    else:
        class_names = sorted(train_labeled_df["label"].unique().tolist())
    print(f"Class names: {class_names}")

    summary = _train_single_run(
        train_df=train_df,
        val_df=val_labeled_df.copy(),
        args=args,
        output_dir=output_dir,
        seed=args.seed,
        run_name="single_run",
        device=device,
        class_names=class_names,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
