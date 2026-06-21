"""Orchestrate periodontal severity retraining and evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


LESION_CONFIGS = {
    "bone_loss": {
        "class_names": ["mild", "medium", "severe"],
        "data_dir": Path("data/severity_periodontal/bone_loss"),
        "service_weights": Path("artifacts/severity/serve/bone_loss/best.pt"),
    },
    "furcation_involvement": {
        "class_names": ["mild", "severe"],
        "data_dir": Path("data/severity_periodontal/furcation_involvement"),
        "service_weights": Path("artifacts/severity/serve/furcation_involvement/best.pt"),
    },
}


def _run_command(command: list[str], repo_root: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo_root, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _promote_weights(service_weights: Path, candidate_weights: Path, lesion_name: str) -> dict:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = service_weights.with_name(f"best.before_{lesion_name}_retrain_{timestamp}.pt")
    service_weights.parent.mkdir(parents=True, exist_ok=True)
    if service_weights.is_file():
        shutil.copy2(service_weights, backup_path)
    else:
        backup_path = None
    shutil.copy2(candidate_weights, service_weights)
    return {
        "backup": str(backup_path) if backup_path is not None else None,
        "promoted": str(service_weights),
    }


def _build_train_command(
    repo_root: Path,
    lesion_name: str,
    lesion_cfg: dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    data_dir = repo_root / lesion_cfg["data_dir"]
    command = [
        sys.executable,
        "scripts/training/train_severity_classifier.py",
        "--train-csv",
        str(data_dir / "train.csv"),
        "--val-csv",
        str(data_dir / "val.csv"),
        "--output-dir",
        str(output_dir),
        "--img-size",
        str(args.img_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--selection-metric",
        args.selection_metric,
        "--early-stopping-metric",
        args.early_stopping_metric,
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--early-stopping-min-delta",
        str(args.early_stopping_min_delta),
        "--lr-patience",
        str(args.lr_patience),
        "--lr-factor",
        str(args.lr_factor),
        "--device",
        args.device,
        "--model-name",
        args.model_name,
        "--xrv-weights",
        args.xrv_weights,
        "--class-names",
        *lesion_cfg["class_names"],
    ]
    if args.init_mode == "serve":
        service_weights = repo_root / lesion_cfg["service_weights"]
        if not service_weights.is_file():
            raise FileNotFoundError(
                f"Serve weights not found for {lesion_name}: {service_weights}"
            )
        command.extend(["--init-checkpoint", str(service_weights)])
    return command


def _build_eval_command(
    checkpoint: Path,
    test_csv: Path,
    out_json: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/evaluation/eval_severity_classifier.py",
        "--checkpoint",
        str(checkpoint),
        "--test-csv",
        str(test_csv),
        "--out-json",
        str(out_json),
        "--batch-size",
        str(args.eval_batch_size),
        "--device",
        args.device,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain and reevaluate the periodontal severity classifiers."
    )
    parser.add_argument(
        "--lesions",
        nargs="+",
        choices=sorted(LESION_CONFIGS),
        default=sorted(LESION_CONFIGS),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(f"artifacts/severity/periodontal_retrain/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    parser.add_argument("--init-mode", choices=["serve", "pretrained"], default="serve")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-metric", choices=["val_loss", "val_f1_macro"], default="val_f1_macro")
    parser.add_argument("--early-stopping-metric", choices=["val_loss", "val_f1_macro"], default="val_f1_macro")
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--model-name", default="xrv_densenet121", choices=["xrv_densenet121"])
    parser.add_argument("--xrv-weights", default="densenet121-res224-all")
    parser.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promotion-metric", choices=["accuracy", "f1_macro", "f1_weighted"], default="f1_macro")
    parser.add_argument("--promotion-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    run_root = (repo_root / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    overall_summary = {
        "run_root": str(run_root),
        "init_mode": args.init_mode,
        "lesions": {},
    }

    for lesion_name in args.lesions:
        lesion_cfg = LESION_CONFIGS[lesion_name]
        lesion_run_dir = run_root / lesion_name
        lesion_run_dir.mkdir(parents=True, exist_ok=True)

        data_dir = (repo_root / lesion_cfg["data_dir"]).resolve()
        test_csv = data_dir / "test.csv"
        candidate_weights = lesion_run_dir / "best.pt"
        candidate_metrics_path = lesion_run_dir / "metrics.json"
        candidate_test_metrics_path = lesion_run_dir / "test_metrics.json"
        served_test_metrics_path = lesion_run_dir / "served_test_metrics.json"

        if not args.skip_train:
            train_command = _build_train_command(
                repo_root=repo_root,
                lesion_name=lesion_name,
                lesion_cfg=lesion_cfg,
                output_dir=lesion_run_dir,
                args=args,
            )
            _run_command(train_command, repo_root)

        if not candidate_weights.is_file():
            raise FileNotFoundError(
                f"Candidate checkpoint missing for {lesion_name}: {candidate_weights}"
            )

        _run_command(
            _build_eval_command(candidate_weights, test_csv, candidate_test_metrics_path, args),
            repo_root,
        )
        candidate_train_metrics = _load_json(candidate_metrics_path) if candidate_metrics_path.is_file() else None
        candidate_test_metrics = _load_json(candidate_test_metrics_path)

        service_weights = (repo_root / lesion_cfg["service_weights"]).resolve()
        served_test_metrics = None
        if service_weights.is_file():
            _run_command(
                _build_eval_command(service_weights, test_csv, served_test_metrics_path, args),
                repo_root,
            )
            served_test_metrics = _load_json(served_test_metrics_path)

        comparison = {"metric": args.promotion_metric, "passed": served_test_metrics is None}
        if served_test_metrics is not None:
            candidate_value = float(candidate_test_metrics[args.promotion_metric])
            served_value = float(served_test_metrics[args.promotion_metric])
            comparison.update(
                {
                    "candidate": candidate_value,
                    "served": served_value,
                    "delta": candidate_value - served_value,
                }
            )
            comparison["passed"] = candidate_value >= (served_value + args.promotion_min_delta)

        promotion = {"attempted": False, "completed": False}
        if args.promote and comparison["passed"]:
            promotion["attempted"] = True
            promotion.update(_promote_weights(service_weights, candidate_weights, lesion_name))
            promotion["completed"] = True

        lesion_summary = {
            "data_dir": str(data_dir),
            "class_names": lesion_cfg["class_names"],
            "candidate_weights": str(candidate_weights),
            "service_weights": str(service_weights),
            "candidate_train_metrics": candidate_train_metrics,
            "candidate_test_metrics": candidate_test_metrics,
            "served_test_metrics": served_test_metrics,
            "comparison": comparison,
            "promotion": promotion,
        }
        overall_summary["lesions"][lesion_name] = lesion_summary

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(overall_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall_summary, indent=2))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
