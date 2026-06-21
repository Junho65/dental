"""Orchestrate pediatric detector fine-tuning and promotion checks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO


EXPECTED_CLASS_ORDER = [
    "caries_family",
    "periapical_lesion",
    "impacted_tooth",
    "retained_root",
]
CRITICAL_RECALL_CLASSES = ("impacted_tooth", "retained_root")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [raw_names[idx] for idx in sorted(raw_names)]
    return list(raw_names)


def _run(command: list[str], cwd: Path) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), check=True)


def _resolve_repo_path(repo_root: Path, raw_path: Path) -> Path:
    if raw_path.is_absolute():
        return raw_path.resolve()
    direct = (repo_root / raw_path).resolve()
    if direct.exists():
        return direct
    if raw_path.parts and raw_path.parts[0].lower() == repo_root.name.lower():
        stripped = Path(*raw_path.parts[1:])
        return (repo_root / stripped).resolve()
    return direct


def _ensure_output_missing(root: Path, expected_yaml: Path) -> bool:
    return root.exists() and expected_yaml.exists()


def _count_split_files(root: Path, split: str) -> dict[str, int]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = 0
    labels = 0
    if image_dir.is_dir():
        images = sum(1 for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if label_dir.is_dir():
        labels = sum(1 for path in label_dir.glob("*.txt") if path.is_file())
    return {"images": images, "labels": labels}


def _audit_dataset(root: Path, yaml_path: Path) -> dict:
    config = _load_yaml(yaml_path)
    names = _normalize_names(config.get("names", []))
    split_counts = {split: _count_split_files(root, split) for split in ("train", "val", "test")}
    for split, counts in split_counts.items():
        if counts["images"] != counts["labels"]:
            raise SystemExit(
                f"Dataset audit failed for {root} split={split}: "
                f"images={counts['images']} labels={counts['labels']}"
            )
    if names != EXPECTED_CLASS_ORDER:
        raise SystemExit(f"Dataset class order mismatch for {yaml_path}: {names}")
    return {
        "root": str(root),
        "yaml": str(yaml_path),
        "names": names,
        "splits": split_counts,
    }


def _read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _class_metric(report: dict, class_name: str, key: str) -> float | None:
    for row in report.get("per_class", []):
        if row.get("class_name") == class_name:
            return float(row[key])
    return None


def _compare_reports(
    baseline_main: dict,
    candidate_main: dict,
    baseline_pediatric: dict,
    candidate_pediatric: dict,
    max_main_map_drop: float,
    max_critical_recall_drop: float,
) -> dict:
    passed = True
    reasons: list[str] = []

    main_baseline_map = float(baseline_main["overall"]["map50_95"])
    main_candidate_map = float(candidate_main["overall"]["map50_95"])
    pediatric_baseline_map = float(baseline_pediatric["overall"]["map50_95"])
    pediatric_candidate_map = float(candidate_pediatric["overall"]["map50_95"])
    main_map_drop = main_baseline_map - main_candidate_map
    pediatric_map_delta = pediatric_candidate_map - pediatric_baseline_map

    if main_map_drop > max_main_map_drop:
        passed = False
        reasons.append(
            f"main test mAP50-95 drop {main_map_drop:.6f} exceeds allowed {max_main_map_drop:.6f}"
        )
    if pediatric_candidate_map + 1e-12 < pediatric_baseline_map:
        passed = False
        reasons.append(
            f"pediatric test mAP50-95 decreased from {pediatric_baseline_map:.6f} to {pediatric_candidate_map:.6f}"
        )

    critical_recalls: dict[str, dict[str, float | None]] = {}
    for class_name in CRITICAL_RECALL_CLASSES:
        baseline_recall = _class_metric(baseline_main, class_name, "recall")
        candidate_recall = _class_metric(candidate_main, class_name, "recall")
        critical_recalls[class_name] = {
            "baseline_recall": baseline_recall,
            "candidate_recall": candidate_recall,
            "drop": None if baseline_recall is None or candidate_recall is None else baseline_recall - candidate_recall,
        }
        if baseline_recall is None or candidate_recall is None:
            passed = False
            reasons.append(f"missing recall metric for critical class '{class_name}' on main test")
            continue
        if baseline_recall - candidate_recall > max_critical_recall_drop:
            passed = False
            reasons.append(
                f"{class_name} recall drop {baseline_recall - candidate_recall:.6f} "
                f"exceeds allowed {max_critical_recall_drop:.6f}"
            )

    return {
        "passed": passed,
        "thresholds": {
            "max_main_map_drop": max_main_map_drop,
            "max_critical_recall_drop": max_critical_recall_drop,
        },
        "summary": {
            "main_baseline_map50_95": main_baseline_map,
            "main_candidate_map50_95": main_candidate_map,
            "main_map50_95_drop": main_map_drop,
            "pediatric_baseline_map50_95": pediatric_baseline_map,
            "pediatric_candidate_map50_95": pediatric_candidate_map,
            "pediatric_map50_95_delta": pediatric_map_delta,
        },
        "critical_recalls": critical_recalls,
        "reasons": reasons,
    }


def _verify_weights_class_order(weights_path: Path) -> list[str]:
    model = YOLO(str(weights_path))
    class_order = _normalize_names(getattr(model, "names", []))
    if class_order != EXPECTED_CLASS_ORDER:
        raise SystemExit(
            f"Weight class order mismatch for {weights_path}: {class_order} != {EXPECTED_CLASS_ORDER}"
        )
    return class_order


def _verify_predictor_load(repo_root: Path, weights_path: Path) -> list[str]:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from django_app.classifier.inference import Predictor

    predictor = Predictor(checkpoint_path=str(weights_path))
    if predictor.model_class_names != EXPECTED_CLASS_ORDER:
        raise SystemExit(
            "Predictor loaded unexpected class order: "
            f"{predictor.model_class_names} != {EXPECTED_CLASS_ORDER}"
        )
    return predictor.model_class_names


def _promote_weights(service_weights: Path, candidate_weights: Path) -> dict:
    today = datetime.now().strftime("%Y%m%d")
    backup_path = service_weights.with_name(f"best.before_pediatric_ft_{today}.pt")
    shutil.copy2(service_weights, backup_path)
    shutil.copy2(candidate_weights, service_weights)
    return {
        "backup": str(backup_path),
        "promoted": str(service_weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the served 4-class detector with pediatric data.")
    parser.add_argument(
        "--pediatric-source",
        type=Path,
        default=Path("data/detection_kaggle_pediatric_selected_6class/pediatric_selected_6class.yaml"),
    )
    parser.add_argument(
        "--pediatric-out",
        type=Path,
        default=Path("data/detection_kaggle_pediatric_selected_4class"),
    )
    parser.add_argument("--pediatric-yaml-name", default="pediatric_selected_4class.yaml")
    parser.add_argument(
        "--main-root",
        type=Path,
        default=Path("data/detection_main_4class_no_cyst_no_periodontal"),
    )
    parser.add_argument(
        "--main-yaml",
        type=Path,
        default=Path("data/detection_main_4class_no_cyst_no_periodontal/main_4class_no_cyst_no_periodontal.yaml"),
    )
    parser.add_argument(
        "--merged-out",
        type=Path,
        default=Path("data/detection_main_4class_with_pediatric"),
    )
    parser.add_argument("--merged-yaml-name", default="main_4class_with_pediatric.yaml")
    parser.add_argument(
        "--service-weights",
        type=Path,
        default=Path("artifacts/detection/serve/dental_4class_detection_best.pt"),
    )
    parser.add_argument("--project", type=Path, default=Path("artifacts/detection"))
    parser.add_argument("--run-name", default="yolov8s_serve_pediatric_ft_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--max-main-map-drop", type=float, default=0.01)
    parser.add_argument("--max-critical-recall-drop", type=float, default=0.05)
    parser.add_argument("--candidate-weights", type=Path, default=None)
    parser.add_argument("--skip-smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promote", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_exe = sys.executable

    pediatric_source = _resolve_repo_path(repo_root, args.pediatric_source)
    pediatric_out = _resolve_repo_path(repo_root, args.pediatric_out)
    pediatric_yaml = pediatric_out / args.pediatric_yaml_name
    main_root = _resolve_repo_path(repo_root, args.main_root)
    main_yaml = _resolve_repo_path(repo_root, args.main_yaml)
    merged_out = _resolve_repo_path(repo_root, args.merged_out)
    merged_yaml = merged_out / args.merged_yaml_name
    service_weights = _resolve_repo_path(repo_root, args.service_weights)
    project_root = _resolve_repo_path(repo_root, args.project)
    reports_root = repo_root / "reports" / args.run_name
    reports_root.mkdir(parents=True, exist_ok=True)

    if not pediatric_source.exists():
        raise SystemExit(f"Missing pediatric source YAML: {pediatric_source}")
    if not main_yaml.exists():
        raise SystemExit(f"Missing main dataset YAML: {main_yaml}")
    if not service_weights.exists():
        raise SystemExit(f"Missing service weights: {service_weights}")

    if not _ensure_output_missing(pediatric_out, pediatric_yaml):
        _run(
            [
                python_exe,
                "scripts/data/filter_yolo_classes.py",
                "--data",
                str(pediatric_source),
                "--out",
                str(pediatric_out),
                "--keep-names",
                ",".join(EXPECTED_CLASS_ORDER),
                "--yaml-name",
                args.pediatric_yaml_name,
            ],
            cwd=repo_root,
        )

    if not _ensure_output_missing(merged_out, merged_yaml):
        _run(
            [
                python_exe,
                "scripts/data/merge_yolo_detection_datasets.py",
                "--base",
                str(main_root),
                "--extra",
                str(pediatric_out),
                "--out",
                str(merged_out),
                "--yaml-name",
                args.merged_yaml_name,
            ],
            cwd=repo_root,
        )

    dataset_audit = {
        "pediatric_4class": _audit_dataset(pediatric_out, pediatric_yaml),
        "main_with_pediatric": _audit_dataset(merged_out, merged_yaml),
    }

    _run(
        [
            python_exe,
            "scripts/data/build_yolo_label_cache.py",
            "--data",
            str(main_yaml),
            "--splits",
            "test",
            "--refresh",
        ],
        cwd=repo_root,
    )
    for data_yaml in (pediatric_yaml, merged_yaml):
        _run(
            [
                python_exe,
                "scripts/data/build_yolo_label_cache.py",
                "--data",
                str(data_yaml),
                "--refresh",
            ],
            cwd=repo_root,
        )

    (reports_root / "dataset_audit.json").write_text(
        json.dumps(dataset_audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    baseline_main_report = reports_root / "baseline_main4.json"
    baseline_pediatric_report = reports_root / "baseline_pediatric4.json"
    _run(
        [
            python_exe,
            "scripts/evaluation/eval_detection.py",
            "--weights",
            str(service_weights),
            "--data",
            str(main_yaml),
            "--imgsz",
            str(args.imgsz),
            "--split",
            "test",
            "--workers",
            str(args.eval_workers),
            "--out",
            str(baseline_main_report),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python_exe,
            "scripts/evaluation/eval_detection.py",
            "--weights",
            str(service_weights),
            "--data",
            str(pediatric_yaml),
            "--imgsz",
            str(args.imgsz),
            "--split",
            "test",
            "--workers",
            str(args.eval_workers),
            "--out",
            str(baseline_pediatric_report),
        ],
        cwd=repo_root,
    )

    candidate_weights = args.candidate_weights
    if candidate_weights is None:
        candidate_weights = project_root / args.run_name / "weights" / "best.pt"
    else:
        candidate_weights = _resolve_repo_path(repo_root, candidate_weights)

    if not args.skip_train:
        if not args.skip_smoke:
            _run(
                [
                    python_exe,
                    "scripts/training/train_detection.py",
                    "--data",
                    str(merged_yaml),
                    "--model",
                    str(service_weights),
                    "--epochs",
                    str(args.smoke_epochs),
                    "--patience",
                    str(args.patience),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--project",
                    str(project_root),
                    "--name",
                    f"{args.run_name}_smoke",
                    "--workers",
                    str(args.workers),
                    "--device",
                    str(args.device),
                    "--tensorboard",
                ],
                cwd=repo_root,
            )

        _run(
            [
                python_exe,
                "scripts/training/train_detection.py",
                "--data",
                str(merged_yaml),
                "--model",
                str(service_weights),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--imgsz",
                str(args.imgsz),
                "--batch",
                str(args.batch),
                "--project",
                str(project_root),
                "--name",
                args.run_name,
                "--workers",
                str(args.workers),
                "--device",
                str(args.device),
                "--tensorboard",
            ],
            cwd=repo_root,
        )

    if not candidate_weights.exists():
        raise SystemExit(f"Missing candidate weights after training/evaluation flow: {candidate_weights}")

    class_order = _verify_weights_class_order(candidate_weights)
    candidate_main_report = reports_root / "candidate_main4.json"
    candidate_pediatric_report = reports_root / "candidate_pediatric4.json"
    _run(
        [
            python_exe,
            "scripts/evaluation/eval_detection.py",
            "--weights",
            str(candidate_weights),
            "--data",
            str(main_yaml),
            "--imgsz",
            str(args.imgsz),
            "--split",
            "test",
            "--workers",
            str(args.eval_workers),
            "--out",
            str(candidate_main_report),
        ],
        cwd=repo_root,
    )
    _run(
        [
            python_exe,
            "scripts/evaluation/eval_detection.py",
            "--weights",
            str(candidate_weights),
            "--data",
            str(pediatric_yaml),
            "--imgsz",
            str(args.imgsz),
            "--split",
            "test",
            "--workers",
            str(args.eval_workers),
            "--out",
            str(candidate_pediatric_report),
        ],
        cwd=repo_root,
    )

    comparison = _compare_reports(
        baseline_main=_read_report(baseline_main_report),
        candidate_main=_read_report(candidate_main_report),
        baseline_pediatric=_read_report(baseline_pediatric_report),
        candidate_pediatric=_read_report(candidate_pediatric_report),
        max_main_map_drop=args.max_main_map_drop,
        max_critical_recall_drop=args.max_critical_recall_drop,
    )

    promotion = {"attempted": False, "completed": False}
    promoted_predictor_order: list[str] | None = None
    if comparison["passed"] and args.promote:
        promotion["attempted"] = True
        promotion.update(_promote_weights(service_weights, candidate_weights))
        promotion["completed"] = True
        promoted_predictor_order = _verify_predictor_load(repo_root, service_weights)

    summary = {
        "service_weights": str(service_weights),
        "candidate_weights": str(candidate_weights),
        "candidate_class_order": class_order,
        "dataset_audit": dataset_audit,
        "comparison": comparison,
        "promotion": promotion,
        "predictor_class_order_after_promotion": promoted_predictor_order,
    }
    summary_path = reports_root / "promotion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
