from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET = "lokisilvres/dental-disease-panoramic-detection-dataset"
TRAIN_ARG_PATTERNS = {
    "imgsz": re.compile(r"\bimgsz\s*=\s*([0-9]+)", re.IGNORECASE),
    "epochs": re.compile(r"\bepochs\s*=\s*([0-9]+)", re.IGNORECASE),
    "batch": re.compile(r"\bbatch\s*=\s*([0-9-]+)", re.IGNORECASE),
    "optimizer": re.compile(r"\boptimizer\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE),
    "lr0": re.compile(r"\blr0\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
    "lrf": re.compile(r"\blrf\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
    "fliplr": re.compile(r"\bfliplr\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
    "mosaic": re.compile(r"\bmosaic\s*=\s*([0-9.eE+-]+)", re.IGNORECASE),
    "close_mosaic": re.compile(r"\bclose_mosaic\s*=\s*([0-9]+)", re.IGNORECASE),
}
KEYWORD_PATTERNS = {
    "uses_yolo": re.compile(r"\bYOLO\s*\(|ultralytics", re.IGNORECASE),
    "mentions_detection": re.compile(r"\bdetect|detection|bbox|bounding", re.IGNORECASE),
    "mentions_segmentation": re.compile(r"\bsegment|segmentation|mask|polygon", re.IGNORECASE),
    "mentions_clahe": re.compile(r"\bCLAHE\b|equalizeHist|contrast", re.IGNORECASE),
    "mentions_flip": re.compile(r"\bfliplr|flip", re.IGNORECASE),
}


def _token_exists() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) or (Path.home() / ".kaggle" / "kaggle.json").is_file()


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)


def _require_kaggle(summary: dict) -> bool:
    if shutil.which("kaggle") is None:
        summary["status"] = "skipped"
        summary["reason"] = "Kaggle CLI was not found on PATH."
        return False
    if not _token_exists():
        summary["status"] = "skipped"
        summary["reason"] = "Kaggle API credentials were not found."
        return False
    return True


def _parse_kernel_refs(output: str, limit: int) -> list[str]:
    refs: list[str] = []
    for line in output.splitlines():
        match = re.search(r"([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)", line)
        if match and match.group(1) not in refs:
            refs.append(match.group(1))
        if len(refs) >= limit:
            break
    return refs


def _extract_notebook_text(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")
    chunks: list[str] = []
    for cell in payload.get("cells", []):
        source = cell.get("source", "")
        if isinstance(source, list):
            chunks.extend(str(item) for item in source)
        else:
            chunks.append(str(source))
    return "\n".join(chunks)


def _summarize_text(text: str) -> dict:
    train_args = {}
    for name, pattern in TRAIN_ARG_PATTERNS.items():
        values = sorted(set(match.group(1) for match in pattern.finditer(text)))
        if values:
            train_args[name] = values
    return {
        "train_args": train_args,
        "signals": {name: bool(pattern.search(text)) for name, pattern in KEYWORD_PATTERNS.items()},
        "model_mentions": sorted(set(re.findall(r"\b(?:yolo(?:v)?8|yolo11|yolov5)[a-z0-9._-]*", text, flags=re.IGNORECASE)))[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull and summarize Kaggle notebooks linked to the dental dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = (args.out or Path("reports/kaggle_notebook_analysis") / timestamp).resolve()
    notebooks_dir = out_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "dataset": args.dataset,
        "output_dir": str(out_root),
        "notebooks": [],
    }
    if not _require_kaggle(summary):
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    list_result = _run(["kaggle", "kernels", "list", "--dataset", args.dataset])
    summary["kernel_list_returncode"] = list_result.returncode
    summary["kernel_list_stderr"] = list_result.stderr.strip()
    refs = _parse_kernel_refs(list_result.stdout, args.limit) if list_result.returncode == 0 else []
    summary["kernel_refs"] = refs

    for ref in refs:
        safe = ref.replace("/", "__")
        target = notebooks_dir / safe
        target.mkdir(parents=True, exist_ok=True)
        pull_result = _run(["kaggle", "kernels", "pull", ref, "-p", str(target)], cwd=notebooks_dir)
        record = {
            "ref": ref,
            "pull_returncode": pull_result.returncode,
            "pull_stderr": pull_result.stderr.strip(),
            "files": [],
        }
        for path in sorted(target.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".ipynb", ".py"}:
                continue
            text = _extract_notebook_text(path)
            item = {"path": str(path), **_summarize_text(text)}
            record["files"].append(item)
        summary["notebooks"].append(record)

    summary["status"] = "ok"
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
