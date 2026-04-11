from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.severity.inference import SeverityPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-confidence severity pseudo-labels.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/severity_unlabeled/train.csv"),
        help="CSV listing unlabeled crop paths.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    predictor = SeverityPredictor(args.weights, device=device)
    df = pd.read_csv(args.input_csv)

    rows = []
    for row in tqdm(df.to_dict("records"), desc="pseudo", leave=False):
        image_path = Path(row["image_path"])
        prediction = predictor.predict_pil(Image.open(image_path).convert("RGB"))
        if prediction["confidence"] < args.threshold:
            continue
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "label": prediction["class_name"],
                "confidence": prediction["confidence"],
                "source": row.get("source", "pseudo"),
            }
        )

    out_df = pd.DataFrame(rows, columns=["image_path", "label", "confidence", "source"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "input_rows": len(df),
                "pseudo_labels_written": len(out_df),
                "threshold": args.threshold,
                "output_csv": str(args.output_csv.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
