import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_NAMES = ["caries", "deep_caries", "periapical_lesion", "impacted_tooth"]


def parse_test_label(label_text: str) -> str:
    label_text = label_text.lower()
    if "çürük" in label_text or "caries" in label_text:
        return "caries"
    if "kanal" in label_text or "periapical" in label_text:
        return "periapical_lesion"
    if "gömülü" in label_text or "impacted" in label_text:
        return "impacted_tooth"
    return ""


def parse_validation_coco(json_path: Path, image_root: Path) -> List[Dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    image_by_id = {i["id"]: i["file_name"] for i in payload["images"]}
    disease_map = {c["id"]: c["name"] for c in payload["categories_3"]}
    labels_by_image = {}
    for ann in payload["annotations"]:
        img_name = image_by_id[ann["image_id"]]
        disease_name = disease_map.get(ann.get("category_id_3", -1), "")
        key = disease_name.lower()
        mapped = ""
        if "deep" in key:
            mapped = "deep_caries"
        elif "caries" in key:
            mapped = "caries"
        elif "periapical" in key:
            mapped = "periapical_lesion"
        elif "impact" in key:
            mapped = "impacted_tooth"
        if not mapped:
            continue
        labels_by_image.setdefault(img_name, set()).add(mapped)

    rows = []
    for img_name, label_set in labels_by_image.items():
        img_path = image_root / img_name
        if img_path.exists():
            rows.append({"image_path": str(img_path.resolve()), "labels": "|".join(sorted(label_set))})
    return rows


def parse_test_json_labels(test_label_dir: Path, test_img_dir: Path) -> List[Dict]:
    rows = []
    for jf in sorted(test_label_dir.glob("*.json")):
        payload = json.loads(jf.read_text(encoding="utf-8", errors="ignore"))
        image_name = payload.get("imagePath", f"{jf.stem}.png")
        labels = set()
        for s in payload.get("shapes", []):
            mapped = parse_test_label(str(s.get("label", "")))
            if mapped:
                labels.add(mapped)
        if not labels:
            continue
        img_path = test_img_dir / image_name
        if img_path.exists():
            rows.append({"image_path": str(img_path.resolve()), "labels": "|".join(sorted(labels))})
    return rows


def main():
    raw_root = Path("data/raw/dentex")
    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)

    val_json = raw_root / "DENTEX" / "validation_triple.json"
    val_img_dir = raw_root / "DENTEX" / "validation_data" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    test_label_dir = raw_root / "DENTEX" / "test_data" / "disease" / "label"
    test_img_dir = raw_root / "DENTEX" / "test_data" / "disease" / "input"

    rows = []
    if val_json.exists() and val_img_dir.exists():
        rows.extend(parse_validation_coco(val_json, val_img_dir))
    if test_label_dir.exists() and test_img_dir.exists():
        rows.extend(parse_test_json_labels(test_label_dir, test_img_dir))

    if not rows:
        raise RuntimeError(
            "No labeled rows created. Check DENTEX folder layout and JSON schema."
        )

    df = pd.DataFrame(rows).drop_duplicates("image_path")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=2 / 3, random_state=42)

    train_df.to_csv(processed / "train.csv", index=False)
    val_df.to_csv(processed / "val.csv", index=False)
    test_df.to_csv(processed / "test.csv", index=False)

    stats = {}
    for name in CLASS_NAMES:
        stats[name] = int(df["labels"].str.contains(name).sum())
    (processed / "class_distribution.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print("Saved CSV splits and class_distribution.json")


if __name__ == "__main__":
    main()
