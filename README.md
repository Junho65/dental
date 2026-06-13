# Dental X-ray Detection + Django Service

This project follows your local-first plan:
- download/prepare DENTEX-style data
- convert to object-detection labels (YOLO format)
- train a PyTorch detection model (YOLOv8/YOLO11)
- evaluate and save detection weights
- serve inference with Django

## 1) Environment setup (stability-first for your GPU/CUDA)

```bash
python -m venv .venv
.venv\Scripts\activate
python scripts/install_torch.py
pip install -r requirements.txt
```

## 2) Dataset download and detection preprocessing

```bash
python scripts/download_dataset.py
python scripts/prepare_detection_dataset.py
```

Outputs:
- `data/detection/images/{train,val,test}`
- `data/detection/labels/{train,val,test}`
- `data/detection/dentex_detection.yaml`

## 2B) Add UMFIH Dental Pathology Dataset

UMFIH is a Zenodo YOLO dataset with 14 classes. This repo maps only the classes that safely align
with the existing detector:
- `Carious lesion (4)` -> `caries`
- `Apical periodontitis (7)` -> `periapical_lesion`
- `Impacted tooth (6)` -> `impacted_tooth`

Download and extract:

```bash
python scripts/download_umfih_dataset.py
```

Prepare the 4-class YOLO subset:

```bash
python scripts/prepare_umfih_yolo.py
```

This creates:
- `data/detection_umfih/images/{train,val,test}`
- `data/detection_umfih/labels/{train,val,test}`
- `data/detection_umfih/umfih_detection.yaml`

To merge UMFIH into the existing merged detector dataset:

```bash
python scripts/merge_yolo_detection_datasets.py --base data/detection_merged --extra data/detection_umfih --out data/detection_merged_umfih
```

`merge_yolo_detection_datasets.py` now hardlinks images when possible, so adding UMFIH does not
duplicate the image bytes on the same drive.

Train with the new merged dataset:

```bash
python train_detection.py
```

By default, `python train_detection.py` now prepares a hierarchical detector if needed and trains on:
- `data/detection_hierarchical/hierarchical_detection.yaml`
- classes: `caries_family`, `periapical_lesion`, `impacted_tooth`

## 2C) Preprocessing Script Guide

Use the preprocessing scripts below as the main execution guide. `ARCHITECTURE.md` describes the
system conceptually, while this section is the source of truth for script-level workflow.

### `scripts/prepare_detection_dataset.py`

Purpose:
- Convert labeled DENTEX data into the project's 4-class YOLO detection format.

Input:
- Raw DENTEX files under `data/raw/dentex`

Output:
- `data/detection/images/{train,val,test}`
- `data/detection/labels/{train,val,test}`
- `data/detection/dentex_detection.yaml`

When to run:
- First, when building the base DENTEX detection dataset

Command:

```bash
python scripts/prepare_detection_dataset.py
```

### `scripts/prepare_cariesxrays_yolo.py`

Purpose:
- Convert CariesXrays Pascal VOC annotations into YOLO format aligned to the project's class order.

Input:
- Raw CariesXrays dataset under `data/raw/cariesxrays`

Output:
- `data/detection_cariesxrays/images/{train,val,test}`
- `data/detection_cariesxrays/labels/{train,val,test}`
- `data/detection_cariesxrays/cariesxrays_yolo.yaml`

When to run:
- After preparing DENTEX, when you want to expand the detector with additional caries examples

Typical command:

```bash
python scripts/prepare_cariesxrays_yolo.py --raw data/raw/cariesxrays --out data/detection_cariesxrays --stem-prefix cx_
```

### `scripts/prepare_umfih_yolo.py`

Purpose:
- Remap the UMFIH YOLO dataset into the project's 4-class detection schema.

Input:
- Extracted UMFIH dataset under `data/raw/umfih/extracted`

Output:
- `data/detection_umfih/images/{train,val,test}`
- `data/detection_umfih/labels/{train,val,test}`
- `data/detection_umfih/umfih_detection.yaml`

When to run:
- After downloading and extracting UMFIH, when you want to add pathology coverage beyond the base dataset

Command:

```bash
python scripts/prepare_umfih_yolo.py
```

### `scripts/merge_yolo_detection_datasets.py`

Purpose:
- Merge two YOLO detection datasets that share the same class order into one training dataset.

Input:
- A base YOLO dataset such as `data/detection` or `data/detection_merged`
- An extra YOLO dataset such as `data/detection_cariesxrays` or `data/detection_umfih`

Output:
- A merged YOLO dataset root and YAML file, such as `data/detection_merged` or `data/detection_merged_umfih`

When to run:
- After the individual YOLO datasets have been prepared and you want a single combined detector dataset

Typical commands:

```bash
python scripts/merge_yolo_detection_datasets.py --base data/detection --extra data/detection_cariesxrays --out data/detection_merged
python scripts/merge_yolo_detection_datasets.py --base data/detection_merged --extra data/detection_umfih --out data/detection_merged_umfih
python scripts/merge_yolo_detection_datasets.py --base data/detection_main_4class_no_cyst_no_periodontal --extra data/detection_kaggle_pediatric_selected_4class --out data/detection_main_4class_with_pediatric --yaml-name main_4class_with_pediatric.yaml
```

Notes:
- The script now reads the class order from each dataset YAML and refuses to merge if the orders differ.
- Use `--yaml-name` when the merged output should expose a dataset-specific YAML such as `main_4class_with_pediatric.yaml`.

### `scripts/run_pediatric_service_finetune.py`

Purpose:
- Fine-tune the currently served 4-class detector starting from `artifacts/detection/serve/best.pt`
- Build the pediatric 4-class subset from `data/detection_kaggle_pediatric_selected_6class`
- Merge it into `data/detection_main_4class_no_cyst_no_periodontal`
- Evaluate baseline vs candidate on both the main and pediatric test sets
- Promote the new `best.pt` into `artifacts/detection/serve/best.pt` only when the gating rules pass

Default outputs:
- `data/detection_kaggle_pediatric_selected_4class`
- `data/detection_main_4class_with_pediatric`
- `artifacts/detection/yolov8s_serve_pediatric_ft_v1`
- `reports/yolov8s_serve_pediatric_ft_v1`

Command:

```bash
python scripts/run_pediatric_service_finetune.py
```

Useful dry-run style command when you want dataset prep and evaluation wiring without replacing the service weights:

```bash
python scripts/run_pediatric_service_finetune.py --skip-train --candidate-weights artifacts/detection/serve/best.pt --no-promote
```

### Recommended execution order

1. `python scripts/prepare_detection_dataset.py`
2. `python scripts/prepare_cariesxrays_yolo.py --raw data/raw/cariesxrays --out data/detection_cariesxrays --stem-prefix cx_`
3. `python scripts/prepare_umfih_yolo.py`
4. `python scripts/merge_yolo_detection_datasets.py --base data/detection --extra data/detection_cariesxrays --out data/detection_merged`
5. `python scripts/merge_yolo_detection_datasets.py --base data/detection_merged --extra data/detection_umfih --out data/detection_merged_umfih`
6. `python train_detection.py`

## 3) Train detection model

```bash
python train_detection.py
```

Weights output:
- `artifacts/detection/yolov8s_hierarchical/weights/best.pt`

Current training defaults:
- `epochs=50`
- `imgsz=416`
- `batch=8` on GTX 1660 6GB; if CUDA OOM occurs, retry with `batch=4`, then `batch=2`
- `workers=4`
- default dataset: `data/detection_hierarchical/hierarchical_detection.yaml`
- default model init: `yolov8s.pt` with `pretrained=True` (COCO pretrained fine-tuning)
- `deep_caries` train-image oversampling disabled by default
- TensorBoard scalar logging enabled

Recommended local training command for the current GTX 1660 environment:

```bash
python train_detection.py
```

The root command already defaults to:
- `data=data/detection_hierarchical/hierarchical_detection.yaml`
- `model=yolov8s.pt`
- `imgsz=416`
- `batch=8`
- `workers=4`

Use extra flags only when you want to override the defaults:

```bash
python train_detection.py --name yolov8s_experiment
python train_detection.py --batch 4
```

Recommended model comparison on GTX 1660 6GB:

```bash
python train_detection.py --model yolov8n.pt --name yolov8n_hierarchical_baseline --batch 16 --imgsz 416
python train_detection.py --model yolov8s.pt --name yolov8s_hierarchical --batch 8 --imgsz 416
python train_detection.py --model yolo11s.pt --name yolo11s_hierarchical --batch 8 --imgsz 416
```

Run a 1-epoch smoke test before full training when trying a new model:

```bash
python train_detection.py --model yolov8s.pt --name yolov8s_smoke --epochs 1 --batch 8 --imgsz 416
python train_detection.py --model yolo11s.pt --name yolo11s_smoke --epochs 1 --batch 8 --imgsz 416
```

The decision metric is `metrics/mAP50-95(B)` in each run's `results.csv`. The existing `yolov8n_hierarchical2` reference is about `0.126` at epoch 10 and about `0.173` final, so a larger model should beat both the early and final reference before replacing `artifacts/detection/serve/best.pt`.

Model-size rationale: Ultralytics' official YOLOv8 table lists `YOLOv8s` as larger than `YOLOv8n` in params/FLOPs, while the official YOLO11 table lists `YOLO11s` as a lighter latest-family small candidate than `YOLOv8s`. See the Ultralytics docs for [YOLOv8](https://docs.ultralytics.com/models/yolov8/) and [YOLO11](https://docs.ultralytics.com/models/yolo11/).

If you want to oversample a different class or train on a hierarchical dataset that no longer has
`deep_caries` as a standalone class:

```bash
python train_detection.py --data data/detection_hierarchical/hierarchical_detection.yaml --oversample-class impacted_tooth --deep-caries-balance
python train_detection.py --data data/detection_merged_umfih/merged_detection.yaml --deep-caries-balance
```

## 3B) Hierarchical detection + follow-up classifiers

Current service policy:
- the detector may still be trained on `caries_family`
- the served API normalizes caries-family detections to `caries`
- the old caries-vs-deep-caries refinement stage is retired from the service path because labeled `deep_caries` coverage is too sparse for reliable deployment

The remaining follow-up classifier workflows are:
- metadata-only periapical follow-up routing
- periodontal severity classification on detected `bone_loss` / `furcation_involvement` crops

Build the hierarchical detection dataset from the merged YOLO dataset:

```bash
python scripts/prepare_hierarchical_detection_dataset.py --data data/detection_merged/merged_detection.yaml
python train_detection.py --data data/detection_hierarchical/hierarchical_detection.yaml --no-deep-caries-balance
```

If you want to train a follow-up crop classifier for another lesion family without changing
the served detector taxonomy, first export crop CSVs from the labels you want to classify:

```bash
python scripts/prepare_followup_crop_dataset.py --data data/your_followup_dataset/data.yaml --class-names class_a class_b --out data/followup_custom
```

Then train with the same class order:

```bash
python scripts/train_severity_classifier.py --train-csv data/followup_custom/train.csv --val-csv data/followup_custom/val.csv --output-dir artifacts/severity/periapical_followup --class-names class_a class_b
```

At serving time, an optional metadata-only periapical follow-up model can be loaded with:
- `DENTAL_PERIAPICAL_FOLLOWUP_WEIGHTS`
- `DENTAL_PERIAPICAL_FOLLOWUP_CONF`

For the periodontal severity classifiers, reevaluate an existing checkpoint on the held-out
test split with:

```bash
python scripts/eval_severity_classifier.py --checkpoint artifacts/severity/serve/bone_loss/best.pt --test-csv data/severity_periodontal/bone_loss/test.csv --out-json artifacts/severity/serve/bone_loss/test_metrics.json
python scripts/eval_severity_classifier.py --checkpoint artifacts/severity/serve/furcation_involvement/best.pt --test-csv data/severity_periodontal/furcation_involvement/test.csv --out-json artifacts/severity/serve/furcation_involvement/test_metrics.json
```

To retrain and reevaluate both periodontal severity classifiers in one run:

```bash
python scripts/run_periodontal_severity_retrain.py --epochs 20 --selection-metric val_f1_macro --early-stopping-metric val_f1_macro
```

By default this warm-starts each run from the currently served checkpoint, writes per-lesion
train/test metrics under `artifacts/severity/periodontal_retrain/<timestamp>/`, and compares the
candidate test metrics against the currently served checkpoint. Add `--promote` to replace
`artifacts/severity/serve/{bone_loss,furcation_involvement}/best.pt` only when the candidate
beats the served model on the chosen promotion metric.

TensorBoard while training:

```bash
python train_detection.py --name yolov8s_live
tensorboard --logdir "$env:TEMP\dental_yolo_tb" --reload_interval 2
```

Open `http://localhost:6006/` and select the run folder matching `--name`.

If you want to mirror `results.csv` into a separate live TensorBoard log while a run is in progress:

```bash
python scripts/watch_results_csv_tensorboard.py --run-name yolov8s_live
tensorboard --logdir "$env:TEMP\dental_tensorboard_live" --reload_interval 2
```

If a past run has only `results.csv` and no event files, backfill TensorBoard logs from it:

```bash
python scripts/log_results_csv_to_tensorboard.py runs/detect/artifacts/detection/yolov8s_live/results.csv
```

## 4) Evaluate

```bash
python scripts/eval_detection.py --weights artifacts/detection/yolov8s_hierarchical/weights/best.pt --data data/detection_hierarchical/hierarchical_detection.yaml --imgsz 416
```

Metrics output:
- `reports/detection_metrics.json`

## 5) Run Django inference API

```bash
python manage.py runserver
```

Endpoints:
- `GET /health/`
- `POST /predict/` (multipart form-data with field name: `image`)

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict/ -F "image=@sample.png"
```

Response example:

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "caries",
      "confidence": 0.88,
      "severity_confidence": 0.93,
      "bbox_xyxy": [120.1, 340.2, 220.3, 440.4]
    }
  ]
}
```

## Windows troubleshooting (WinError 1455)

If you see `WinError 1455` while importing `torch`/`ultralytics`, Windows virtual memory (page file) is too small.

- Increase page file size in System Advanced Settings.
- Reboot Windows.
- Re-run:

```bash
python train_detection.py
```
