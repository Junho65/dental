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

## 3B) Hierarchical detection + severity classifier

Use this workflow when you want:
- detection to learn `caries_family` instead of forcing `caries` vs `deep_caries` directly
- a second-stage classifier to decide `caries` vs `deep_caries`
- optional pseudo-labeling on CariesXrays crops

At serving time, the same severity classifier can also be applied to flat detector outputs
(`caries` / `deep_caries`) as a refinement or override stage.

Build the hierarchical detection dataset from the merged YOLO dataset:

```bash
python scripts/prepare_hierarchical_detection_dataset.py --data data/detection_merged/merged_detection.yaml
python train_detection.py --data data/detection_hierarchical/hierarchical_detection.yaml --no-deep-caries-balance
```

Prepare labeled severity crops from DENTEX and unlabeled lesion crops from CariesXrays:

```bash
python scripts/prepare_severity_dataset.py
```

Train the severity classifier on labeled DENTEX crops:

```bash
python scripts/train_severity_classifier.py
```

The default severity backbone is `TorchXRayVision DenseNet121` with
`densenet121-res224-all` chest X-ray pretrained weights.

Generate high-confidence pseudo-labels from CariesXrays lesion crops, then retrain with them:

```bash
python scripts/pseudolabel_severity.py --weights artifacts/severity/xrv_densenet121/best.pt --output-csv artifacts/severity/pseudo/train.csv
python scripts/train_severity_classifier.py --pseudo-csv artifacts/severity/pseudo/train.csv --output-dir artifacts/severity/xrv_densenet121_pseudo
```

For Django serving, place the selected severity checkpoint at:
- `artifacts/severity/serve/best.pt`

Or set:
- `DENTAL_SEVERITY_WEIGHTS`
- `DENTAL_SEVERITY_CONF`

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
