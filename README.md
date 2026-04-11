# Dental X-ray Detection + Django Service

This project follows your local-first plan:
- download/prepare DENTEX-style data
- convert to object-detection labels (YOLO format)
- train a PyTorch detection model (YOLOv8)
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
python scripts/train_detection.py --data data/detection_merged_umfih/merged_detection.yaml
```

## 3) Train detection model

```bash
python scripts/train_detection.py
```

Weights output:
- `artifacts/detection/yolov8n_dentex/weights/best.pt`

Current training defaults:
- `epochs=30`
- `imgsz=512`
- `batch=2`
- `workers=0`
- `deep_caries` train-image oversampling enabled
- TensorBoard scalar logging enabled

If you want to oversample a different class or train on a hierarchical dataset that no longer has
`deep_caries` as a standalone class:

```bash
python scripts/train_detection.py --data data/detection_hierarchical/hierarchical_detection.yaml --no-deep-caries-balance
python scripts/train_detection.py --oversample-class impacted_tooth
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
python scripts/train_detection.py --data data/detection_hierarchical/hierarchical_detection.yaml --no-deep-caries-balance
```

Prepare labeled severity crops from DENTEX and unlabeled lesion crops from CariesXrays:

```bash
python scripts/prepare_severity_dataset.py
```

Train the severity classifier on labeled DENTEX crops:

```bash
python scripts/train_severity_classifier.py
```

Generate high-confidence pseudo-labels from CariesXrays lesion crops, then retrain with them:

```bash
python scripts/pseudolabel_severity.py --weights artifacts/severity/efficientnet_b0/best.pt --output-csv artifacts/severity/pseudo/train.csv
python scripts/train_severity_classifier.py --pseudo-csv artifacts/severity/pseudo/train.csv --output-dir artifacts/severity/efficientnet_b0_pseudo
```

For Django serving, place the selected severity checkpoint at:
- `artifacts/severity/serve/best.pt`

Or set:
- `DENTAL_SEVERITY_WEIGHTS`
- `DENTAL_SEVERITY_CONF`

TensorBoard while training:

```bash
python scripts/train_detection.py --name yolov8n_live
tensorboard --logdir "$env:TEMP\dental_yolo_tb" --reload_interval 2
```

Open `http://localhost:6006/` and select the run folder matching `--name`.

If you want to mirror `results.csv` into a separate live TensorBoard log while a run is in progress:

```bash
python scripts/watch_results_csv_tensorboard.py --run-name yolov8n_live
tensorboard --logdir "$env:TEMP\dental_tensorboard_live" --reload_interval 2
```

If a past run has only `results.csv` and no event files, backfill TensorBoard logs from it:

```bash
python scripts/log_results_csv_to_tensorboard.py runs/detect/artifacts/detection/yolov8n_live/results.csv
```

## 4) Evaluate

```bash
python scripts/eval_detection.py --weights artifacts/detection/yolov8n_dentex/weights/best.pt
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
python scripts/train_detection.py
```
