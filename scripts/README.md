# Script guide

Run all commands from the project root. The default evaluation path is intentionally short; the
remaining scripts support dataset extensions and model-specific experiments.

## Evaluator quick path

```powershell
python scripts/data/download_dentex_dataset.py
python scripts/data/prepare_detection_dataset.py
python scripts/training/train_detection.py --data data/detection/dentex_detection.yaml
python scripts/evaluation/eval_detection.py --weights artifacts/detection/yolov8s_hierarchical/weights/best.pt --data data/detection/dentex_detection.yaml
python manage.py runserver
```

Downloading the dataset is unnecessary when `data/raw/dentex` is already populated. Training is
also unnecessary when the served checkpoint is included under `artifacts/detection/serve`.

## Directory layout

| Directory | Contents |
|---|---|
| `data/` | Dataset acquisition, conversion, merging, filtering, and validation |
| `training/` | Detector/classifier training and retraining orchestration |
| `evaluation/` | Standalone checkpoint and confidence evaluation |

## Main pipeline

| Stage | Script | Purpose |
|---|---|---|
| Acquire | `data/download_dentex_dataset.py` | Download and extract the DENTEX files used by preprocessing |
| Acquire | `data/download_umfih_dataset.py` | Download and extract the optional UMFIH extension dataset |
| Prepare | `data/prepare_detection_dataset.py` | Convert DENTEX annotations to four-class YOLO data |
| Prepare | `data/prepare_cariesxrays_yolo.py` | Convert optional CariesXrays VOC annotations |
| Prepare | `data/prepare_umfih_yolo.py` | Remap the optional UMFIH classes |
| Merge | `data/merge_yolo_detection_datasets.py` | Merge compatible YOLO datasets |
| Prepare | `data/prepare_hierarchical_detection_dataset.py` | Collapse caries classes for hierarchical detection |
| Train | `training/train_detection.py` | Train the Ultralytics detector |
| Evaluate | `evaluation/eval_detection.py` | Evaluate a detector checkpoint and write JSON metrics |

## Severity and extension pipelines

| Group | Scripts |
|---|---|
| Severity data | `data/prepare_severity_dataset.py`, `data/pseudolabel_severity.py`, `data/prepare_followup_crop_dataset.py`, `data/prepare_periodontal_severity_dataset.py` |
| Severity model | `training/train_severity_classifier.py`, `evaluation/eval_severity_classifier.py`, `training/run_periodontal_severity_retrain.py` |
| Periodontal | `data/prepare_pdcnn_periodontal_yolo.py`, `evaluation/eval_periodontal_conf.py` |
| Pediatric | `data/prepare_pediatric_kaggle_detection_yolo.py`, `training/run_pediatric_service_finetune.py` |
| Six-class data | `data/prepare_roboflow_6class_yolo.py`, `data/merge_yolo_6class_stratified.py`, `data/filter_yolo_classes.py` |
| Data quality | `data/audit_yolo_detection_dataset.py`, `data/audit_roboflow_duplicates.py`, `data/build_yolo_label_cache.py` |
| Data reports | `data/generate_label_frequency_json.py`, `data/generate_csv_label_frequency_json.py` |

Historical Kaggle, Roboflow, and Zenodo download helpers were removed because they were tied to
one-off experiments or credentials. Dataset names, sources, licenses, and expected raw-data paths
remain documented in `ARCHITECTURE.md`.

The retired Zenodo five-class merge and one-class PDCNN conversion experiments were removed because
the current documented training and service pipelines no longer consume their outputs.
