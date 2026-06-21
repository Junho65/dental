# Script guide

모든 명령은 저장소 루트에서 실행합니다. 현재 서빙 모델 학습 경로와 보조 데이터 준비 경로를
구분해야 합니다. 인자 없는 `training/train_detection.py`는 3-class hierarchical baseline용이며,
현재 서빙되는 4-class 주요 detector의 기본 재학습 명령은 루트 [`README.md`](../README.md)에 있습니다.

## 빠른 실행

저장소에 서빙 가중치 4개와 `db.sqlite3`가 포함되어 있으므로 데이터 다운로드나 학습 없이 실행할 수 있습니다.

```powershell
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
python manage.py runserver
```

## 현재 모델 학습 진입점

| 대상 | 진입점 | 기본 입력 |
|---|---|---|
| 주요 4-class detector | `training/train_detection.py` | 명시적으로 `data/detection_main_4class_with_pediatric/main_4class_with_pediatric.yaml` 전달 |
| 소아 fine-tuning + promotion gate | `training/run_pediatric_service_finetune.py` | 주요 4-class base + 소아 4-class |
| 치주 2-class detector | `training/train_detection.py` | 명시적으로 `data/detection_periodontal_pdcnn_2class_bg/periodontal_pdcnn_2class.yaml` 전달 |
| 치주 severity 2종 | `training/run_periodontal_severity_retrain.py` | `data/severity_periodontal/{bone_loss,furcation_involvement}` |

실제 명령, 기본값, 가중치와 평가 방법은 루트 README의 모델 학습·평가 절을 따릅니다.

## 디렉터리 구성

| 디렉터리 | 내용 |
|---|---|
| `scripts/data/` | 데이터 변환, 병합, 필터링, 중복·라벨 감사, 통계 생성 |
| `scripts/training/` | detector/classifier 학습과 재학습 orchestration |
| `scripts/evaluation/` | detector, severity, confidence 독립 평가 |

## Data scripts

| 스크립트 | 용도 |
|---|---|
| `data/download_dentex_dataset.py` | Hugging Face에서 DENTEX 다운로드 |
| `data/download_umfih_dataset.py` | Zenodo에서 UMFIH archive 다운로드·압축 해제 |
| `data/prepare_detection_dataset.py` | DENTEX를 4-class YOLO 데이터로 변환 |
| `data/prepare_cariesxrays_yolo.py` | CariesXrays VOC annotation 변환 |
| `data/prepare_umfih_yolo.py` | UMFIH 14-class 중 호환 병변만 4-class schema로 remap |
| `data/merge_yolo_detection_datasets.py` | class order가 동일한 YOLO 데이터셋 병합 |
| `data/prepare_hierarchical_detection_dataset.py` | caries/deep-caries를 합친 3-class baseline 생성 |
| `data/prepare_pediatric_kaggle_detection_yolo.py` | 소아 Kaggle 원본에서 선택 6-class 데이터 생성 |
| `data/filter_yolo_classes.py` | 필요한 class subset만 유지하고 ID 재매핑 |
| `data/prepare_pdcnn_periodontal_yolo.py` | PDCNN에서 background 포함 치주 2-class 데이터 생성 |
| `data/prepare_periodontal_severity_dataset.py` | 치주 manifest에서 BL/FI severity crop 생성 |
| `data/prepare_followup_crop_dataset.py` | 임의 병변 follow-up classifier용 crop CSV 생성 |
| `data/prepare_severity_dataset.py` | 기존 caries severity crop 데이터 준비 |
| `data/pseudolabel_severity.py` | severity pseudo label 생성 |
| `data/prepare_roboflow_6class_yolo.py` | Roboflow 계열 원본을 6-class schema로 변환 |
| `data/merge_yolo_6class_stratified.py` | 6-class 데이터의 stratified 병합 |
| `data/audit_yolo_detection_dataset.py` | YOLO label, split, image 크기 감사 |
| `data/audit_roboflow_duplicates.py` | exact/near duplicate 감사와 keep CSV 생성 |
| `data/build_yolo_label_cache.py` | 대용량 label cache 생성 |
| `data/generate_label_frequency_json.py` | detection label 빈도 통계 생성 |
| `data/generate_csv_label_frequency_json.py` | classifier CSV label 빈도 통계 생성 |

## Training scripts

| 스크립트 | 용도 |
|---|---|
| `training/train_detection.py` | 범용 Ultralytics detector 학습 |
| `training/run_pediatric_service_finetune.py` | 소아 병합, smoke/full 학습, 이중 평가, promotion gate |
| `training/train_severity_classifier.py` | TorchXRayVision DenseNet121 crop classifier 학습 |
| `training/run_periodontal_severity_retrain.py` | BL/FI severity warm-start 재학습·평가·선택적 승격 |

## Evaluation scripts

| 스크립트 | 용도 |
|---|---|
| `evaluation/eval_detection.py` | YOLO checkpoint의 split별 detection 지표 저장 |
| `evaluation/eval_periodontal_conf.py` | 치주 지표, F1-optimal confidence, background FP 평가 |
| `evaluation/eval_severity_classifier.py` | severity checkpoint의 held-out CSV 평가 |

## 범위와 재현성

- `data/`는 Git에 포함되지 않으며 루트 README의 Google Drive 패키지를 사용합니다.
- DENTEX와 UMFIH downloader만 유지합니다. Kaggle 인증이나 일회성 URL에 의존한 downloader는 제거했습니다.
- DENTEX → hierarchical 3-class 흐름은 보조 baseline이며 현재 서빙 4-class 모델과 동일하지 않습니다.
- 기록된 치주 모델의 최초 5-class 초기 checkpoint와 과거 main baseline backup은 제출 대상이 아닙니다.
  포함된 서빙 checkpoint에서 계속 학습하거나 COCO 사전학습 `yolov8s.pt`에서 새로 학습할 수 있습니다.
- 평가·promotion 근거는 `reports/`, 최종 데이터 통계는 `artifacts/data_stats/`에 있습니다.
