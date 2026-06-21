# Dental X-ray Detection + Django Service

치과 파노라마 X-ray에서 주요 병변과 치주 병변을 탐지하고, 치주 병변의 중증도와
건강보험심사평가원 수가 기반 치료 정보를 함께 제공하는 Django 프로젝트입니다.

현재 저장소에는 Django 서빙에 필요한 가중치 4개와 SQLite DB가 포함되어 있습니다.
따라서 웹 서비스 실행만 할 때는 학습 데이터셋이나 별도 학습이 필요하지 않습니다.

스크립트 전체 목록은 [`scripts/README.md`](scripts/README.md)에서 확인할 수 있습니다.

## 1. 현재 서빙 구성

| 구성 요소 | 클래스 | 기본 가중치 | 학습 데이터 |
|---|---|---|---|
| 주요 병변 detector | `caries_family`, `periapical_lesion`, `impacted_tooth`, `retained_root` | `artifacts/detection/serve/dental_4class_detection_best.pt` | `data/detection_main_4class_with_pediatric` |
| 치주 detector | `bone_loss`, `furcation_involvement` | `artifacts/detection/serve/periodontal_2class_detection_best.pt` | `data/detection_periodontal_pdcnn_2class_bg` |
| Bone-loss severity | `mild`, `medium`, `severe` | `artifacts/severity/serve/bone_loss/best.pt` | `data/severity_periodontal/bone_loss` |
| Furcation severity | `mild`, `severe` | `artifacts/severity/serve/furcation_involvement/best.pt` | `data/severity_periodontal/furcation_involvement` |

주요 detector의 `caries_family`는 API에서 `caries`로 정규화됩니다. 최종 API 클래스 순서는
다음과 같습니다.

```text
caries, periapical_lesion, impacted_tooth,
bone_loss, furcation_involvement, retained_root
```

치근단 병소용 follow-up 분류기는 선택 기능입니다. 기본 경로인
`artifacts/severity/serve/periapical_followup/best.pt`가 현재 저장소에 없으므로 기본 상태에서는
비활성화되고, 치근단 병소 치료 경로는 규칙 기반 기본값을 사용합니다.

## 2. 빠른 실행

모든 명령은 저장소 루트에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py check
python manage.py runserver
```

CUDA를 사용할 경우 평가 PC의 드라이버에 맞는 PyTorch wheel을 먼저 설치한 뒤
`requirements.txt`를 설치하는 편이 안전합니다. 기본 웹 추론 장치는 CPU입니다.

실행 후 다음 주소를 사용합니다.

- 대시보드: `http://127.0.0.1:8000/`
- 상태 확인: `GET http://127.0.0.1:8000/health/`
- 추론 API: `POST http://127.0.0.1:8000/predict/`

첫 추론 요청 때 모델들이 지연 로딩되므로 첫 요청은 이후 요청보다 오래 걸릴 수 있습니다.

## 3. 추론 API

`/predict/`에는 `image` 필드로 multipart 파일을 전송합니다.

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/ -F "image=@sample.png"
```

치주 병변이 검출된 경우의 축약 응답 예시는 다음과 같습니다.

```json
{
  "detections": [
    {
      "class_id": 3,
      "model_class_id": 0,
      "class_name": "bone_loss",
      "model_class_name": "bone_loss",
      "confidence": 0.91,
      "bbox_xyxy": [120.1, 340.2, 220.3, 440.4],
      "severity_class_name": "medium",
      "severity_confidence": 0.78,
      "severity_probabilities": {
        "mild": 0.15,
        "medium": 0.78,
        "severe": 0.07
      },
      "severity_applied": true,
      "treatment_estimate": {}
    }
  ],
  "periapical_followup_enabled": false,
  "periodontal_severity_enabled": true,
  "fee_estimate_enabled": true,
  "inference_ms": 1234.5,
  "server_total_ms": 1250.2,
  "request_id": "...",
  "predict_conf": 0.1,
  "predict_imgsz": 512
}
```

실제 검출 객체에는 모델 클래스, 서빙 클래스, bbox, confidence가 포함됩니다. 치주 병변에는
severity 필드가 추가되고, 저장된 수가 정보가 있으면 `treatment_estimate`가 추가됩니다.

## 4. SQLite와 치료 수가 데이터

Django가 사용하는 DB는 저장소 루트의 `db.sqlite3`입니다. DB에는 migration 이력과
`classifier_mdfeeitem` 수가 데이터가 포함되어 있으며 Git에서 추적합니다.

수가 데이터를 다시 동기화할 때만 공공데이터포털 API 키가 필요합니다.

```powershell
$env:DATAGO_KEY="발급받은_서비스키"
python manage.py sync_mdfees
```

또는 Git에서 제외되는 루트 `.env` 파일에 다음과 같이 저장할 수 있습니다.

```dotenv
DATAGO_KEY=발급받은_서비스키
DATAGO_MDFEE_TIMEOUT=30
```

수가 API를 사용할 수 없거나 DB에 대응 항목이 없으면 서버는 설정된 치료 검색어를 이용한
keyword-only 정보를 반환합니다.

현재 Django 설정은 `DEBUG=True`, 전체 host 허용, 인증 없음인 평가·개발용 구성입니다.
인터넷에 그대로 공개하는 운영 설정으로 사용하면 안 됩니다.

## 5. 서빙 환경변수

환경변수는 프로세스 환경 또는 루트 `.env`에서 읽습니다.

| 변수 | 기본값/기본 경로 | 용도 |
|---|---|---|
| `DENTAL_YOLO_WEIGHTS` | `artifacts/detection/serve/dental_4class_detection_best.pt` | 주요 detector 교체 |
| `DENTAL_WEIGHTS_PATH` | 없음 | 주요 detector의 이전 호환 별칭 |
| `DENTAL_PERIODONTAL_WEIGHTS` | `artifacts/detection/serve/periodontal_2class_detection_best.pt` | 치주 detector 교체 |
| `DENTAL_BL_SEVERITY_WEIGHTS` | `artifacts/severity/serve/bone_loss/best.pt` | bone-loss severity 교체 |
| `DENTAL_FI_SEVERITY_WEIGHTS` | `artifacts/severity/serve/furcation_involvement/best.pt` | furcation severity 교체 |
| `DENTAL_PERIAPICAL_FOLLOWUP_WEIGHTS` | 선택 사항 | 치근단 follow-up 분류기 활성화 |
| `DENTAL_PREDICT_DEVICE` | `cpu` | YOLO 추론 장치 |
| `DENTAL_SEVERITY_DEVICE` | `DENTAL_PREDICT_DEVICE` 값 | severity 추론 장치 |
| `DENTAL_PREDICT_CONF` | `0.1` | 주요 detector confidence |
| `DENTAL_PERIODONTAL_PREDICT_CONF` | `0.4` | 치주 detector confidence |
| `DENTAL_PREDICT_IMGSZ` | `512` | 두 detector의 서빙 입력 크기 |
| `DENTAL_SEVERITY_CROP_MARGIN` | `0.15` | 치주 crop 여백 비율 |
| `DENTAL_PERIAPICAL_FOLLOWUP_CONF` | `0.75` | follow-up 적용 confidence |
| `DENTAL_MDFEE_ENDPOINT` | 코드의 공공데이터포털 endpoint | 수가 API endpoint 교체 |
| `DATAGO_MDFEE_TIMEOUT` | `30` | 수가 API timeout(초) |

상대경로로 지정한 가중치는 저장소 루트를 기준으로 해석됩니다.

## 6. 학습 데이터셋

`data/`는 용량과 라이선스 문제로 Git에 포함하지 않습니다. 평가용으로 정리한 학습 데이터셋은
다음 패키지에서 받을 수 있습니다.

- [학습 데이터셋 패키지 (Google Drive)](https://drive.google.com/file/d/1rKDykKpzKTjm9O2X0Vg3ZVDYdnK9Agm-/view?usp=sharing)

압축을 저장소 루트에 풀었을 때 아래 경로가 있어야 현재 모델 학습·평가 명령을 실행할 수 있습니다.

| 경로 | 규모 | 용도 |
|---|---:|---|
| `data/detection_main_4class_with_pediatric` | 8,267 images / 24,629 boxes | 현재 주요 detector |
| `data/detection_periodontal_pdcnn_2class_bg` | 1,745 images / 33,413 boxes | 현재 치주 detector |
| `data/severity_periodontal/bone_loss` | 29,870 crops | 3-class bone-loss severity |
| `data/severity_periodontal/furcation_involvement` | 3,543 crops | 2-class furcation severity |

통계 원본은 `artifacts/data_stats/`에 있으며 데이터 YAML과 CSV는 저장소 상대경로를 사용합니다.

### 원본 데이터 출처

| 원본 | 링크 | 기본/대표 raw 경로 | 프로젝트 내 역할 |
|---|---|---|---|
| DENTEX | [Hugging Face](https://huggingface.co/datasets/LUNA0206/DENTEX) | `data/raw/dentex` | 주요 병변 기본 라벨 |
| CariesXrays | [AAAI2024_CariesXrays](https://github.com/Binz-Chen/AAAI2024_CariesXrays) | `data/raw/cariesxrays` | 우식 병변 보강 |
| UMFIH 14-class | [Zenodo 15487430](https://zenodo.org/records/15487430) | `data/raw/umfih/extracted` | 우식·치근단·매복치 보강 |
| Adult panoramic | [Kaggle](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset) | `data/raw/kaggle/dental_disease_panoramic_detection` | 주요 병변 보강 |
| Pediatric panoramic | [Kaggle](https://www.kaggle.com/datasets/truthisneverlinear/childrens-dental-panoramic-radiographs-dataset) | `data/raw/kaggle/archive_4_bundle` | 소아 도메인 보강 |
| PDCNN periodontal | [GitHub](https://github.com/PuckBlink/PDCNN), [Zenodo 15565284](https://zenodo.org/records/15565284) | `data/raw/pdcnn_periodontitis_bone_loss` | 치주 detector와 severity |

각 원본의 라이선스와 재배포 조건은 원본 페이지에서 별도로 확인해야 합니다. DENTEX downloader는
데이터 카드의 `CC-BY-NC-SA-4.0` 조건을 명시합니다.

## 7. 데이터 준비

최종 압축 데이터셋을 사용하면 이 단계는 건너뛸 수 있습니다. 다음 명령은 개별 원본에서
보조·기본 데이터셋을 다시 만들 때 사용합니다.

```powershell
# DENTEX 4-class 기본 데이터
python scripts/data/download_dentex_dataset.py
python scripts/data/prepare_detection_dataset.py

# 선택 UMFIH 확장
python scripts/data/download_umfih_dataset.py
python scripts/data/prepare_umfih_yolo.py

# PDCNN 치주 2-class + background, 이후 severity crop
python scripts/data/prepare_pdcnn_periodontal_yolo.py
python scripts/data/prepare_periodontal_severity_dataset.py

# 준비된 소아 원본에서 선택 6-class 데이터 구성
python scripts/data/prepare_pediatric_kaggle_detection_yolo.py
```

`prepare_detection_dataset.py`의 출력은 `data/detection/dentex_detection.yaml`이고 클래스는
`caries`, `deep_caries`, `periapical_lesion`, `impacted_tooth`입니다. 이 기본 데이터는 현재
서빙용 4-class 결합 데이터셋과 동일하지 않습니다.

`prepare_hierarchical_detection_dataset.py`는 `caries`와 `deep_caries`를 `caries_family`로
합친 3-class 보조 baseline을 만듭니다. 이것 역시 현재 서빙용 4-class 데이터셋과 구분해야 합니다.

원시 데이터부터 현재 최종 결합 데이터셋 전체를 만드는 단일 downloader는 제공하지 않습니다.
Kaggle 인증과 일회성 수집에 의존하던 downloader는 제거했으며, 정확한 재학습에는 위의 준비된
데이터셋 패키지를 사용하는 것이 권장됩니다.

## 8. 모델 학습

### 8.1 현재 주요 detector 재학습

```powershell
python scripts/training/train_detection.py `
  --data data/detection_main_4class_with_pediatric/main_4class_with_pediatric.yaml `
  --model artifacts/detection/serve/dental_4class_detection_best.pt `
  --name main_4class_retrain `
  --epochs 50 --imgsz 416 --batch 8 --workers 4
```

COCO 사전학습 모델에서 새로 시작하려면 `--model yolov8s.pt`를 사용합니다. 해당 파일이 로컬에
없으면 Ultralytics가 다운로드합니다.

소아 fine-tuning, main/pediatric 이중 평가와 promotion gate를 한 번에 실행하려면 다음 스크립트를
사용합니다. 기본값은 현재 서빙 가중치를 초기값으로 사용하며, 안전한 검토 실행에는
`--no-promote`를 붙입니다.

```powershell
python scripts/training/run_pediatric_service_finetune.py --no-promote
```

기록된 승격 결과는
[`reports/yolov8s_serve_pediatric_ft_v1/promotion_summary.json`](reports/yolov8s_serve_pediatric_ft_v1/promotion_summary.json)에 있습니다.

### 8.2 치주 detector 재학습

```powershell
python scripts/training/train_detection.py `
  --data data/detection_periodontal_pdcnn_2class_bg/periodontal_pdcnn_2class.yaml `
  --model artifacts/detection/serve/periodontal_2class_detection_best.pt `
  --name periodontal_2class_retrain `
  --epochs 80 --patience 15 --imgsz 640 --batch 4 --workers 0
```

기록된 승격 모델은 background 음성 이미지를 포함해 재학습되었습니다. 과거 실행은 별도의
5-class checkpoint에서 시작했지만 그 중간 checkpoint는 제출 대상이 아닙니다. 포함된 서빙
checkpoint에서 계속 학습하거나 `yolov8s.pt`에서 새로 학습할 수 있습니다.

### 8.3 치주 severity 재학습

```powershell
python scripts/training/run_periodontal_severity_retrain.py `
  --epochs 20 `
  --selection-metric val_f1_macro `
  --early-stopping-metric val_f1_macro
```

기본값은 현재 서빙 checkpoint로 두 분류기를 warm-start하고 결과를
`artifacts/severity/periodontal_retrain/<timestamp>/`에 저장합니다. 기본적으로 서빙 가중치를
교체하지 않습니다. 후보의 test macro-F1이 기존 모델 이상일 때 교체하려면 `--promote`를 추가합니다.

### 8.4 범용 detector 스크립트 기본값

인자 없이 `scripts/training/train_detection.py`를 실행하면 현재 서빙 4-class 모델이 아니라
3-class hierarchical baseline을 학습합니다.

| 인자 | 기본값 |
|---|---|
| `--data` | `data/detection_hierarchical/hierarchical_detection.yaml` |
| `--model` | `yolov8s.pt` |
| `--epochs` / `--patience` | `50` / `10` |
| `--imgsz` / `--batch` | `416` / `8` |
| `--workers` / `--device` | `4` / `0` |
| `--project` / `--name` | `artifacts/detection` / `yolov8s_hierarchical` |
| AMP / TensorBoard | 활성화 |

GTX 1660 6GB에서 CUDA OOM이 발생하면 `--batch 4`, 이후 `--batch 2`로 낮춥니다.
TensorBoard 로그는 `%TEMP%/dental_yolo_tb/<run-name>`에 기록됩니다.

```powershell
tensorboard --logdir "$env:TEMP\dental_yolo_tb" --reload_interval 2
```

## 9. 평가

### 주요 detector

```powershell
python scripts/evaluation/eval_detection.py `
  --weights artifacts/detection/serve/dental_4class_detection_best.pt `
  --data data/detection_main_4class_with_pediatric/main_4class_with_pediatric.yaml `
  --imgsz 416 --split test --workers 0 `
  --out reports/main_serve_eval.json
```

### 치주 detector와 background false positive

```powershell
python scripts/evaluation/eval_periodontal_conf.py `
  --weights artifacts/detection/serve/periodontal_2class_detection_best.pt `
  --data data/detection_periodontal_pdcnn_2class_bg/periodontal_pdcnn_2class.yaml `
  --imgsz 640 --split val --workers 0 `
  --out reports/periodontal_serve_eval.json
```

기록된 비교 결과는
[`reports/periodontal_bg_eval_new.json`](reports/periodontal_bg_eval_new.json)에 있습니다.

### Severity classifier

```powershell
python scripts/evaluation/eval_severity_classifier.py `
  --checkpoint artifacts/severity/serve/bone_loss/best.pt `
  --test-csv data/severity_periodontal/bone_loss/test.csv `
  --out-json reports/bone_loss_severity_test.json

python scripts/evaluation/eval_severity_classifier.py `
  --checkpoint artifacts/severity/serve/furcation_involvement/best.pt `
  --test-csv data/severity_periodontal/furcation_involvement/test.csv `
  --out-json reports/furcation_severity_test.json
```

## 10. 저장소 구조

```text
artifacts/
  data_stats/             최종 학습 데이터 통계와 하이퍼파라미터 요약
  detection/serve/        서빙 detector 2개
  severity/serve/         서빙 severity classifier 2개
django_app/               Django 화면, API, 수가 연동
reports/                  평가·promotion·데이터 감사 결과
scripts/data/             데이터 준비·병합·감사
scripts/training/         detector와 classifier 학습
scripts/evaluation/       checkpoint 평가
src/severity/             severity dataset/model/inference 구현
db.sqlite3                Django SQLite DB
manage.py                 Django 명령 진입점
```

생성되는 `data/`, `runs/`, 임시 로그, 로컬 `.env`, Ultralytics 설정은 Git에서 제외합니다.

## 11. 문제 해결

### `warning: unable to unlink 'db.sqlite3'`

Windows에서 `python manage.py runserver`가 DB를 사용 중일 때 Git이 DB를 교체하려 하면 발생합니다.
개발 서버 터미널에서 `Ctrl+C`로 종료한 뒤 branch 전환·merge를 다시 실행합니다.

### CUDA OOM

학습 batch를 `8 → 4 → 2` 순서로 낮춥니다. 웹 서비스는 기본값이 CPU이므로 GPU 메모리가 작은
평가 PC에서도 실행할 수 있습니다.

### `WinError 1455`

`torch` 또는 `ultralytics` import 중 발생하면 Windows 가상 메모리(page file)를 늘리고 재부팅합니다.

### 첫 추론이 느림

YOLO detector 2개와 severity classifier 2개가 첫 요청 때 로딩됩니다. 이후 요청 시간과 구분할 수
있도록 응답에 `inference_ms`와 `server_total_ms`가 함께 포함됩니다.
