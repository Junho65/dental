# Dental AI Project Architecture

## 1. 프로젝트 개요

본 프로젝트는 치과 X-ray 이미지에서 병변을 자동 검출하고, 충치 계열 병변은 추가 분류기를 통해 `caries`와
`deep_caries`로 세분화하는 2-stage 진단 보조 시스템을 구축하는 것을 목표로 한다.

이 프로젝트의 문제의식에는 국내 치과 진료 현장에서 제기되는 과잉진료 우려와 진료비 설명 부족 문제가 포함된다.
한국소비자원(2025)은 치과 관련 피해구제 신청에서 진료비 관련 분쟁이 증가하고 있으며, 치료비용계획서 제공
활성화가 필요하다고 지적했다. 또한 파노라마 X-ray 판독은 병변이 작거나 겹쳐 보이는 경우 해석 난도가 높아,
환자 입장에서는 진단의 적절성을 스스로 검증하기 어렵다. 이러한 맥락에서 설명 가능한 보조 판독 도구에 대한
수요가 존재한다.

본 프로젝트는 병변 위치와 유형을 일관된 기준으로 제시함으로써 판독의 재현성을 높이고 설명 가능성을 보완하는
진단 보조 시스템을 지향한다. 궁극적으로는 임상의의 판단을 지원하고, 환자와 의료진 사이의 커뮤니케이션 근거를
강화하는 것이 목적이다. 장기적으로는 X-ray 상의
병변 분류 결과를 바탕으로 치료 옵션별 적정 진료비 범위를 제시하는 보조 모듈까지 확장하는 것을 목표로 하며,
이 과정에서는 치과의사 자문을 반영해 임상적 타당성과 현실성을 확보한다.

현재 기본 학습 흐름은 다음과 같다.

- 1단계 detection: `caries_family`, `periapical_lesion`, `impacted_tooth`
- 2단계 severity classification: `caries` vs `deep_caries`
- 확장 후보 detection: Roboflow 공개 데이터 기반 `bone_loss`, `cyst`, `retained_root` 추가 실험
- 서비스 형태: Django 기반 웹서비스, 프런트엔드 업로드 화면과 `/predict/` API를 함께 제공
- 사용자 흐름: 사용자가 웹 화면에서 파노라마 X-ray 이미지를 업로드하면 백엔드 모델이 추론을 수행
- 향후 확장: 병변 유형별 치료 옵션과 적정 진료비 범위 제시 모듈 추가

프로젝트의 핵심 방향은 다음과 같다.

- 작은 병변과 위치 정보를 함께 다루기 위해 classification보다 detection을 우선 채택
- `deep_caries` 데이터 부족과 coarse label 문제를 줄이기 위해 hierarchical detection 구조 도입
- 로컬 학습, 평가, 서빙이 가능한 local-first 워크플로우 유지

## 2. 프로젝트 구성도

### 2.1 전체 시스템 아키텍처

```mermaid
graph TD
    raw1["DENTEX"]
    raw2["CariesXrays"]
    raw3["UMFIH"]
    user["End User"]
    web["Web Frontend"]

    subgraph build["Dataset Build"]
        prep1["DENTEX YOLO Preprocess"]
        prep2["CariesXrays YOLO Preprocess"]
        prep3["UMFIH Class Remap"]
        merge["Merged Detection Dataset Build"]
        hier["Hierarchical Detection Remap"]
        sev["Severity Crop Dataset Build"]
    end

    subgraph train["Training"]
        det["Hierarchical Detection Training"]
        sevtrain["Severity Classifier Training"]
        pseudo["Pseudo-label Refinement"]
    end

    subgraph serve["Serving"]
        yolo["YOLO Detector"]
        crop["ROI Crop"]
        cls["Severity Classifier"]
        refine["Severity Refinement"]
        cost["Cost Estimation Module"]
        rule["Dental Expert Rule Base"]
        api["Django /predict/ API"]
    end

    raw1 -->|"JSON annotations + images"| prep1
    raw2 -->|"VOC XML + images"| prep2
    raw3 -->|"YOLO labels + images"| prep3
    prep1 -->|"YOLO images + labels"| merge
    prep2 -->|"YOLO images + labels"| merge
    prep3 -->|"Remapped YOLO labels"| merge
    merge -->|"4-class merged detection dataset"| hier
    raw1 -->|"caries/deep_caries lesion source"| sev
    raw2 -->|"caries lesion source"| sev
    hier -->|"hierarchical train/val/test YAML"| det
    sev -->|"cropped lesion images + class labels"| sevtrain
    sevtrain -->|"teacher classifier checkpoint"| pseudo
    pseudo -->|"pseudo-labeled lesion crops"| sevtrain
    det -->|"detector checkpoint"| yolo
    sevtrain -->|"severity classifier checkpoint"| cls
    user -->|"panoramic X-ray upload"| web
    web -->|"multipart image request"| api
    api -->|"image tensor"| yolo
    yolo -->|"bbox + coarse lesion class"| api
    api -->|"caries_family ROI only"| crop
    crop -->|"cropped lesion tensor"| cls
    cls -->|"caries vs deep_caries score"| refine
    refine -->|"refined label merge"| api
    api -.->|"planned lesion summary"| cost
    rule -.->|"treatment and fee heuristics"| cost
    cost -.->|"estimated treatment options + fee range"| api
    api -->|"inference JSON + visualization data"| web
    web -->|"result view"| user
```

### 2.2 추론 흐름

```mermaid
graph LR
    user["User"] -->|"image upload"| web["Web Frontend"]
    web -->|"multipart/form-data request"| api["Django /predict/ API"]
    api -->|"decoded X-ray tensor"| det["Hierarchical Detector"]
    det -->|"bbox + confidence + coarse class"| bbox["caries_family / periapical_lesion / impacted_tooth"]
    bbox -->|"caries_family ROI only"| crop["ROI Crop"]
    crop -->|"cropped lesion tensor"| sev["Severity Classifier"]
    sev -->|"caries / deep_caries score"| refine["Severity Refinement"]
    bbox -->|"periapical_lesion / impacted_tooth kept as-is"| merge["Result Merge"]
    refine -->|"refined caries label"| merge
    merge -->|"lesion summary JSON"| out["Response Builder"]
    merge -.->|"planned lesion summary"| cost["Cost Estimation Module"]
    rule["Dental Expert Rule Base"] -.->|"treatment rules + fee table"| cost
    cost -.->|"estimated treatment options + fee range"| out
    out -->|"JSON + overlay metadata"| web
    web -->|"prediction result page"| user
```

### 2.3 현재 기본 학습 흐름

- 기본 학습 흐름: hierarchical detection training
- 기본 detection 데이터 구성: merged detection dataset에서 파생된 hierarchical dataset
- 기본 detection backbone: YOLOv8s
- 기본 초기화 방식: `yolov8s.pt` COCO pretrained checkpoint에서 시작하는 fine-tuning (`pretrained=True`)
- 기본 epoch: `50`
- 기본 해상도: `imgsz=416`
- 기본 배치 크기: `batch=8` (GTX 1660 6GB 기준, OOM 시 `4`, `2` 순서로 축소)
- 기본 DataLoader workers: `workers=4`
- 기본 early stopping patience: `10`

hierarchical detection 데이터가 준비되지 않은 경우에는 merged detection 데이터에서 coarse class 체계로
자동 재구성한 뒤 학습을 시작한다.

## 3. 사용할 데이터셋

### 3.1 DENTEX

- 용도: 기본 치과 X-ray detection 라벨 소스
- 원본 구조: JSON annotation 기반
- 프로젝트 내 역할:
  - detection supervised training source
  - severity crop supervised source
- annotation 처리 방식:
  - validation split은 JSON에 포함된 bbox를 그대로 사용해 YOLO bbox로 정규화한다.
  - test split은 polygon 형태의 `points`를 읽고, 각 점의 `x/y` 최솟값과 최댓값으로 축 정렬 외접 bbox를 만든 뒤 YOLO bbox로 변환한다.

사용 클래스:

- `caries`
- `deep_caries`
- `periapical_lesion`
- `impacted_tooth`

### 3.2 CariesXrays

- 용도: 충치 계열 bbox 확장 데이터
- 원본 구조: Pascal VOC XML
- 프로젝트 내 역할:
  - detection에서 충치 계열 표본 확장
  - severity pseudo-label 후보 crop source
- annotation 처리 방식:
  - VOC XML의 `object/bndbox`를 읽어 YOLO bbox로 변환한다.
  - 공개 라벨 중 충치 계열로 안전하게 해석되는 `Decay`만 사용하고 다른 클래스는 학습 라벨에서 제외한다.

프로젝트 매핑:

- VOC `Decay` -> project class `caries`

### 3.3 UMFIH Dental Pathology Dataset

- 용도: 추가 병변 데이터 보강
- 원본 구조: YOLO format
- 프로젝트 내 역할:
  - detection 데이터 다양성 확장
  - `periapical_lesion`, `impacted_tooth`, 일부 `caries` 보강
- annotation 처리 방식:
  - 원본 YOLO annotation을 그대로 읽되, 프로젝트 클래스 체계와 직접 정렬되는 클래스만 유지한다.
  - 유지 대상 클래스의 class id를 프로젝트 4-class 순서에 맞게 remap한다.

프로젝트 매핑:

- `Carious lesion (4)` -> `caries`
- `Apical periodontitis (7)` -> `periapical_lesion`
- `Impacted tooth (6)` -> `impacted_tooth`

### 3.4 현재 기본 학습 데이터셋

현재 기본 detection 학습은 merged detection 데이터에서 파생된 hierarchical 데이터셋을 사용한다.

- 원본 merged dataset classes:
  - `caries`
  - `deep_caries`
  - `periapical_lesion`
  - `impacted_tooth`
- hierarchical dataset classes:
  - `caries_family`
  - `periapical_lesion`
  - `impacted_tooth`

현재 로컬에 준비된 active split 규모:

- `train`: 5405 images
- `val`: 745 images
- `test`: 1316 images

### 3.5 Severity 분류 데이터셋

충치 계열 refinement를 위해 별도 crop classification 데이터셋을 사용한다.

- 입력: lesion crop
- 출력 클래스:
  - `caries`
  - `deep_caries`

추가로 CariesXrays lesion crop은 unlabeled pool로 저장한 뒤 pseudo-labeling에 사용할 수 있다.

### 3.6 Roboflow 확장 후보 데이터셋

지원 병변 범위를 넓히기 위해 Roboflow Universe의 공개 dental X-ray 데이터셋을 확장 후보로 검토한다. 1차 확장
실험에서는 기존 3-class detection에 다음 클래스를 추가하는 것을 목표로 한다.

- `bone_loss`
- `cyst`
- `retained_root`

1차 확장 후 detection taxonomy는 다음 6-class 체계를 사용한다.

- `caries_family`
- `periapical_lesion`
- `impacted_tooth`
- `bone_loss`
- `cyst`
- `retained_root`

Roboflow 원본 클래스는 프로젝트 표준 클래스명으로 remap한다.

- `Caries`, `cavity`, `decay` -> `caries_family`
- `Periapical lesion` -> `periapical_lesion`
- `impacted tooth` -> `impacted_tooth`
- `Bone Loss` -> `bone_loss`
- `Cyst` -> `cyst`
- `Retained root`, `Root Piece` -> `retained_root`

`Crown`, `Implant`, `Filling`, `Root canal filling`, `Amalgam filling`, `Composite filling` 등은 병변이라기보다
치료/보철 소견에 가까우므로 v1 확장 학습에서는 제외하고, 향후 별도 treatment/restoration finding 그룹으로 분리한다.

## 4. 데이터 전처리

### 4.1 Detection 전처리

Detection 데이터 전처리 단계는 다음과 같다.

1. DENTEX JSON annotation을 YOLO bbox format으로 변환
2. CariesXrays VOC annotation을 YOLO format으로 변환
3. UMFIH YOLO annotation을 프로젝트 4-class 체계로 remap
4. 여러 detection 데이터셋을 하나의 merged dataset으로 병합
5. 필요 시 train 이미지 oversampling manifest 생성

구현 수준에서의 주요 변환 규칙은 다음과 같다.

- DENTEX validation:
  - JSON annotation에 포함된 bbox를 사용한다.
  - bbox를 `(x_center, y_center, width, height)` 형태의 YOLO 정규화 좌표로 변환한다.
- DENTEX test:
  - 각 lesion shape의 polygon `points`를 읽는다.
  - `x_min`, `x_max`, `y_min`, `y_max`를 계산해 축 정렬 외접 bbox를 만든다.
  - 생성된 bbox를 YOLO 정규화 좌표로 변환한다.
- 클래스명 매핑:
  - DENTEX 문자열 라벨은 규칙 기반으로 `caries`, `deep_caries`, `periapical_lesion`, `impacted_tooth` 중 하나로 매핑한다.
  - CariesXrays는 VOC `Decay`를 `caries`로 매핑한다.
  - UMFIH는 `Carious lesion`, `Apical periodontitis`, `Impacted tooth`만 유지하고 각각 `caries`, `periapical_lesion`, `impacted_tooth`로 remap한다.
- 최종 출력 형식:
  - 모든 detection 데이터셋은 `images/{train,val,test}`와 `labels/{train,val,test}` 구조를 갖는 Ultralytics YOLO 형식으로 통일한다.
  - 각 label 파일은 `class_id x_center y_center width height` 한 줄당 한 객체 형식을 사용한다.

이 단계의 세부 실행 순서와 사용 명령은 별도 실행 가이드 문서에서 관리한다.

### 4.2 Hierarchical detection 전처리

Hierarchical detection에서는 4-class detection 라벨을 3-class coarse 라벨로 재구성한다.

- `caries` -> `caries_family`
- `deep_caries` -> `caries_family`
- `periapical_lesion` -> `periapical_lesion`
- `impacted_tooth` -> `impacted_tooth`

이 단계의 실제 실행 절차와 명령은 별도 실행 가이드 문서에서 설명한다.

### 4.3 Severity 데이터 전처리

Severity 데이터셋은 DENTEX lesion annotation을 crop으로 잘라 분류 문제로 재구성한다.

- labeled crop 생성
- `train/val/test` CSV 생성
- optional unlabeled crop pool 생성

세부 실행 절차와 데이터 준비 명령은 별도 실행 가이드 문서에서 설명한다.

### 4.4 데이터 분할 전략

현재 프로젝트에서 detection split은 데이터셋별 원본 구조를 최대한 유지하거나, 준비 스크립트에서 분할하여 사용한다.

향후 개선 포인트는 다음과 같다.

- detection 이미지 단위 multi-label stratified split 도입
- 희소 클래스(`deep_caries`, `impacted_tooth`) 분포 안정화
- train/val/test 간 클래스 불균형 완화

### 4.5 Roboflow 중복 이미지 검사

Roboflow 데이터는 기존 DENTEX, CariesXrays, UMFIH 이미지와 겹칠 수 있으므로, 병합 전에 이미지 내용 기반 중복
검사를 수행한다. 파일명은 Roboflow export 과정에서 바뀔 수 있으므로 중복 판정 기준으로 사용하지 않는다.

중복 검사는 `scripts/audit_roboflow_duplicates.py`에서 수행한다.

- exact duplicate:
  - 이미지 파일 bytes 기준 `SHA256`이 동일하면 중복 확정
- near duplicate:
  - 이미지를 정규화한 뒤 직접 구현한 `dHash`를 계산
  - Hamming distance `<= 4`이면 중복 확정
  - Hamming distance `5..8`이면 의심 중복으로 기록하되, 첫 실험에서는 보수적으로 제외
- 기존 `val` 또는 `test`와 겹치는 Roboflow 이미지는 데이터 누수를 막기 위해 무조건 제외
- 기존 `train`과 겹치는 Roboflow 이미지도 기본적으로 제외
- Roboflow 내부 중복은 하나만 남기고 제외

중복 검사 결과는 `reports/roboflow_audit/<timestamp>` 아래에 저장한다.

- `duplicate_exact.csv`
- `duplicate_near.csv`
- `duplicate_suspect.csv`
- `roboflow_keep.csv`
- `dedupe_summary.json`
- `duplicate_near_montage.png`
- `duplicate_suspect_montage.png`

## 5. 사용할 모델들 소개

### 5.1 Detection 모델

기본 detection 모델은 Ultralytics YOLOv8s이다.

- 현재 기본 task: hierarchical 3-class detection
- 현재 기본 초기화: 치과 X-ray 전용 공개 detection pretrained checkpoint는 사용하지 않고, COCO pretrained `yolov8s.pt`를 dental dataset에 fine-tuning

선택 이유:

- `YOLOv8n`보다 params/FLOPs가 커서 더 높은 표현력을 기대할 수 있음
- GTX 1660 6GB에서 `imgsz=416`, `batch=8` 기준으로 실험 가능한 크기
- Ultralytics 생태계를 이용해 학습, 검증, 체크포인트 관리가 단순함
- 비교 후보로 `YOLO11s`를 같은 데이터와 해상도에서 평가한다. Ultralytics 공식 수치 기준 `YOLO11s`는 `YOLOv8s`보다 가벼운 최신 small 계열 후보이며, `YOLOv8m` 이상은 6GB VRAM에서 OOM 및 학습 시간 리스크가 커 기본 후보에서 제외한다.

### 5.2 Severity 분류 모델

충치 세부 단계 분류는 TorchXRayVision DenseNet121 기반 classifier를 사용한다.

- 기본 모델명: `xrv_densenet121`
- 기본 pretrained weights: `densenet121-res224-all`
- fine-tuning 방식: pretrained backbone freeze + classifier head만 학습하는 head-only fine-tuning
- 입력 크기 기본값: `224`
- 출력 클래스:
  - `caries`
  - `deep_caries`

선택 이유:

- chest X-ray pretrained backbone을 사용해 ImageNet 대비 의료영상 도메인 편차를 줄일 수 있음
- 학습 데이터가 매우 작아 pretrained 표현을 최대한 보존하는 보수적 fine-tuning 전략이 필요함
- lesion crop 분류에 적합한 비교적 경량 CNN
- 로컬 GPU에서 학습 가능한 크기
- pseudo-label 재학습 루프와 결합하기 쉬움

### 5.3 서빙 구조

Django inference 단계에서는 detection 결과를 그대로 반환하지 않고, 조건부 refinement를 수행한다.

- 프런트엔드는 사용자가 파노라마 X-ray 이미지를 업로드하는 웹 화면을 제공한다.
- 백엔드는 업로드된 이미지를 `/predict/` API로 전달받아 전처리, detection, refinement, 응답 조립을 수행한다.
- 최종 응답은 bbox, 클래스, confidence, 세부 분류 결과, 향후에는 진료비 추정 정보를 포함하는 JSON 형태를 지향한다.
- `caries_family`가 검출되면 crop classifier로 세분화
- flat detector를 쓸 경우에도 `caries` / `deep_caries`를 후처리 refinement 가능

## 6. 성능평가 방안

### 6.1 Detection 평가 지표

Detection 모델은 다음 지표로 평가한다.

- `precision`
- `recall`
- `mAP50`
- `mAP50-95`

이 중 핵심 기준 지표는 `mAP50-95`다.

이유:

- IoU 0.50부터 0.95까지 여러 임계값에서 평균을 내므로 더 엄격함
- 단순히 객체를 찾는 것뿐 아니라 bbox 위치 정확도까지 반영함
- detection 모델 간 상대 비교에 가장 적합함

### 6.2 Early Stopping 기준

Detection 학습의 early stopping은 Ultralytics validator가 계산하는 fitness를 기준으로 동작한다.

- current detection fitness 기준: `metrics/mAP50-95(B)`
- 기본 patience: `10`

즉, 검증 `mAP50-95(B)`가 일정 epoch 동안 개선되지 않으면 학습을 중단한다.

### 6.3 Severity classifier 평가 지표

Severity 분류기는 다음 지표를 사용한다.

- `val_loss`
- `accuracy`
- `macro F1`

현재 기본 설정:

- best checkpoint 선택 기준: `val_loss`
- early stopping 기준: `val_loss`

### 6.4 실험 비교 방안

모델 비교는 다음 원칙으로 수행한다.

- 같은 데이터셋에서 비교
- 같은 `imgsz`에서 비교
- 가능한 경우 같은 batch / epoch / augmentation 조건 유지
- detection은 `best.pt` 기준으로 동일한 val/test 셋에서 재평가

주요 비교 축:

- flat 4-class vs hierarchical 3-class detection
- `imgsz=416` vs `imgsz=512`
- class balancing on/off
- pseudo-label severity classifier 사용 여부

### 6.5 최근 Detection 학습 결과

가장 최근에 완료된 detection 학습은 `yolov8s_hierarchical_e102_continue40`이다.

- 완료 시각: 2026-05-26 00:27 KST
- 학습 방식: 기존 `yolov8s_hierarchical_e102`의 `last.pt`에서 40 epochs 추가 학습
- 총 누적 학습량: 50 epochs (`yolov8s_hierarchical_e102` 10 epochs + `continue40` 40 epochs)
- 모델: YOLOv8s hierarchical 3-class detection
- 데이터: `data/detection_hierarchical/hierarchical_detection.yaml`
- 해상도 / 배치: `imgsz=416`, `batch=8`
- 장비: NVIDIA GeForce GTX 1660 6GB
- 검증 데이터: 745 images, 2085 instances
- best checkpoint: `runs/detect/artifacts/detection/yolov8s_hierarchical_e102_continue40/weights/best.pt`

Best checkpoint 기준 validation 결과:

| class | images | instances | precision | recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 745 | 2085 | 0.497 | 0.439 | 0.406 | 0.188 |
| caries_family | 731 | 1821 | 0.433 | 0.430 | 0.352 | 0.147 |
| periapical_lesion | 53 | 117 | 0.374 | 0.154 | 0.149 | 0.046 |
| impacted_tooth | 59 | 147 | 0.683 | 0.735 | 0.717 | 0.370 |

이전 기준선 `yolov8s_hierarchical_e102` 마지막 epoch와 비교하면 전체 mAP50은 `0.345`에서 `0.406`으로,
mAP50-95는 `0.153`에서 `0.188`로 개선되었다. 반면 precision은 `0.537`에서 `0.497`로 낮아지고,
recall은 `0.375`에서 `0.439`로 상승하여, 추가 학습 후 모델이 더 많은 병변 후보를 검출하는 방향으로 이동했다.

클래스별로는 `impacted_tooth` 성능이 가장 높고, `periapical_lesion`은 recall과 mAP가 낮아 추가 데이터 보강,
라벨 품질 점검, class balancing 또는 loss/augmentation 조정이 필요한 우선 개선 대상이다.

### 6.6 기록 및 모니터링

Detection 학습은 TensorBoard와 `results.csv`를 통해 모니터링한다.

주요 기록 항목:

- `train/box_loss`
- `train/cls_loss`
- `train/dfl_loss`
- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `metrics/precision(B)`
- `metrics/recall(B)`
- `val/*`

## 7. 개발 일정

현재 기준 시점은 2026년 4월 2주차이며, 개발 완료 목표 시점은 2026년 6월 2주차로 설정한다. 아래 일정은
향후 간트차트로 전환하기 쉽도록 주차 단위의 마일스톤 중심으로 정리한다.

### 7.1 4월 2주차-4월 3주차: 데이터 정비

- DENTEX, CariesXrays, UMFIH 원본 데이터와 메타정보 재확인
- detection dataset, hierarchical dataset 재생성 및 경로 검증
- split 전략 점검 및 stratified split 적용 여부 검토

### 7.2 4월 4주차-5월 1주차: Detection baseline 확정

- hierarchical detection 기본 학습 파이프라인 안정화
- `imgsz`, `batch`, `workers`, `amp` 조합 실험
- early stopping 기준과 best checkpoint 선정 방식 점검

### 7.3 5월 2주차-5월 3주차: Severity classifier 고도화

- labeled crop 기반 baseline classifier 학습
- class imbalance 보정 및 selection metric 조정
- best severity checkpoint 선정

### 7.4 5월 4주차: Pseudo-label 실험

- unlabeled crop에 teacher inference 수행
- high-confidence pseudo-label 추가
- pseudo-label 포함 재학습 전후 성능 비교

### 7.5 6월 1주차: 서빙 통합 및 API 안정화

- detection + severity refinement API 통합
- 환경변수 기반 가중치 교체 흐름 정리
- 에러 핸들링 및 응답 포맷 안정화

### 7.6 6월 2주차: 문서화 및 개발 완료

- 실험 결과표 및 모델 비교표 정리
- 배포/운영 문서 보강
- 최종 발표용 자료 및 간트차트 정리

## 8. 활용 방안

본 프로젝트의 활용 가능 시나리오는 다음과 같다.

- 치과 X-ray 1차 판독 보조
- 충치 의심 병변의 위치 제안 및 세부 단계 보조 분류
- 병변 유형 기반 치료 옵션 정리 및 적정 진료비 범위 제시
- 교육용 annotation 보조 도구
- 연구용 baseline 및 실험 플랫폼
- 향후 진단 리포트 자동화와 상담 지원 시스템의 전단계 모듈

추가 확장 가능성:

- 치과의사 자문 기반 진료비 계산 모듈 구축
- 환자 단위 리포트 생성
- active learning 기반 재라벨링 루프
- 다기관 데이터 추가 학습
- 웹 대시보드와 진료 워크플로우 통합

## 9. 참고문헌 (Reference)

### 9.1 데이터셋 및 데이터 소스

- Chen, B., Fu, S., Liu, Y., Pan, J., Lu, G., & Zhang, Z. (2024). *CariesXrays: Enhancing caries detection in hospital-scale panoramic dental X-rays via feature pyramid contrastive learning*. Proceedings of the AAAI Conference on Artificial Intelligence, 38(20), 21940-21948. https://doi.org/10.1609/aaai.v38i20.30196
- Chen, B. (n.d.). *AAAI2024_CariesXrays* [Data set and code repository]. GitHub. https://github.com/Binz-Chen/AAAI2024_CariesXrays
- Hamamci, I. E., Er, S., Simsar, E., Yuksel, A. E., Gultekin, S., Ozdemir, S. D., Yang, K., Li, H. B., Pati, S., Stadlinger, B., Mehl, A., Gundogar, M., & Menze, B. (2023). *DENTEX: An abnormal tooth detection with dental enumeration and diagnosis benchmark for panoramic X-rays*. arXiv. https://arxiv.org/abs/2305.19112
- LUNA0206. (n.d.). *DENTEX* [Data set]. Hugging Face. https://huggingface.co/datasets/LUNA0206/DENTEX
- Mureșanu, S., Hedeșiu, M., & Iacob, L.-M. (2025). *Dataset for automating dental condition detection on panoramic radiographs* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15487430

### 9.2 모델 및 프레임워크

- Django Software Foundation. (n.d.). *Django documentation (Version 4.2)*. https://docs.djangoproject.com/en/4.2/contents/
- Encode OSS Ltd. (n.d.). *Django REST framework*. https://www.django-rest-framework.org/
- PyTorch Contributors. (n.d.). *PyTorch documentation*. https://docs.pytorch.org/docs/stable/index.html
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks*. Proceedings of the 36th International Conference on Machine Learning, 97, 6105-6114. https://proceedings.mlr.press/v97/tan19a.html
- Ultralytics. (n.d.). *Ultralytics YOLO docs*. https://docs.ultralytics.com/
- Ultralytics. (n.d.). *YOLOv8 model documentation*. https://docs.ultralytics.com/models/yolov8/
- Ultralytics. (n.d.). *YOLO11 model documentation*. https://docs.ultralytics.com/models/yolo11/

### 9.3 배경 자료

- Korea Consumer Agency. (2025, December 5). *Dental treatment cost disputes are rapidly increasing; provision of treatment cost plans needs to be activated*. https://www.kca.go.kr/home/sub.do?menukey=4005&mode=view&no=1003974692
