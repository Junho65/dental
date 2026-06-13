import tempfile
import time
import uuid
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser

from .inference import Predictor
from .mdfee import MDFEE_SOURCE, attach_treatment_estimates

_predictor = None


def _json_no_cache(data: dict, status: int = 200) -> JsonResponse:
    resp = JsonResponse(data, status=status)
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp["Pragma"] = "no-cache"
    return resp
_CLASS_INFO = [
    {
        "name": "caries",
        "label": "Caries",
        "label_ko": "충치",
        "official_label_ko": "치아우식증",
        "description": "법랑질이나 상아질의 충치를 의미합니다.",
        "patient_explanation": "치아 표면이나 안쪽이 단단함을 잃고 손상된 부위예요. 보통 진행 전이라면 메우는 치료로 마무리됩니다.",
        "patient_next_step": "치과에서 정밀 검진 후 레진 수복 등 보존 치료를 받는 것이 일반적입니다.",
        "color": "#ff7a59",
    },
    {
        "name": "periapical_lesion",
        "label": "Periapical Lesion",
        "label_ko": "치근단 병소",
        "official_label_ko": "치근단 병소",
        "description": "치근단 주변 염증성 병소 의심 영역입니다.",
        "patient_explanation": "치아 뿌리 끝 주변에 염증이 의심되는 부위예요. 신경이 손상되었을 가능성이 있습니다.",
        "patient_next_step": "근관(신경) 치료 또는 재근관 치료가 필요한지 치과에서 확인해 보세요.",
        "color": "#33c7b5",
    },
    {
        "name": "impacted_tooth",
        "label": "Impacted Tooth",
        "label_ko": "매복치",
        "official_label_ko": "매복치",
        "description": "매복치처럼 정상 맹출이 어려운 치아를 뜻합니다.",
        "patient_explanation": "잇몸이나 뼈 속에 머물러 정상적으로 나오지 못한 치아예요. 사랑니인 경우가 많습니다.",
        "patient_next_step": "발치 또는 외과적 처치 필요 여부를 치과 전문의와 상담해 보세요.",
        "color": "#4d8dff",
    },
    {
        "name": "bone_loss",
        "label": "Bone Loss",
        "label_ko": "치조골 소실",
        "official_label_ko": "치조골 소실",
        "description": "치주염으로 인한 치조골(치아를 지지하는 뼈) 소실 의심 부위입니다.",
        "patient_explanation": "잇몸 질환으로 치아를 지지하는 뼈가 녹아내린 부위예요. 심각도에 따라 경도·중등도·중증으로 구분됩니다.",
        "patient_next_step": "치주 전문의 상담을 통해 스케일링, 치근 활택술, 또는 외과적 치주 치료를 받으세요.",
        "color": "#e05c5c",
    },
    {
        "name": "furcation_involvement",
        "label": "Furcation Involvement",
        "label_ko": "분기부 병소",
        "official_label_ko": "치근 이개부 병변",
        "description": "다근치 치아의 치근 이개부까지 치주 병변이 침범한 부위입니다.",
        "patient_explanation": "어금니 뿌리가 갈라지는 부위까지 잇몸 병변이 진행된 상태예요. 치근 이개부 병변이라고도 하며, 심각도에 따라 경도·중증으로 구분됩니다.",
        "patient_next_step": "치주 전문의 상담이 필요하며, 중증일 경우 발치 또는 터널형성술을 고려할 수 있습니다.",
        "color": "#c084fc",
    },
    {
        "name": "retained_root",
        "label": "Retained Root",
        "label_ko": "잔존치근",
        "official_label_ko": "잔존치근",
        "description": "발치 후 치근이 잇몸 안에 남아있는 상태입니다.",
        "patient_explanation": "치아가 빠지거나 부러진 뒤 뿌리 일부가 잇몸 안에 남아있는 부위예요. 방치하면 염증이나 통증이 생길 수 있습니다.",
        "patient_next_step": "치과에서 잔존 치근 발거(외과적 발치) 필요 여부를 확인하세요.",
        "color": "#a0785a",
    },
]


def dashboard(request):
    context = {
        "class_info": _CLASS_INFO,
    }
    resp = render(request, "classifier/dashboard.html", context)
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp["Pragma"] = "no-cache"
    return resp


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@api_view(["GET"])
def health(request):
    return JsonResponse({"status": "ok"})


@api_view(["POST"])
@parser_classes([MultiPartParser])
def predict(request):
    file_obj = request.FILES.get("image")
    if file_obj is None:
        return _json_no_cache({"error": "Missing file field: image"}, status=400)

    if getattr(file_obj, "size", 0) == 0:
        return _json_no_cache({"error": "Empty upload (0 bytes)"}, status=400)

    suffix = Path(file_obj.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        for chunk in file_obj.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        t0 = time.perf_counter()
        predictor = get_predictor()
        t_after_model = time.perf_counter()
        result = predictor.predict_file(tmp_path)
        t1 = time.perf_counter()
        if not isinstance(result, dict):
            result = {"detections": []}
        attach_treatment_estimates(result)
        # Model forward only (weights are loaded lazily on first request in Predictor.__init__).
        result["inference_ms"] = round((t1 - t_after_model) * 1000, 1)
        result["server_total_ms"] = round((t1 - t0) * 1000, 1)
        result["request_id"] = str(uuid.uuid4())
        result["weights_path"] = predictor.weights_path
        result["predict_conf"] = predictor.conf
        result["predict_imgsz"] = predictor.imgsz
        result.setdefault("fee_source", MDFEE_SOURCE if result.get("fee_estimate_enabled") else None)
        return _json_no_cache(result)
    except Exception as e:
        return _json_no_cache({"error": str(e)}, status=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
