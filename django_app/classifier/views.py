import tempfile
import time
import uuid
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser

from .inference import Predictor

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
        "description": "법랑질이나 상아질의 충치를 의미합니다.",
        "patient_explanation": "치아 표면이나 안쪽이 단단함을 잃고 손상된 부위예요. 보통 진행 전이라면 메우는 치료로 마무리됩니다.",
        "patient_next_step": "치과에서 정밀 검진 후 레진 수복 등 보존 치료를 받는 것이 일반적입니다.",
        "color": "#ff7a59",
        "treatment": "레진 수복 치료",
        "cost_min": 200000,
        "cost_max": 200000,
    },
    {
        "name": "deep_caries",
        "label": "Deep Caries",
        "label_ko": "심부 충치",
        "description": "신경 치료 가능성을 의심할 수 있는 깊은 충치입니다.",
        "patient_explanation": "충치가 깊게 진행되어 신경 가까이까지 도달했을 가능성이 있는 부위예요.",
        "patient_next_step": "신경 치료 여부와 크라운 필요성을 치과 전문의와 상담해 보세요.",
        "color": "#ffb347",
        "treatment": "신경 치료 및 크라운 상담",
        "cost_min": 200000,
        "cost_max": 200000,
    },
    {
        "name": "caries_family",
        "label": "Caries Family",
        "label_ko": "충치 계열 병변",
        "description": "충치 계열 병변으로 검출됐지만 depth 세분화는 아직 확정되지 않은 상태입니다.",
        "patient_explanation": "충치 계열로 의심되는 부위예요. 얕은 충치인지 깊은 충치인지는 추가 확인이 필요합니다.",
        "patient_next_step": "치과에서 추가 검사를 통해 얕은 충치인지 깊은 충치인지 확인이 권장됩니다.",
        "color": "#ffd166",
        "treatment": "정밀 검사 후 수복 치료 결정",
        "cost_min": 200000,
        "cost_max": 200000,
    },
    {
        "name": "periapical_lesion",
        "label": "Periapical Lesion",
        "label_ko": "치근단 병소",
        "description": "치근단 주변 염증성 병소 의심 영역입니다.",
        "patient_explanation": "치아 뿌리 끝 주변에 염증이 의심되는 부위예요. 신경이 손상되었을 가능성이 있습니다.",
        "patient_next_step": "근관(신경) 치료 또는 재근관 치료가 필요한지 치과에서 확인해 보세요.",
        "color": "#33c7b5",
        "treatment": "근관 치료 또는 재근관 치료",
        "cost_min": 200000,
        "cost_max": 200000,
    },
    {
        "name": "impacted_tooth",
        "label": "Impacted Tooth",
        "label_ko": "매복치",
        "description": "매복치처럼 정상 맹출이 어려운 치아를 뜻합니다.",
        "patient_explanation": "잇몸이나 뼈 속에 머물러 정상적으로 나오지 못한 치아예요. 사랑니인 경우가 많습니다.",
        "patient_next_step": "발치 또는 외과적 처치 필요 여부를 치과 전문의와 상담해 보세요.",
        "color": "#4d8dff",
        "treatment": "발치 또는 외과적 처치 상담",
        "cost_min": 200000,
        "cost_max": 200000,
    },
]


def dashboard(request):
    context = {
        "class_info": _CLASS_INFO,
    }
    return render(request, "classifier/dashboard.html", context)


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
        # Model forward only (weights are loaded lazily on first request in Predictor.__init__).
        result["inference_ms"] = round((t1 - t_after_model) * 1000, 1)
        result["server_total_ms"] = round((t1 - t0) * 1000, 1)
        result["request_id"] = str(uuid.uuid4())
        result["weights_path"] = predictor.weights_path
        result["predict_conf"] = predictor.conf
        result["predict_imgsz"] = predictor.imgsz
        return _json_no_cache(result)
    except Exception as e:
        return _json_no_cache({"error": str(e)}, status=500)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
