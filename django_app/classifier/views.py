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
        "description": "법랑질이나 상아질의 충치를 의미합니다.",
        "color": "#ff7a59",
    },
    {
        "name": "deep_caries",
        "label": "Deep Caries",
        "description": "신경 치료 가능성을 의심할 수 있는 깊은 충치입니다.",
        "color": "#ffb347",
    },
    {
        "name": "caries_family",
        "label": "Caries Family",
        "description": "충치 계열 병변으로 검출됐지만 depth 세분화는 아직 확정되지 않은 상태입니다.",
        "color": "#ffd166",
    },
    {
        "name": "periapical_lesion",
        "label": "Periapical Lesion",
        "description": "치근단 주변 염증성 병소 의심 영역입니다.",
        "color": "#33c7b5",
    },
    {
        "name": "impacted_tooth",
        "label": "Impacted Tooth",
        "description": "매복치처럼 정상 맹출이 어려운 치아를 뜻합니다.",
        "color": "#4d8dff",
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
