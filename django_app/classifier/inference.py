import os
from pathlib import Path

from PIL import Image

from src.severity.inference import SeverityPredictor

# Repo root (…/dental), independent of process cwd when Django is started elsewhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVE_DETECTION_WEIGHTS = _PROJECT_ROOT / "artifacts/detection/serve/best.pt"
_SERVE_SEVERITY_WEIGHTS = _PROJECT_ROOT / "artifacts/severity/serve/best.pt"

API_CLASS_NAMES = [
    "caries",
    "deep_caries",
    "periapical_lesion",
    "impacted_tooth",
    "caries_family",
]
API_CLASS_TO_ID = {name: idx for idx, name in enumerate(API_CLASS_NAMES)}
CARIES_REFINEMENT_SOURCE_NAMES = {"caries_family", "caries", "deep_caries"}


def _resolve_required_weights(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.is_file():
            return p

    env_path = os.getenv("DENTAL_YOLO_WEIGHTS") or os.getenv("DENTAL_WEIGHTS_PATH")
    if env_path:
        p = Path(env_path)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if p.is_file():
            return p

    if _SERVE_DETECTION_WEIGHTS.is_file():
        return _SERVE_DETECTION_WEIGHTS

    raise FileNotFoundError(
        "No deployment YOLO detection weights found. Copy the selected model to "
        f"{_SERVE_DETECTION_WEIGHTS} or set DENTAL_YOLO_WEIGHTS."
    )


def _resolve_optional_weights(explicit: str | None, env_var_name: str, default_path: Path) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    env_path = os.getenv(env_var_name)
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(default_path)
    for path in candidates:
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        if path.is_file():
            return path
    return None


def _normalize_class_names(raw_names) -> list[str]:
    if isinstance(raw_names, dict):
        return [raw_names[idx] for idx in sorted(raw_names)]
    return list(raw_names)


def _crop_bbox(image: Image.Image, bbox_xyxy: list[float], margin_ratio: float) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)
    margin_x = box_w * margin_ratio
    margin_y = box_h * margin_ratio
    left = max(0, int(round(x1 - margin_x)))
    top = max(0, int(round(y1 - margin_y)))
    right = min(image.width, int(round(x2 + margin_x)))
    bottom = min(image.height, int(round(y2 + margin_y)))
    return image.crop((left, top, right, bottom))


class Predictor:
    def __init__(
        self,
        checkpoint_path: str | None = None,
        severity_checkpoint_path: str | None = None,
    ):
        ckpt_file = _resolve_required_weights(checkpoint_path)
        self.weights_path = str(ckpt_file)

        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)
        self.model_class_names = _normalize_class_names(getattr(self.model, "names", []))
        # Default to CPU for the web service because the target machine has only 2GB VRAM.
        self.device = os.getenv("DENTAL_PREDICT_DEVICE", "cpu")
        self.conf = float(os.getenv("DENTAL_PREDICT_CONF", "0.1"))
        self.imgsz = int(os.getenv("DENTAL_PREDICT_IMGSZ", "512"))

        self.severity_device = os.getenv("DENTAL_SEVERITY_DEVICE", self.device)
        self.severity_conf = float(os.getenv("DENTAL_SEVERITY_CONF", "0.75"))
        self.severity_margin = float(os.getenv("DENTAL_SEVERITY_CROP_MARGIN", "0.15"))
        self.refine_flat_caries = os.getenv("DENTAL_REFINE_FLAT_CARIES", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        severity_ckpt = _resolve_optional_weights(
            severity_checkpoint_path,
            "DENTAL_SEVERITY_WEIGHTS",
            _SERVE_SEVERITY_WEIGHTS,
        )
        self.severity_weights_path = str(severity_ckpt) if severity_ckpt is not None else None
        self.severity_model = (
            SeverityPredictor(severity_ckpt, device=self.severity_device) if severity_ckpt is not None else None
        )

    def _predict_detection(self, image_path: str):
        try:
            return self.model.predict(
                image_path,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            return self.model.predict(
                image_path,
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device="cpu",
            )

    def _should_refine_detection(self, detection: dict) -> bool:
        if self.severity_model is None:
            return False

        class_name = detection["class_name"]
        if class_name == "caries_family":
            return True
        if self.refine_flat_caries and class_name in {"caries", "deep_caries"}:
            return True
        return False

    def _refine_caries_family(self, image: Image.Image, detection: dict) -> dict:
        if not self._should_refine_detection(detection):
            return detection

        detector_class_name = detection["class_name"]
        detector_class_id = detection["class_id"]
        severity = self.severity_model.predict_pil(
            _crop_bbox(image, detection["bbox_xyxy"], self.severity_margin)
        )
        detection["coarse_class_name"] = "caries_family"
        detection["coarse_class_id"] = API_CLASS_TO_ID["caries_family"]
        detection["detector_class_name"] = detector_class_name
        detection["detector_class_id"] = detector_class_id
        detection["severity_confidence"] = severity["confidence"]
        detection["severity_probabilities"] = severity["probabilities"]

        if severity["confidence"] >= self.severity_conf:
            detection["class_name"] = severity["class_name"]
            detection["class_id"] = API_CLASS_TO_ID[severity["class_name"]]
        return detection

    def predict_file(self, image_path: str):
        results = self._predict_detection(image_path)
        if not results:
            return {"detections": [], "severity_enabled": self.severity_model is not None}

        image = Image.open(image_path).convert("RGB")
        detections = []
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                model_class_id = int(box.cls.item())
                model_class_name = (
                    self.model_class_names[model_class_id]
                    if model_class_id < len(self.model_class_names)
                    else str(model_class_id)
                )
                detection = {
                    "class_id": API_CLASS_TO_ID.get(model_class_name, -1),
                    "model_class_id": model_class_id,
                    "class_name": model_class_name,
                    "model_class_name": model_class_name,
                    "confidence": float(box.conf.item()),
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
                detections.append(self._refine_caries_family(image, detection))

        return {
            "detections": detections,
            "severity_enabled": self.severity_model is not None,
            "severity_weights_path": self.severity_weights_path,
            "detection_model_names": self.model_class_names,
            "refine_flat_caries": self.refine_flat_caries,
        }
