import os
from pathlib import Path
from typing import Optional

from PIL import Image

from src.severity.inference import SeverityPredictor

# Repo root (…/dental), independent of process cwd when Django is started elsewhere.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SERVE_DETECTION_WEIGHTS = _PROJECT_ROOT / "artifacts/detection/serve/best.pt"
_SERVE_PERIODONTAL_WEIGHTS = _PROJECT_ROOT / "artifacts/detection/serve/periodontal_best.pt"
_SERVE_PERIAPICAL_FOLLOWUP_WEIGHTS = _PROJECT_ROOT / "artifacts/severity/serve/periapical_followup/best.pt"
_SERVE_BL_SEVERITY_WEIGHTS = _PROJECT_ROOT / "artifacts/severity/serve/bone_loss/best.pt"
_SERVE_FI_SEVERITY_WEIGHTS = _PROJECT_ROOT / "artifacts/severity/serve/furcation_involvement/best.pt"

API_CLASS_NAMES = [
    "caries",
    "periapical_lesion",
    "impacted_tooth",
    "bone_loss",
    "furcation_involvement",
    "retained_root",
]
API_CLASS_TO_ID = {name: idx for idx, name in enumerate(API_CLASS_NAMES)}


def _resolve_required_weights(explicit: Optional[str]) -> Path:
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


def _resolve_optional_weights(explicit: Optional[str], env_var_name: str, default_path: Path) -> Optional[Path]:
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


def _normalize_served_class_name(model_class_name: str) -> str:
    if model_class_name in {"caries", "deep_caries", "caries_family"}:
        return "caries"
    return model_class_name


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
        checkpoint_path: Optional[str] = None,
    ):
        ckpt_file = _resolve_required_weights(checkpoint_path)
        self.weights_path = str(ckpt_file)

        from ultralytics import YOLO

        self.model = YOLO(self.weights_path)
        self.model_class_names = _normalize_class_names(getattr(self.model, "names", []))
        # Default to CPU for the web service because the target machine has only 2GB VRAM.
        self.device = os.getenv("DENTAL_PREDICT_DEVICE", "cpu")
        self.conf = float(os.getenv("DENTAL_PREDICT_CONF", "0.1"))
        # Periodontal detector runs at a higher operating point than the main detector.
        # On the background-augmented model the F1-optimal conf is ~0.4 (vs ~0.1 for the
        # over-detecting positives-only model), which sharply cuts false positives on
        # healthy teeth. Kept as a separate knob so the main lesion detector stays at 0.1.
        self.periodontal_conf = float(os.getenv("DENTAL_PERIODONTAL_PREDICT_CONF", "0.4"))
        self.imgsz = int(os.getenv("DENTAL_PREDICT_IMGSZ", "512"))

        self.severity_device = os.getenv("DENTAL_SEVERITY_DEVICE", self.device)
        self.severity_margin = float(os.getenv("DENTAL_SEVERITY_CROP_MARGIN", "0.15"))
        self.periapical_followup_conf = float(os.getenv("DENTAL_PERIAPICAL_FOLLOWUP_CONF", "0.75"))

        # Optional periapical follow-up model. This keeps the served lesion class as
        # periapical_lesion and only attaches downstream metadata for treatment routing.
        periapical_ckpt = _resolve_optional_weights(
            None,
            "DENTAL_PERIAPICAL_FOLLOWUP_WEIGHTS",
            _SERVE_PERIAPICAL_FOLLOWUP_WEIGHTS,
        )
        self.periapical_followup_weights_path = (
            str(periapical_ckpt) if periapical_ckpt is not None else None
        )
        self.periapical_followup_model = (
            SeverityPredictor(periapical_ckpt, device=self.severity_device)
            if periapical_ckpt is not None
            else None
        )

        # Periodontal 2-class YOLO detector
        periodontal_ckpt = _resolve_optional_weights(None, "DENTAL_PERIODONTAL_WEIGHTS", _SERVE_PERIODONTAL_WEIGHTS)
        self.periodontal_model = None
        self.periodontal_class_names = []
        if periodontal_ckpt is not None:
            self.periodontal_model = YOLO(str(periodontal_ckpt))
            self.periodontal_class_names = _normalize_class_names(
                getattr(self.periodontal_model, "names", [])
            )

        # Periodontal severity models
        bl_ckpt = _resolve_optional_weights(None, "DENTAL_BL_SEVERITY_WEIGHTS", _SERVE_BL_SEVERITY_WEIGHTS)
        self.bl_severity_model = (
            SeverityPredictor(bl_ckpt, device=self.severity_device) if bl_ckpt is not None else None
        )

        fi_ckpt = _resolve_optional_weights(None, "DENTAL_FI_SEVERITY_WEIGHTS", _SERVE_FI_SEVERITY_WEIGHTS)
        self.fi_severity_model = (
            SeverityPredictor(fi_ckpt, device=self.severity_device) if fi_ckpt is not None else None
        )

    def _predict_detection(self, image_path: str, model, conf: Optional[float] = None):
        conf_threshold = self.conf if conf is None else conf
        try:
            return model.predict(
                image_path,
                conf=conf_threshold,
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

            return model.predict(
                image_path,
                conf=conf_threshold,
                imgsz=self.imgsz,
                verbose=False,
                device="cpu",
            )

    def _annotate_periapical_followup(self, image: Image.Image, detection: dict) -> dict:
        if self.periapical_followup_model is None:
            return detection
        if detection["class_name"] != "periapical_lesion":
            return detection

        followup = self.periapical_followup_model.predict_pil(
            _crop_bbox(image, detection["bbox_xyxy"], self.severity_margin)
        )
        detection["followup_class_name"] = followup["class_name"]
        detection["followup_confidence"] = followup["confidence"]
        detection["followup_probabilities"] = followup["probabilities"]
        detection["followup_applied"] = followup["confidence"] >= self.periapical_followup_conf
        return detection

    def _add_periodontal_severity(self, image: Image.Image, detection: dict) -> dict:
        class_name = detection["class_name"]
        severity_model = None
        if class_name == "bone_loss":
            severity_model = self.bl_severity_model
        elif class_name == "furcation_involvement":
            severity_model = self.fi_severity_model

        if severity_model is None:
            return detection

        severity = severity_model.predict_pil(
            _crop_bbox(image, detection["bbox_xyxy"], self.severity_margin)
        )
        detection["severity_class_name"] = severity["class_name"]
        detection["severity_confidence"] = severity["confidence"]
        detection["severity_probabilities"] = severity["probabilities"]
        # Periodontal severity is always resolved as a discrete class.
        detection["severity_applied"] = True
        return detection

    def _boxes_to_detections(self, results, class_names: list[str]) -> list[dict]:
        detections = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            model_class_id = int(box.cls.item())
            model_class_name = (
                class_names[model_class_id]
                if model_class_id < len(class_names)
                else str(model_class_id)
            )
            served_class_name = _normalize_served_class_name(model_class_name)
            detections.append({
                "class_id": API_CLASS_TO_ID.get(served_class_name, -1),
                "model_class_id": model_class_id,
                "class_name": served_class_name,
                "model_class_name": model_class_name,
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
            })
        return detections

    def predict_file(self, image_path: str):
        image = Image.open(image_path).convert("RGB")

        # Main detector (caries, periapical, impacted, ...)
        main_results = self._predict_detection(image_path, self.model)
        detections = self._boxes_to_detections(main_results, self.model_class_names)
        detections = [self._annotate_periapical_followup(image, d) for d in detections]

        # Periodontal detector (bone_loss, furcation_involvement)
        if self.periodontal_model is not None:
            perio_results = self._predict_detection(
                image_path, self.periodontal_model, conf=self.periodontal_conf
            )
            perio_detections = self._boxes_to_detections(perio_results, self.periodontal_class_names)
            perio_detections = [self._add_periodontal_severity(image, d) for d in perio_detections]
            detections.extend(perio_detections)

        return {
            "detections": detections,
            "periapical_followup_enabled": self.periapical_followup_model is not None,
            "periodontal_severity_enabled": self.bl_severity_model is not None or self.fi_severity_model is not None,
            "periapical_followup_weights_path": self.periapical_followup_weights_path,
            "detection_model_names": self.model_class_names,
            "periodontal_model_names": self.periodontal_class_names,
        }
