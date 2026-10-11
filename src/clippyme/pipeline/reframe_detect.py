"""ML-bound detection layer for the reframe pipeline (cv2 + YOLO + MediaPipe).

Owns the lazy model singletons (YOLOv8n, MediaPipe FaceDetection/FaceMesh) and
the per-frame detectors. Split out of ``reframe.py`` so the pure tracking
classes (``reframe_track``) stay host-importable; ``reframe.py`` re-exports
these names for back-compat.
"""
import cv2
import mediapipe as mp
from ultralytics import YOLO

from clippyme.pipeline.hardware import DEVICE
from clippyme.pipeline.mediapipe_tasks_compat import create_tasks_adapter

_yolo_model = None

def _get_yolo_model():
    """Lazy-load YOLOv8n on first body-detection call."""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO('yolov8n.pt')
        _yolo_model.to(DEVICE)
    return _yolo_model

_has_solutions = hasattr(mp, "solutions")
mp_face_detection = getattr(mp.solutions, "face_detection", None) if _has_solutions else None
mp_face_mesh = getattr(mp.solutions, "face_mesh", None) if _has_solutions else None

_face_detection = None
_face_mesh = None


def _get_face_detection():
    """Lazy-init MediaPipe FaceDetection on first use (avoids ~300ms TFLite
    load at import time on every subprocess, incl. --reframe-only switches)."""
    global _face_detection
    if _face_detection is None and mp_face_detection is not None:
        _face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    elif _face_detection is None:
        _face_detection = create_tasks_adapter(mp)
    return _face_detection


def _get_face_mesh():
    """Lazy-init MediaPipe FaceMesh on first use (see _get_face_detection)."""
    global _face_mesh
    if _face_mesh is None and mp_face_mesh is not None:
        _face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    elif _face_mesh is None:
        _face_mesh = create_tasks_adapter(mp, landmarks=True)
    return _face_mesh

# MediaPipe FaceMesh mouth landmark indices (inner lips + corners). Defined here
# because compute_mouth_aspect_ratio was extracted from main.py during the
# refactor; without these the MAR call raised NameError on every frame, silently
# disabling active-speaker selection (the exception was swallowed by the
# corrupt-frame guard → ~37% duplicated frames).
_MOUTH_TOP = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT = 78
_MOUTH_RIGHT = 308


def compute_mouth_aspect_ratio(frame_bgr, face_box) -> float | None:
    """
    Crops the face region and runs FaceMesh to extract the Mouth Aspect Ratio
    (vertical mouth opening / horizontal mouth width).

    Returns a normalized MAR in [0, ~1.5] or None if landmarks couldn't be
    extracted (e.g. profile view, occlusion). The absolute value matters less
    than its *variance over time* — a still mouth has near-zero variance, a
    talking mouth oscillates.
    """
    x, y, w, h = face_box
    H, W = frame_bgr.shape[:2]
    # Pad the crop a bit so FaceMesh has context
    pad = int(max(w, h) * 0.2)
    x1 = max(0, int(x - pad))
    y1 = max(0, int(y - pad))
    x2 = min(W, int(x + w + pad))
    y2 = min(H, int(y + h + pad))
    if x2 - x1 < 30 or y2 - y1 < 30:
        return None
    face_mesh = _get_face_mesh()
    if face_mesh is None:
        return None
    roi = frame_bgr[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = face_mesh.process(rgb)
    if not res or not getattr(res, "multi_face_landmarks", None):
        return None
    lm = res.multi_face_landmarks[0].landmark
    rh, rw = roi.shape[:2]
    top = (lm[_MOUTH_TOP].x * rw, lm[_MOUTH_TOP].y * rh)
    bot = (lm[_MOUTH_BOTTOM].x * rw, lm[_MOUTH_BOTTOM].y * rh)
    left = (lm[_MOUTH_LEFT].x * rw, lm[_MOUTH_LEFT].y * rh)
    right = (lm[_MOUTH_RIGHT].x * rw, lm[_MOUTH_RIGHT].y * rh)
    mouth_h = ((top[0] - bot[0]) ** 2 + (top[1] - bot[1]) ** 2) ** 0.5
    mouth_w = ((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2) ** 0.5
    if mouth_w < 1:
        return None
    return mouth_h / mouth_w


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Falls back to YOLO upper-body detection when FaceDetection is unavailable.
    """
    face_det = _get_face_detection()
    if face_det is not None:
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_det.process(rgb_frame)
        candidates = []
        if not results or not getattr(results, "detections", None):
            return []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * width)
            y = int(bboxC.ymin * height)
            w = int(bboxC.width * width)
            h = int(bboxC.height * height)
            candidates.append({
                'box': [x, y, w, h],
                'score': w * h  # Area as score
            })
        return candidates

    # Fallback: Detect person upper bodies using YOLO
    results = _get_yolo_model()(frame, verbose=False, classes=[0])
    candidates = []
    if results:
        for result in results:
            for box in getattr(result, "boxes", []):
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                w = x2 - x1
                h = y2 - y1
                head_h = max(20, int(h * 0.35))
                candidates.append({
                    'box': [x1, y1, w, head_h],
                    'score': w * head_h
                })
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    results = _get_yolo_model()(frame, verbose=False, classes=[0])  # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Return the full person bbox. SmoothedCameraman.update_target
                # applies the head-zone offset itself when is_person_box=True
                # (y_center = y + h*0.15). Previously we truncated here AND
                # there, which stacked the offsets and aimed above the head.
                best_box = [x1, y1, w, h]
                
    return best_box


# --- Task 14: action observation seam ---------------------------------------
# Normalized category families for action framing. The detector maps the
# provider's class names onto this fixed vocabulary; anything unmapped is
# dropped here so the pure observation contract only ever sees supported
# families (it additionally rejects anything outside the vocabulary).

ACTION_CATEGORY_FAMILIES = ("person", "animal", "vehicle", "object")

_ACTION_CLASS_TO_FAMILY = {
    # person
    "person": "person",
    # animals
    "dog": "animal", "cat": "animal", "horse": "animal", "bird": "animal",
    "cow": "animal", "sheep": "animal", "elephant": "animal",
    "bear": "animal", "zebra": "animal", "giraffe": "animal",
    # vehicles
    "car": "vehicle", "motorcycle": "vehicle", "bicycle": "vehicle",
    "bus": "vehicle", "truck": "vehicle", "boat": "vehicle",
    "airplane": "vehicle", "train": "vehicle",
    # commonly-demoed held objects / products / sports gear
    "sports ball": "object", "bottle": "object", "cup": "object",
    "wine glass": "object", "cell phone": "object", "laptop": "object",
    "book": "object", "handbag": "object", "backpack": "object",
    "suitcase": "object", "skis": "object", "snowboard": "object",
    "skateboard": "object", "surfboard": "object", "tennis racket": "object",
    "frisbee": "object", "kite": "object", "baseball bat": "object",
    "baseball glove": "object",
}


def detect_action_candidates(frame, min_confidence: float = 0.25):
    """Detect action-subject candidates for the Task 14 action planner.

    Reuses the existing lazily-loaded YOLOv8 singleton (no second network,
    no new dependency, no live provider call): one forward pass, all COCO
    classes, mapped onto the normalized ``ACTION_CATEGORY_FAMILIES``
    vocabulary. Returns a list of ``{"box": (x, y, w, h), "confidence":
    float, "category": family}`` with float pixel boxes.

    Raises ``RuntimeError`` when the model cannot be loaded (the caller
    treats this as detector-unavailable and keeps the accepted GENERAL
    fallback with an explicit reason). Unmapped classes are dropped, and
    detections below ``min_confidence`` are dropped — weak evidence must
    never authorize a tight crop.
    """
    try:
        model = _get_yolo_model()
    except Exception as exc:
        raise RuntimeError(f"yolo-unavailable: {exc}") from exc
    if model is None:
        raise RuntimeError("yolo-unavailable: model is None")
    results = model(frame, verbose=False)  # all classes, single inference
    names = getattr(model, "names", {}) or {}
    out = []
    for result in results:
        for box in getattr(result, "boxes", []):
            cls_name = str(names.get(int(box.cls[0]), "")).lower()
            family = _ACTION_CLASS_TO_FAMILY.get(cls_name)
            if family is None:
                continue
            try:
                conf = float(box.conf[0])
            except (IndexError, TypeError, ValueError):
                continue
            if not (0.0 <= conf <= 1.0) or conf < min_confidence:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            out.append({
                "box": (x1, y1, w, h),
                "confidence": conf,
                "category": family,
            })
    return out
