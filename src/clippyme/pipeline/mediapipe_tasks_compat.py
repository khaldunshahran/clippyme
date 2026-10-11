"""Local-model Tasks adapters for MediaPipe wheels without ``solutions``.

IMAGE mode is intentional: the caller supplies independently cropped faces,
not a monotonically timed video stream. No model downloads occur at runtime.
"""
import os
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def bind_model_identity(config_digest):
    """Bind provisioned model bytes, not installation paths, to plan reuse.

    Unconfigured installations keep the accepted legacy identity. Models are
    provisioned before starting the worker and immutable for its lifetime.
    """
    models = {}
    for key in ('CLIPPYME_FACE_DETECTOR_MODEL',
                'CLIPPYME_FACE_LANDMARKER_MODEL'):
        value = os.environ.get(key, '').strip()
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            raise RuntimeError(f'{key}: configured local model is missing')
        models[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not models:
        return config_digest
    payload = {'adapter': 'mediapipe-tasks-image-v1',
               'config': config_digest, 'models': models}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class TasksFaceAdapter:
    def __init__(self, mp, task, *, landmarks=False):
        self.mp = mp
        self.task = task
        self.landmarks = landmarks

    def process(self, rgb):
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.task.detect(image)
        if self.landmarks:
            return SimpleNamespace(multi_face_landmarks=[
                SimpleNamespace(landmark=face)
                for face in result.face_landmarks
            ])
        height, width = rgb.shape[:2]
        detections = []
        for detection in result.detections:
            box = detection.bounding_box
            # Clip before returning the legacy normalized bounding box contract.
            x0 = max(0, min(width, box.origin_x))
            y0 = max(0, min(height, box.origin_y))
            x1 = max(0, min(width, box.origin_x + box.width))
            y1 = max(0, min(height, box.origin_y + box.height))
            if x1 <= x0 or y1 <= y0:
                continue
            bbox = SimpleNamespace(xmin=x0 / width, ymin=y0 / height,
                                   width=(x1 - x0) / width,
                                   height=(y1 - y0) / height)
            detections.append(SimpleNamespace(location_data=SimpleNamespace(
                relative_bounding_box=bbox)))
        return SimpleNamespace(detections=detections)

    def close(self):
        self.task.close()


def create_tasks_adapter(mp, *, landmarks=False):
    """Use an explicitly provisioned local model; retain legacy fallback if unset."""
    key = ('CLIPPYME_FACE_LANDMARKER_MODEL' if landmarks
           else 'CLIPPYME_FACE_DETECTOR_MODEL')
    value = os.environ.get(key, '').strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f'{key}: configured local model is missing')
    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions(model_asset_path=str(path.resolve()))
    if landmarks:
        options = vision.FaceLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            num_faces=1, min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False)
        task = vision.FaceLandmarker.create_from_options(options)
    else:
        options = vision.FaceDetectorOptions(
            base_options=base, running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5)
        task = vision.FaceDetector.create_from_options(options)
    return TasksFaceAdapter(mp, task, landmarks=landmarks)
