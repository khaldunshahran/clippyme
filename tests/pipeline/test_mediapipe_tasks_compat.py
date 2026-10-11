"""Tasks result compatibility without importing CV/ML libraries."""
from types import SimpleNamespace as NS
import pytest
from clippyme.pipeline.mediapipe_tasks_compat import (
    TasksFaceAdapter, create_tasks_adapter,
    bind_model_identity,
)


def test_model_identity_preserves_unconfigured_legacy_digest(monkeypatch):
    monkeypatch.delenv('CLIPPYME_FACE_DETECTOR_MODEL', raising=False)
    monkeypatch.delenv('CLIPPYME_FACE_LANDMARKER_MODEL', raising=False)
    assert bind_model_identity('accepted') == 'accepted'


def test_model_bytes_and_configuration_invalidate_cached_plan(monkeypatch, tmp_path):
    monkeypatch.delenv('CLIPPYME_FACE_LANDMARKER_MODEL', raising=False)
    model = tmp_path / 'detector.task'
    model.write_bytes(b'first-model')
    monkeypatch.setenv('CLIPPYME_FACE_DETECTOR_MODEL', str(model))
    first = bind_model_identity('config-a')
    identical = tmp_path / 'moved.task'
    identical.write_bytes(model.read_bytes())
    monkeypatch.setenv('CLIPPYME_FACE_DETECTOR_MODEL', str(identical))
    assert bind_model_identity('config-a') == first
    assert bind_model_identity('config-b') != first
    identical.write_bytes(b'replacement-model')
    assert bind_model_identity('config-a') != first
    monkeypatch.setenv('CLIPPYME_FACE_LANDMARKER_MODEL', str(model))
    assert bind_model_identity('config-a') != first


class RGB:
    shape = (100, 200, 3)


def _mp():
    return NS(Image=lambda **kw: kw, ImageFormat=NS(SRGB='rgb'))


def test_tasks_boxes_are_normalized_clipped_and_degenerate_dropped():
    boxes = [NS(origin_x=-20, origin_y=10, width=80, height=40),
             NS(origin_x=210, origin_y=0, width=5, height=5)]
    task = NS(detect=lambda im: NS(detections=[
        NS(bounding_box=b) for b in boxes]))
    result = TasksFaceAdapter(_mp(), task).process(RGB())
    assert len(result.detections) == 1
    box = result.detections[0].location_data.relative_bounding_box
    assert (box.xmin, box.ymin, box.width, box.height) == (0, .1, .3, .4)


def test_landmark_indices_and_empty_result_are_preserved():
    points = [NS(x=i / 500, y=i / 1000) for i in range(478)]
    task = NS(detect=lambda im: NS(face_landmarks=[points]))
    adapter = TasksFaceAdapter(_mp(), task, landmarks=True)
    assert adapter.process(RGB()).multi_face_landmarks[0].landmark[14] is points[14]
    task.detect = lambda im: NS(face_landmarks=[])
    assert adapter.process(RGB()).multi_face_landmarks == []


def test_unconfigured_model_retains_existing_fallback(monkeypatch):
    monkeypatch.delenv('CLIPPYME_FACE_DETECTOR_MODEL', raising=False)
    assert create_tasks_adapter(_mp()) is None


def test_missing_configured_model_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv('CLIPPYME_FACE_DETECTOR_MODEL', str(tmp_path/'absent'))
    with pytest.raises(RuntimeError, match='configured local model is missing'):
        create_tasks_adapter(_mp())


@pytest.mark.parametrize('landmarks', [False, True])
def test_real_factory_contract_is_image_mode_local_only(monkeypatch, tmp_path, landmarks):
    model = tmp_path/'model'; model.write_bytes(b'fixture')
    key = ('CLIPPYME_FACE_LANDMARKER_MODEL' if landmarks
           else 'CLIPPYME_FACE_DETECTOR_MODEL')
    monkeypatch.setenv(key, str(model))
    captured = []
    task = NS(close=lambda: captured.append('closed'))
    ctor = NS(create_from_options=lambda options: captured.append(options) or task)
    mp = _mp()
    mp.tasks = NS(BaseOptions=lambda **kw: NS(**kw), vision=NS(
        RunningMode=NS(IMAGE='IMAGE'), FaceDetector=ctor, FaceLandmarker=ctor,
        FaceDetectorOptions=lambda **kw: NS(**kw),
        FaceLandmarkerOptions=lambda **kw: NS(**kw)))
    adapter = create_tasks_adapter(mp, landmarks=landmarks)
    assert captured[0].running_mode == 'IMAGE'
    assert captured[0].base_options.model_asset_path == str(model.resolve())
    if landmarks:
        assert captured[0].num_faces == 1
        assert captured[0].output_face_blendshapes is False
    adapter.close(); assert captured[-1] == 'closed'
