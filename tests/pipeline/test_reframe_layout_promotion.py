"""Task 02: synthetic integration - only the evidenced scene promotes.

Three-scene source: faceless moving object | screen+presenter | faceless object.
Injected semantic/detector evidence (no live APIs, no real recognition).
Verifies the production wiring (maybe_promote_screencast) plus actual
rendered composition via create_screencast_frame with colored landmarks.
"""
import pytest

cv2 = pytest.importorskip("cv2", reason="CV runtime unavailable")
import numpy as np

from clippyme.pipeline.layouts.screencast_layout import maybe_promote_screencast
from clippyme.pipeline.layouts import screencast_layout as sl


W, H = 640, 480
FPS = 30


def _scene_frame(kind, t):
    """Synthetic frame with colored landmarks."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if kind == "object":
        # Faceless moving object: red square sweeping horizontally
        x = int(50 + 400 * (t % 30) / 30)
        cv2.rectangle(img, (x, 200), (x + 80, 280), (0, 0, 255), -1)
    elif kind == "screen":
        # Screen region: blue checkerboard (top 2/3)
        for yy in range(0, 320, 40):
            for xx in range(0, W, 40):
                if (xx // 40 + yy // 40) % 2 == 0:
                    cv2.rectangle(img, (xx, yy), (xx + 40, yy + 40),
                                  (255, 0, 0), -1)
        # Presenter: stable green circle (bottom area)
        cv2.circle(img, (W // 2, 400), 50, (0, 255, 0), -1)
    else:  # "object2"
        # Faceless object: yellow triangle
        pts = np.array([[320, 150], [250, 300], [390, 300]], np.int32)
        cv2.fillPoly(img, [pts], (0, 255, 255))
    return img


def _build_video(path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    kinds = ["object"] * 30 + ["screen"] * 30 + ["object2"] * 30
    for t, kind in enumerate(kinds):
        vw.write(_scene_frame(kind, t))
    vw.release()


@pytest.fixture()
def three_scene_video(tmp_path):
    p = tmp_path / "three_scene.mp4"
    _build_video(p)
    assert p.exists()
    return str(p)


def _inject_evidence(monkeypatch):
    """Scene 1 = screencast w/ stable presenter; others = none."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")

    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        if scene_id == 1:
            return {"layout": "screencast", "confidence": 0.93,
                    "screen_evidence": True, "reason": "checkerboard + presenter",
                    "scene_id": 1, "samples": 5}
        return {"layout": "none", "confidence": 0.96,
                "screen_evidence": False, "reason": "no screen",
                "scene_id": scene_id, "samples": 5}

    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    # Presenter at the green circle: (320, 400)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {1: (320.0, 400.0)})


def test_only_middle_scene_promotes(three_scene_video, monkeypatch):
    _inject_evidence(monkeypatch)
    scenes = [(0, 30), (30, 60), (60, 90)]
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    promoted = maybe_promote_screencast(
        three_scene_video, scenes, strategies, meta)
    assert promoted == [1]
    assert strategies == ["GENERAL", "SCREENCAST", "GENERAL"]
    assert meta[1] == (320.0, 400.0)


def test_rendered_composition_has_content_and_pane(three_scene_video, monkeypatch):
    """Rendered SCREENCAST frame: blue checkerboard on top, green presenter below."""
    _inject_evidence(monkeypatch)
    scenes = [(0, 30), (30, 60), (60, 90)]
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    maybe_promote_screencast(three_scene_video, scenes, strategies, meta)

    cap = cv2.VideoCapture(three_scene_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 45)  # middle of scene 1
    ok, frame = cap.read()
    cap.release()
    assert ok

    out_w, out_h = 360, 640
    rendered = sl.create_screencast_frame(frame, out_w, out_h, meta[1])
    assert rendered.shape == (out_h, out_w, 3)

    # Top band should contain blue checkerboard content
    content_h, speaker_h = sl.content_bands(W, H, out_w, out_h)
    top = rendered[:content_h // 2]
    blue_pixels = np.sum((top[:, :, 0] > 150) &
                         (top[:, :, 1] < 100) & (top[:, :, 2] < 100))
    assert blue_pixels > 1000, "top content band missing blue screen landmark"

    # Bottom band should contain the green presenter (framed on injected center)
    bottom = rendered[content_h:]
    green_pixels = np.sum((bottom[:, :, 1] > 150) &
                          (bottom[:, :, 0] < 100) & (bottom[:, :, 2] < 100))
    assert green_pixels > 500, "speaker band missing green presenter landmark"


def test_general_scenes_do_not_get_pane(three_scene_video, monkeypatch):
    """Faceless scenes keep GENERAL: no fabricated duplicated pane."""
    _inject_evidence(monkeypatch)
    scenes = [(0, 30), (30, 60), (60, 90)]
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    maybe_promote_screencast(three_scene_video, scenes, strategies, meta)
    assert 0 not in meta and 2 not in meta


@pytest.mark.parametrize("comfort", ["1", "0"])
def test_both_paths_consume_same_decisions(three_scene_video, monkeypatch, comfort):
    """Streaming and comfort paths see identical promotion decisions."""
    monkeypatch.setenv("REFRAME_COMFORT", comfort)
    _inject_evidence(monkeypatch)
    scenes = [(0, 30), (30, 60), (60, 90)]
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    promoted = maybe_promote_screencast(
        three_scene_video, scenes, strategies, meta)
    # The decision layer runs before either path; both must agree.
    assert promoted == [1]
    assert strategies[1] == "SCREENCAST"
