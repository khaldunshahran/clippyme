"""Task 02: host-safe policy/parser/provider/config tests.

All provider evidence is injected (no live APIs). All sampling is
plain-data via monkeypatched helpers.
"""
import json
import os

import pytest

from clippyme.pipeline.layouts import layout_classifier as lc


# --- parse_layout_response -------------------------------------------------

def test_parse_valid_screencast():
    text = json.dumps({
        "layout": "screencast",
        "confidence": 0.92,
        "screen_evidence": True,
        "why": "code editor visible, presenter in corner",
    })
    r = lc.parse_layout_response(text)
    assert r is not None
    assert r["layout"] == "screencast"
    assert r["confidence"] == 0.92
    assert r["screen_evidence"] is True


def test_parse_standard_maps_to_none():
    text = json.dumps({"layout": "standard", "confidence": 0.9, "why": "x"})
    r = lc.parse_layout_response(text)
    assert r is not None and r["layout"] == "none"


@pytest.mark.parametrize("conf", [float("nan"), float("inf"), -float("inf"),
                                   True, False, "0.9", None, [0.9]])
def test_parse_rejects_bad_confidence(conf):
    text = json.dumps({"layout": "screencast", "confidence": conf, "why": "x"})
    assert lc.parse_layout_response(text) is None


@pytest.mark.parametrize("conf", [-0.1, 1.1])
def test_parse_rejects_out_of_range_confidence(conf):
    text = json.dumps({"layout": "screencast", "confidence": conf, "why": "x"})
    assert lc.parse_layout_response(text) is None


def test_parse_rejects_invalid_layout():
    text = json.dumps({"layout": "panorama", "confidence": 0.9, "why": "x"})
    assert lc.parse_layout_response(text) is None


def test_parse_rejects_garbage():
    assert lc.parse_layout_response("not json") is None
    assert lc.parse_layout_response("") is None
    assert lc.parse_layout_response(None) is None
    assert lc.parse_layout_response(json.dumps([1, 2])) is None
    assert lc.parse_layout_response(json.dumps({"confidence": 0.9})) is None


def test_parse_missing_fields_rejected():
    assert lc.parse_layout_response(json.dumps({"layout": "none"})) is None


# --- classify_scene with injected provider ---------------------------------

def _enable(monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")


def _frames(n):
    return [b"fakejpeg%d" % i for i in range(n)]


def test_classify_scene_promotes_on_evidence(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    provider = lambda frames, prompt: json.dumps({
        "layout": "screencast", "confidence": 0.9,
        "screen_evidence": True, "why": "slides + presenter"})
    r = lc.classify_scene("v.mp4", 2, 100, 200, provider=provider)
    assert r is not None
    assert r["scene_id"] == 2
    assert r["layout"] == "screencast"
    assert r["samples"] == 5


def test_classify_scene_low_confidence_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    provider = lambda frames, prompt: json.dumps({
        "layout": "screencast", "confidence": 0.5,
        "screen_evidence": True, "why": "maybe"})
    assert lc.classify_scene("v.mp4", 0, 0, 90, provider=provider) is None


def test_classify_scene_too_few_samples_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(2))
    provider = lambda frames, prompt: json.dumps({
        "layout": "screencast", "confidence": 0.95,
        "screen_evidence": True, "why": "x"})
    assert lc.classify_scene("v.mp4", 0, 0, 90, provider=provider) is None


def test_classify_scene_provider_error_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    def boom(frames, prompt):
        raise RuntimeError("provider down")
    assert lc.classify_scene("v.mp4", 0, 0, 90, provider=boom) is None


def test_classify_scene_invalid_json_returns_none(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    provider = lambda frames, prompt: "{bad json"
    assert lc.classify_scene("v.mp4", 0, 0, 90, provider=provider) is None


def test_classify_scene_none_label_never_promotes(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    provider = lambda frames, prompt: json.dumps({
        "layout": "none", "confidence": 0.99, "why": "talking head"})
    r = lc.classify_scene("v.mp4", 0, 0, 90, provider=provider)
    assert r is not None and r["layout"] == "none"


# --- config behavior ---------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("Yes", True),
    ("0", False), ("false", False), ("", False), ("no", False),
])
def test_auto_layout_flag(monkeypatch, val, expected):
    monkeypatch.setenv("AUTO_LAYOUT", val)
    assert lc.is_auto_layout_enabled() is expected


def test_auto_layout_unset_is_disabled(monkeypatch):
    monkeypatch.delenv("AUTO_LAYOUT", raising=False)
    assert lc.is_auto_layout_enabled() is False


def test_classify_scene_disabled_flag_returns_none(monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "0")
    called = []
    def provider(frames, prompt):
        called.append(True)
        return json.dumps({"layout": "screencast", "confidence": 0.99,
                           "screen_evidence": True, "why": "x"})
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: _frames(5))
    assert lc.classify_scene("v.mp4", 0, 0, 90, provider=provider) is None
    assert called == []  # provider never invoked when disabled


# --- maybe_promote_screencast policy ------------------------------------------
# These need the reframe module (imports cv2). Skip if CV runtime unavailable.

import importlib


@pytest.fixture()
def rf():
    cv2 = importlib.util.find_spec("cv2")
    if cv2 is None:
        pytest.skip("CV runtime unavailable")
    from clippyme.pipeline.layouts.screencast_layout import maybe_promote_screencast as _mp
    return type("RF", (), {"maybe_promote_screencast": staticmethod(_mp)})()


def _scenes():
    # (start_frame, end_frame) plain tuples
    return [(0, 90), (90, 180), (180, 270)]


def _evidence(layout, conf=0.92, screen=True):
    return {"layout": layout, "confidence": conf,
            "screen_evidence": screen, "reason": "test",
            "scene_id": 0, "samples": 5}


def test_policy_no_promotion_when_disabled(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "")
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    promoted = rf.maybe_promote_screencast(
        "v.mp4", _scenes(), strategies, meta,
        provider=lambda f, p: json.dumps({}))
    assert promoted == []
    assert strategies == ["GENERAL"] * 3


def test_policy_promotes_only_evidence_plus_geometry(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    # Scene 1: screen + stable presenter. Others: no screen.
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        if scene_id == 1:
            return dict(_evidence("screencast"), scene_id=1)
        return dict(_evidence("none", conf=0.95, screen=False), scene_id=scene_id)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {1: (960.0, 540.0)})
    strategies = ["GENERAL", "GENERAL", "GENERAL"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", _scenes(), strategies, meta)
    assert promoted == [1]
    assert strategies == ["GENERAL", "SCREENCAST", "GENERAL"]
    assert meta[1] == (960.0, 540.0)


def test_policy_screen_without_presenter_keeps_general(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        return dict(_evidence("screencast"), scene_id=scene_id)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {})
    strategies = ["GENERAL"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", [(0, 90)], strategies, meta)
    assert promoted == []
    assert strategies == ["GENERAL"]
    assert meta == {}


def test_policy_track_with_evidence_promotes(rf, monkeypatch):
    # TRACK with screen + presenter evidence promotes (webcam scenes).
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        return dict(_evidence("screencast"), scene_id=scene_id)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {0: (1.0, 1.0)})
    strategies = ["TRACK"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", [(0, 90)], strategies, meta)
    assert promoted == [0]
    assert strategies == ["SCREENCAST"]


def test_policy_split_never_promotes(rf, monkeypatch):
    # Two-person SPLIT is never auto-promoted.
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        return dict(_evidence("screencast"), scene_id=scene_id)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {0: (1.0, 1.0), 1: (2.0, 2.0)})
    strategies = ["SPLIT", "OBJECT"]
    meta = {}
    promoted = rf.maybe_promote_screencast(
        "v.mp4", [(0, 90), (90, 180)], strategies, meta)
    assert promoted == []
    assert strategies == ["SPLIT", "OBJECT"]


def test_policy_track_without_evidence_keeps_track(rf, monkeypatch):
    # Negative/unknown TRACK scenes retain TRACK.
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        return None
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    strategies = ["TRACK"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", [(0, 90)], strategies, meta)
    assert promoted == []
    assert strategies == ["TRACK"]


def test_policy_unknown_evidence_keeps_previous(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        return None  # provider error / unreadable / low confidence
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    strategies = ["GENERAL", "GENERAL"]
    meta = {}
    promoted = rf.maybe_promote_screencast(
        "v.mp4", [(0, 90), (90, 180)], strategies, meta)
    assert promoted == []
    assert strategies == ["GENERAL", "GENERAL"]


# --- presenter geometry: no fallback ------------------------------------------

from clippyme.pipeline.layouts import screencast_layout as sl


def _need_cv2():
    import importlib.util
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("CV runtime unavailable")


def test_detect_presenter_geometry_no_faces_returns_empty(tmp_path, monkeypatch):
    _need_cv2()
    # Unreadable video -> {} (no center fallback)
    assert sl.detect_presenter_geometry(str(tmp_path / "nope.mp4"),
                                        [(0, 90)]) == {}


def test_detect_presenter_geometry_never_fabricates_center(monkeypatch):
    _need_cv2()
    # Even when faces exist, the legacy fallback path is not used here:
    # the function only returns entries for reliable detections.
    import inspect
    src = inspect.getsource(sl.detect_presenter_geometry)
    assert "orig_w/2" not in src and "orig_h/2" not in src


# --- strict evidence contract -----------------------------------------------

import json as _json

def test_evidence_false_cannot_become_positive():
    r = lc.parse_layout_response(_json.dumps({
        "layout": "screencast", "confidence": 0.99,
        "screen_evidence": False, "why": "talking head"}))
    assert r is None

def test_evidence_missing_reason_invalid():
    r = lc.parse_layout_response(_json.dumps({
        "layout": "screencast", "confidence": 0.99,
        "screen_evidence": True}))
    assert r is None

def test_evidence_none_layout_ok():
    r = lc.parse_layout_response(_json.dumps({
        "layout": "none", "confidence": 0.95, "why": "vlog"}))
    assert r is not None and r["screen_evidence"] is False


# --- budgets -----------------------------------------------------------------

def test_budget_max_requests_stops_early(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    calls = []
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        calls.append(scene_id)
        return dict(_evidence("screencast"), scene_id=scene_id)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {i: (1.0, 1.0) for i in range(10)})
    strategies = ["GENERAL"] * 10
    meta = {}
    promoted = rf.maybe_promote_screencast(
        "v.mp4", [(i*90, (i+1)*90) for i in range(10)],
        strategies, meta, max_requests=3)
    assert len(calls) == 3
    assert promoted == [0, 1, 2]

def test_budget_deadline_stops_early(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    ticks = [0.0]
    def fake_clock():
        ticks[0] += 100.0  # jump past deadline immediately
        return ticks[0]
    def fake_classify(*a, **k):
        raise AssertionError("should not be called after deadline")
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {0: (1.0, 1.0)})
    strategies = ["GENERAL"]
    meta = {}
    promoted = rf.maybe_promote_screencast(
        "v.mp4", [(0, 90)], strategies, meta,
        deadline_s=1.0, _clock=fake_clock)
    assert promoted == []

def test_no_provider_call_without_geometry(rf, monkeypatch):
    # Scenes lacking presenter geometry skip paid classification.
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    calls = []
    def fake_classify(*a, **k):
        calls.append(1)
        return None
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {})
    strategies = ["GENERAL", "GENERAL"]
    meta = {}
    rf.maybe_promote_screencast("v.mp4", [(0, 90), (90, 180)], strategies, meta)
    assert calls == []


# --- compatibility wrapper ----------------------------------------------------

def test_pick_and_apply_explicit_passthrough(monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "0")
    assert lc.pick_and_apply_layout("v.mp4", explicit_layout="screencast") == "screencast"
    assert lc.pick_and_apply_layout("v.mp4", explicit_layout="split") == "split"

def test_pick_and_apply_auto_disabled_returns_auto(monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "false")
    calls = []
    monkeypatch.setattr(lc, "classify_layout", lambda *a, **k: calls.append(1) or "screencast")
    assert lc.pick_and_apply_layout("v.mp4", explicit_layout="auto") == "auto"
    assert calls == []

def test_pick_and_apply_auto_unset_returns_auto(monkeypatch):
    monkeypatch.delenv("AUTO_LAYOUT", raising=False)
    calls = []
    monkeypatch.setattr(lc, "classify_layout", lambda *a, **k: calls.append(1) or "screencast")
    assert lc.pick_and_apply_layout("v.mp4", explicit_layout="auto") == "auto"
    assert calls == []


# --- inset/webcam support (durable regression) --------------------------------

def test_inset_label_with_evidence_promotes(rf, monkeypatch):
    """Documented inset (screen/webcam) promotes via SCREENCAST composition."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(video_path, scene_id, s_f, e_f, provider=None,
                      api_key=None, model_name=None, _budget=None):
        # Real parser route with injected raw JSON.
        import json as _j
        text = _j.dumps({"layout": "inset", "confidence": 0.97,
                         "screen_evidence": True,
                         "why": "software with stable webcam"})
        from clippyme.pipeline.layouts import layout_classifier as _lc
        parsed = _lc.parse_layout_response(text)
        parsed["scene_id"] = scene_id
        parsed["samples"] = 5
        return parsed
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.screencast_layout.detect_presenter_geometry",
        lambda *a, **k: {0: (320.0, 400.0)})
    strategies = ["TRACK"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", [(0, 90)], strategies, meta)
    assert promoted == [0]
    assert strategies == ["SCREENCAST"]


def test_inset_without_evidence_does_not_promote(rf, monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    def fake_classify(*a, **k):
        return None
    monkeypatch.setattr(
        "clippyme.pipeline.layouts.layout_classifier.classify_scene",
        fake_classify)
    strategies = ["TRACK"]
    meta = {}
    promoted = rf.maybe_promote_screencast("v.mp4", [(0, 90)], strategies, meta)
    assert promoted == []
    assert strategies == ["TRACK"]
