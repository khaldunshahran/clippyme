"""Task 02: durable budget regressions.

Fast failures, actual-call limits, supported SDK retry policy, subsecond
timeouts, geometry/sampling deadline exhaustion, provider gating,
zero/nonfinite inputs, subset ID preservation, client lifecycle.
No live APIs. Future clones must not depend on work/ probe files.
"""
import sys
import types

import pytest

from clippyme.pipeline.layouts import layout_classifier as lc
from clippyme.pipeline.layouts import screencast_layout as sl


def _budget(max_requests=8, deadline_s=120, clock=None):
    import time
    return sl._InvocationBudget(max_requests, deadline_s, clock or time.monotonic)


def _sdk(monkeypatch, state, fail=False):
    """Fake google-genai with supported v1.70.0 API shapes."""
    sdk = types.ModuleType("google.genai")
    sdk_types = types.ModuleType("google.genai.types")

    class HttpRetryOptions:
        def __init__(self, attempts=None):
            self.attempts = attempts

    class HttpOptions:
        def __init__(self, timeout=None, retry_options=None):
            self.timeout = timeout
            self.retry_options = retry_options

    sdk_types.HttpRetryOptions = HttpRetryOptions
    sdk_types.HttpOptions = HttpOptions
    sdk_types.Part = types.SimpleNamespace(from_bytes=lambda **kw: kw)
    sdk_types.GenerateContentConfig = lambda **kw: kw

    def generate(**kw):
        state["calls"] += 1
        if fail:
            raise RuntimeError("immediate failure")
        return types.SimpleNamespace(text="{}")

    def client(**kw):
        state["options"].append(kw["http_options"])
        return types.SimpleNamespace(
            models=types.SimpleNamespace(generate_content=generate))

    sdk.Client = client
    sdk.types = sdk_types
    google = types.ModuleType("google")
    google.genai = sdk
    aux = types.ModuleType("clippyme.pipeline.gemini_service")
    aux.get_auxiliary_gemini_model = lambda *a: "fake-model"
    for name, mod in [("google", google), ("google.genai", sdk),
                      ("google.genai.types", sdk_types),
                      ("clippyme.pipeline.gemini_service", aux)]:
        monkeypatch.setitem(sys.modules, name, mod)
    import time
    monkeypatch.setattr(time, "sleep",
                        lambda s: state.__setitem__("time", state["time"] + s))
    return sdk_types


def test_fast_failures_respect_request_allowance(monkeypatch):
    """max_requests=1 with immediate failures -> exactly 1 SDK call."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    monkeypatch.setattr(sl, "detect_presenter_geometry",
                        lambda *a, **k: {0: (320., 400.)})
    monkeypatch.setattr(lc, "sample_scene_frames", lambda *a, **k: [b"x"] * 3)
    state = {"time": 0., "calls": 0, "options": []}
    _sdk(monkeypatch, state, fail=True)
    sl.maybe_promote_screencast(
        "fake.mp4", [(0, 90)], ["GENERAL"], {},
api_key="fake", max_requests=1, deadline_s=120)
    assert state["calls"] == 1, f"expected 1 call, got {state['calls']}"


def test_sdk_retry_policy_explicitly_disabled(monkeypatch):
    """SDK receives HttpRetryOptions(attempts=1) via retry_options."""
    state = {"time": 0., "calls": 0, "options": []}
    _sdk(monkeypatch, state)
    budget = _budget(3, 120)
    lc._gemini_classify([b"x"] * 3, api_key="fake", _budget=budget)
    assert state["calls"] == 1
    opts = state["options"][0]
    assert opts.retry_options is not None
    assert opts.retry_options.attempts in (0, 1)


def test_subsecond_timeout_capped_by_remaining(monkeypatch):
    """100ms budget -> timeout <= 100ms, never floored to 1000ms."""
    state = {"time": 0., "calls": 0, "options": []}
    _sdk(monkeypatch, state)
    budget = _budget(1, 0.1)
    lc._gemini_classify([b"x"] * 3, api_key="fake", _budget=budget)
    if state["options"]:
        assert max(o.timeout for o in state["options"]) <= 100


def test_geometry_stops_after_deadline(monkeypatch):
    """Deadline exhaustion stops starting new sample reads."""
    import cv2
    import numpy as np
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    state = {"time": 0., "reads": 0}

    class Cap:
        def isOpened(self): return True
        def get(self, k): return 640 if k == cv2.CAP_PROP_FRAME_WIDTH else 480
        def set(self, *a): pass
        def read(self):
            state["reads"] += 1
            state["time"] += 0.25
            return True, np.full((480, 640, 3), 80, dtype=np.uint8)
        def release(self): pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *a: Cap())
    det = types.ModuleType("clippyme.pipeline.reframe_detect")
    det.detect_face_candidates = lambda frame: []
    monkeypatch.setitem(sys.modules, "clippyme.pipeline.reframe_detect", det)
    sl.maybe_promote_screencast(
        "fake", [(0, 30), (30, 60)], ["GENERAL"] * 2, {},
        deadline_s=1, _clock=lambda: state["time"])
    assert state["reads"] <= 4


def test_provider_not_started_after_sampling_exhausts_deadline(monkeypatch):
    """If sampling consumes the deadline, provider is never invoked."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    state = {"time": 0., "calls": 0}

    def samples(*a, **k):
        state["time"] = 2.0
        return [b"x"] * 3

    def provider(*a):
        state["calls"] += 1
        return "{}"

    monkeypatch.setattr(lc, "sample_scene_frames", samples)
    budget = _budget(1, 1, clock=lambda: state["time"])
    lc.classify_scene("fake", 0, 0, 30, provider=provider, _budget=budget)
    assert state["calls"] == 0


def test_zero_scenes_no_geometry(monkeypatch):
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    scanned = []
    monkeypatch.setattr(sl, "detect_presenter_geometry",
                        lambda p, s, *a, **k: scanned.extend(s) or {})
    sl.maybe_promote_screencast("fake", [(0, 30)], ["GENERAL"], {}, max_scenes=0)
    assert scanned == []


def test_nonfinite_budgets_no_raise(monkeypatch):
    """Infinity/NaN budgets fail conservatively without raising."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    monkeypatch.setattr(sl, "detect_presenter_geometry", lambda *a, **k: {})
    # Must not raise.
    sl.maybe_promote_screencast("fake", [(0, 30)], ["GENERAL"], {},
                               max_scenes=float("inf"))
    sl.maybe_promote_screencast("fake", [(0, 30)], ["GENERAL"], {},
                               max_requests=float("nan"))
    sl.maybe_promote_screencast("fake", [(0, 30)], ["GENERAL"], {},
                               deadline_s=float("inf"))


def test_subset_ids_preserved(monkeypatch):
    """Bounded selection preserves original scene IDs."""
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    seen = []
    def geometry(path, scenes, *a, **k):
        seen.extend(scenes)
        return {}
    monkeypatch.setattr(sl, "detect_presenter_geometry", geometry)
    scenes = [(i * 30, (i + 1) * 30) for i in range(10)]
    # Make scenes 5,6 eligible; others SPLIT.
    strategies = ["SPLIT"] * 5 + ["GENERAL", "GENERAL"] + ["SPLIT"] * 3
    sl.maybe_promote_screencast("fake", scenes, strategies, {}, max_scenes=2)
    # Geometry should see the two eligible scenes (original objects).
    assert seen == [scenes[5], scenes[6]]


@pytest.mark.parametrize("fail", [False, True])
def test_client_closed_on_all_exits(monkeypatch, fail):
    """Every owned SDK client is closed exactly once."""
    state = {"time": 0., "calls": 0, "options": [],
             "created": 0, "closed": 0}
    _sdk(monkeypatch, state, fail=fail)
    sdk = sys.modules["google.genai"]
    orig = sdk.Client

    def counting_client(**kw):
        c = orig(**kw)
        state["created"] += 1
        c.close = lambda: state.__setitem__("closed", state["closed"] + 1)
        return c

    monkeypatch.setattr(sdk, "Client", counting_client)
    lc._gemini_classify([b"x"] * 3, api_key="fake",
                        _budget=_budget(3, 120))
    assert state["created"] >= 1
    assert state["closed"] == state["created"], state


@pytest.mark.parametrize("error", ["part", "replacement"])
def test_setup_and_replacement_errors_close_once(monkeypatch, error):
    """Part setup failure must not leak a client; failed replacement must
    not double-close. Every created client closed exactly once."""
    state = {"time": 0., "calls": 0, "options": [],
             "clients": [], "creates": 0}
    _sdk(monkeypatch, state)
    sdk = sys.modules["google.genai"]
    sdk_types = sys.modules["google.genai.types"]
    orig = sdk.Client

    def client(**kw):
        state["creates"] += 1
        if error == "replacement" and state["creates"] == 2:
            raise RuntimeError("injected replacement creation failure")
        c = orig(**kw)
        record = {"closed": 0}
        state["clients"].append(record)
        c.close = lambda: record.__setitem__("closed", record["closed"] + 1)
        return c

    def part(**kw):
        if error == "part":
            raise ValueError("injected Part setup failure")
        state["time"] = 91.
        return kw

    monkeypatch.setattr(sdk, "Client", client)
    monkeypatch.setattr(sdk_types, "Part",
                        types.SimpleNamespace(from_bytes=part))
    result = lc._gemini_classify(
        [b"image"] * 3, api_key="fake",
        _budget=sl._InvocationBudget(3, 120, lambda: state["time"]))
    assert result is None
    assert state["calls"] == 0
    assert state["clients"], "expected at least one client record"
    assert all(c["closed"] == 1 for c in state["clients"]), state
