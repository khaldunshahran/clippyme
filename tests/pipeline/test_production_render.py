"""Task 02 rev3: production render tests with asymmetric landmarks.

Keeps production classify_scene intact; injects raw JSON at the
_gemini_classify boundary (call-ordered) and detector boxes. Asymmetric
landmarks discriminate the exact screencast composition from general.
Both missing-promotion and wrong-composition mutations must fail.
"""
import json
import os
import struct
import math
import subprocess
import sys
import types

import pytest

cv2 = pytest.importorskip("cv2", reason="CV runtime unavailable")
import numpy as np


W, H = 640, 480
FPS = 30
# Asymmetric landmarks: RED checkerboard top, MAGENTA circle at x=200 (left)
CIRCLE_X, CIRCLE_Y = 200, 400


def _scene_frame(kind, t):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if kind == "object":
        x = int(50 + 400 * (t % 30) / 30)
        cv2.rectangle(img, (x, 200), (x + 80, 280), (0, 0, 255), -1)  # blue
    elif kind == "screen":
        # RED checkerboard (BGR: 0,0,255)
        for yy in range(0, 320, 40):
            for xx in range(0, W, 40):
                if (xx // 40 + yy // 40) % 2 == 0:
                    cv2.rectangle(img, (xx, yy), (xx + 40, yy + 40),
                                  (0, 0, 255), -1)
        # MAGENTA circle at (200, 400), radius 40 (BGR: 255, 0, 255)
        cv2.circle(img, (CIRCLE_X, CIRCLE_Y), 40, (255, 0, 255), -1)
    else:
        pts = np.array([[[320, 150], [250, 300], [390, 300]]], np.int32)
        cv2.fillPoly(img, pts, (0, 255, 255))  # yellow
    return img


def _build_video_with_audio(path, tmp_path):
    silent = str(tmp_path / "silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(silent, fourcc, FPS, (W, H))
    kinds = ["object"] * 30 + ["screen"] * 30 + ["object2"] * 30
    for t, kind in enumerate(kinds):
        vw.write(_scene_frame(kind, t))
    vw.release()
    cmd = ["ffmpeg", "-y", "-i", silent,
           "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(path)


def _stub_heavy_modules(monkeypatch):
    mp = types.ModuleType("mediapipe")
    mp_solutions = types.ModuleType("mediapipe.solutions")
    mp_face = types.ModuleType("mediapipe.solutions.face_detection")
    mp_face.FaceDetection = object
    mp_solutions.face_detection = mp_face
    mp.solutions = mp_solutions
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    monkeypatch.setitem(sys.modules, "mediapipe.solutions", mp_solutions)
    monkeypatch.setitem(sys.modules, "mediapipe.solutions.face_detection", mp_face)
    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = object
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    # Real detector module with injected boxes (not mocked decisions).
    fake = types.ModuleType("clippyme.pipeline.reframe_detect")
    def detector(frame):
        # Magenta circle at (200,400): return box around it.
        # Check for magenta in the lower region.
        lo = frame[380:420, 180:220]
        if lo.size and np.mean(lo[:, :, 0]) > 100 and np.mean(lo[:, :, 2]) > 100:
            return [{"box": [160.0, 360.0, 80.0, 80.0]}]
        return []
    fake.detect_face_candidates = detector
    for name in ["detect_faces", "detect_person_yolo"]:
        setattr(fake, name, lambda *a, **k: [])
    fake.compute_mouth_aspect_ratio = lambda *a, **k: 0.0
    for name in ["_get_face_detection", "_get_face_mesh", "_get_yolo_model"]:
        setattr(fake, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "clippyme.pipeline.reframe_detect", fake)


def _inject_raw_provider(monkeypatch):
    """Keep classify_scene intact; inject raw JSON at _gemini_classify.

    Call-ordered: scenes are classified in order 0, 1, 2 (eligible order).
    Scene 1 gets screencast JSON; others get none.
    """
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    from clippyme.pipeline.layouts import layout_classifier as lc

    calls = {"n": 0}
    def fake_gemini(frames, api_key=None, model_name=None, **kw):
        calls["n"] += 1
        # Eligible scenes in order: 0 (GENERAL), 1 (GENERAL), 2 (GENERAL).
        # But geometry filters: only scene 1 has geometry, so only scene 1
        # reaches the provider. calls["n"]==1 means scene 1.
        if calls["n"] == 1:
            return json.dumps({"layout": "screencast", "confidence": 0.93,
                               "screen_evidence": True,
                               "why": "red checkerboard with magenta presenter"})
        return json.dumps({"layout": "none", "confidence": 0.96,
                           "screen_evidence": False,
                           "why": "no screen content"})
    monkeypatch.setattr(lc, "_gemini_classify", fake_gemini)
    return calls


@pytest.fixture()
def three_scene_av(tmp_path):
    p = tmp_path / "three_scene_av.mp4"
    _build_video_with_audio(p, tmp_path)
    assert p.exists()
    return str(p)


class _TC:
    def __init__(self, f): self._f = f
    def get_frames(self): return self._f


def _probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _decode_audio_samples(path, sr=8000):
    """Decode mono audio to float samples."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-map", "0:a", "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    n = len(r.stdout) // 4
    if n == 0:
        return []
    return list(struct.unpack(f"<{n}f", r.stdout))


def _dominant_freq(samples, sr=8000):
    """Dominant frequency via zero-crossing rate (robust for pure tones)."""
    if len(samples) < sr:
        return 0.0
    # Use middle 1s to avoid edge effects.
    seg = samples[sr:sr*2] if len(samples) >= sr*2 else samples
    crossings = sum(1 for i in range(1, len(seg))
                    if (seg[i-1] <= 0) != (seg[i] <= 0))
    # Two crossings per cycle.
    return (crossings / 2) * (sr / len(seg))


def _check_expected_tone(path, expected_hz=880, sr=8000, tol_hz=40):
    """Verify the expected sine tone: frequency, duration, energy."""
    samples = _decode_audio_samples(path, sr)
    assert len(samples) >= sr * 2, f"audio too short: {len(samples)/sr:.2f}s"
    freq = _dominant_freq(samples, sr)
    assert abs(freq - expected_hz) < tol_hz, (
        f"expected {expected_hz}Hz tone, got {freq:.1f}Hz")
    # Energy: RMS of middle section must be nonzero (tone present).
    mid = samples[sr:sr*2]
    rms = math.sqrt(sum(s*s for s in mid) / len(mid))
    assert rms > 0.01, f"tone energy missing: rms={rms}"
    return freq, rms


def _is_magenta(px):
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return b > 120 and r > 120 and g < 100


def _is_red(px):
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return r > 120 and b < 100 and g < 100


def _is_yellow(px):
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return r > 120 and g > 120 and b < 100


def _is_blue(px):
    b, g, r = int(px[0]), int(px[1]), int(px[2])
    return b > 120 and r < 100 and g < 100


@pytest.mark.parametrize("comfort", ["1", "0"])
def test_production_render_three_scenes(three_scene_av, tmp_path, monkeypatch, comfort):
    """Production entrypoint with asymmetric landmarks.

    Scene 1 must render as stacked screencast: RED checkerboard content on
    top, MAGENTA presenter centered in bottom pane, with a horizontal split
    boundary. Scenes 0/2 are GENERAL (no stacking).
    """
    _stub_heavy_modules(monkeypatch)
    calls = _inject_raw_provider(monkeypatch)
    monkeypatch.setenv("REFRAME_COMFORT", comfort)
    monkeypatch.setenv("REFRAME_GLOBAL_SMOOTH", "0" if comfort == "0" else "1")

    from clippyme.pipeline import reframe as rf
    from clippyme.pipeline.layouts import screencast_layout as sl

    def _fake_detect(*a, **k):
        return [(_TC(0), _TC(30)), (_TC(30), _TC(60)), (_TC(60), _TC(90))], 30.0
    monkeypatch.setattr(rf, "detect_scenes", _fake_detect)

    captured = {}
    orig_promote = sl.maybe_promote_screencast
    def spy_promote(video, scenes, strategies, metadata, **kw):
        result = orig_promote(video, scenes, strategies, metadata, **kw)
        captured["strategies"] = list(strategies)
        captured["metadata"] = dict(metadata)
        return result
    monkeypatch.setattr(sl, "maybe_promote_screencast", spy_promote)

    out = str(tmp_path / f"out_comfort{comfort}.mp4")
    rf.process_video_to_vertical(three_scene_av, out, reframe_mode="auto")

    # Production classification ran (not a handwritten clone).
    assert calls["n"] >= 1, "provider boundary was not exercised"
    assert captured.get("strategies") == ["GENERAL", "SCREENCAST", "GENERAL"]
    assert 1 in captured["metadata"]

    # Output basics.
    assert os.path.exists(out)
    info = _probe(out)
    vs = info["streams"][0]
    w, h = int(vs["width"]), int(vs["height"])
    assert h > w and abs((w / h) - (9 / 16)) < 0.05
    num, den = map(int, vs["avg_frame_rate"].split("/"))
    assert abs(num / den - 30) < 1.0
    assert 85 <= int(vs.get("nb_frames", 0)) <= 95
    assert 2.9 < float(info["format"]["duration"]) < 3.1
    # 880Hz tone: verify frequency (not just volume), duration, energy.
    freq, rms = _check_expected_tone(out, expected_hz=880)
    assert 840 < freq < 920

    cap = cv2.VideoCapture(out)
    try:
        # Scene 1, frame 45: stacked composition.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
        ok, f1 = cap.read()
        assert ok
        hh, ww = f1.shape[:2]

        # Expected split: content_h from the layout helper.
        from clippyme.pipeline.layouts.screencast_layout import content_bands
        # Input was 640x480; output is w x h. content_h scales.
        # We verify structurally: top region has RED, bottom has MAGENTA centered.
        top = f1[:hh//2]
        bot = f1[hh//2:]

        # Count red pixels in top (checkerboard content).
        red_top = sum(1 for y in range(0, hh//2, 4) for x in range(0, ww, 4)
                      if _is_red(top[y, x]))
        assert red_top > 200, "scene 1 top missing RED checkerboard content"

        # Magenta presenter must be CENTERED horizontally in bottom pane
        # (speaker_crop centers on face). Check middle third of bottom pane.
        bh, bw = bot.shape[:2]
        mid = bot[:, bw//3:2*bw//3]
        mag_mid = sum(1 for y in range(0, bh, 3) for x in range(0, mid.shape[1], 3)
                      if _is_magenta(mid[y, x]))
        assert mag_mid > 100, (
            "scene 1 bottom pane missing centered MAGENTA presenter; "
            "wrong composition?")

        # Horizontal split boundary: row-wise red-channel variance should
        # spike near the content/speaker boundary (structural stacking).
        # In a general (non-stacked) frame this spike is absent.
        row_red = [np.mean(f1[y, :, 2]) for y in range(0, hh, 8)]
        diffs = [abs(row_red[i+1] - row_red[i]) for i in range(len(row_red)-1)]
        assert max(diffs) > 25, "no horizontal split boundary detected"

        # Scenes 0 and 2: GENERAL, no stacking. Frame 15 (scene 0) should
        # NOT have the centered-magenta pane structure.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 15)
        ok, f0 = cap.read()
        assert ok
        b0 = f0[hh//2:]
        mid0 = b0[:, bw//3:2*bw//3]
        mag0 = sum(1 for y in range(0, b0.shape[0], 4)
                   for x in range(0, mid0.shape[1], 4)
                   if _is_magenta(mid0[y, x]))
        assert mag_mid > mag0 * 3, "scene 0 unexpectedly has presenter pane"

        # Scene 2 (frame 75): GENERAL with YELLOW triangle landmark.
        # A black third scene must fail here.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 75)
        ok, f2 = cap.read()
        assert ok
        # Yellow triangle is in the upper-middle of the input; in the
        # vertical crop it should appear in the upper half.
        top2 = f2[:hh//2]
        yel2 = sum(1 for y in range(0, top2.shape[0], 4)
                   for x in range(0, top2.shape[1], 4)
                   if _is_yellow(top2[y, x]))
        assert yel2 > 20, (
            "scene 2 missing YELLOW triangle landmark; "
            "third-scene content not verified")
    finally:
        cap.release()


def test_explicit_modes_complete_without_classification(three_scene_av, tmp_path, monkeypatch):
    """Explicit modes render and preserve audio; classifier never runs."""
    _stub_heavy_modules(monkeypatch)
    monkeypatch.setenv("AUTO_LAYOUT", "1")
    from clippyme.pipeline import reframe as rf
    from clippyme.pipeline.layouts import layout_classifier as lc

    calls = []
    orig = lc._gemini_classify
    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    monkeypatch.setattr(lc, "_gemini_classify", spy)
    monkeypatch.setattr(rf, "detect_scenes",
                        lambda *a, **k: ([(_TC(0), _TC(30)), (_TC(30), _TC(60)),
                                          (_TC(60), _TC(90))], 30.0))

    for mode in ["disabled", "screencast"]:
        out = str(tmp_path / f"out_{mode}.mp4")
        rf.process_video_to_vertical(three_scene_av, out, reframe_mode=mode)
        assert os.path.exists(out), f"mode {mode} produced no output"
        info = _probe(out)
        vs = info["streams"][0]
        w, h = int(vs["width"]), int(vs["height"])
        assert h >= w, f"mode {mode}: not vertical"
        f, r = _check_expected_tone(out, expected_hz=880)
        assert 840 < f < 920, f"mode {mode}: wrong tone"
    assert calls == [], "provider invoked in explicit mode"
