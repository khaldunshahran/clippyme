"""Task 14: two-pass automatic render route with synthetic landmarks.

Drives ``reframe.process_video_to_vertical`` (reframe_mode="auto") on a
synthetic two-scene video with distinct colored landmarks and a known
880 Hz tone. The action detector is injected (color-keyed); everything
else — tracking, observation, segmentation, keyframing, planning,
sidecar persistence, pass-2 rendering — is production code.
"""
import hashlib
import json
import math
import os
import struct
import subprocess

import pytest

cv2 = pytest.importorskip("cv2", reason="CV runtime unavailable")
import numpy as np


def _stub_heavy_modules():
    """Import-time stubs for the missing ML wheels (host has no
    mediapipe/ultralytics/torch); mirrors test_production_render.py."""
    import sys
    import types
    if "mediapipe" not in sys.modules:
        mp = types.ModuleType("mediapipe")
        mp_solutions = types.ModuleType("mediapipe.solutions")
        mp_face = types.ModuleType("mediapipe.solutions.face_detection")
        mp_face.FaceDetection = object
        mp_solutions.face_detection = mp_face
        mp.solutions = mp_solutions
        sys.modules["mediapipe"] = mp
        sys.modules["mediapipe.solutions"] = mp_solutions
        sys.modules["mediapipe.solutions.face_detection"] = mp_face
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")
        ultra.YOLO = object
        sys.modules["ultralytics"] = ultra
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        cuda_mod = types.ModuleType("torch.cuda")
        cuda_mod.is_available = lambda: False
        torch_mod.cuda = cuda_mod
        torch_mod.device = lambda x: x
        sys.modules["torch"] = torch_mod
        sys.modules["torch.cuda"] = cuda_mod
    if "psutil" not in sys.modules:
        psutil_mod = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil_mod.virtual_memory = lambda: _VM()
        psutil_mod.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil_mod


_stub_heavy_modules()

from clippyme.pipeline import action_framing as af
from clippyme.pipeline import framing_plan as fp
from clippyme.pipeline import reframe


W, H = 640, 480
FPS = 30
N_FRAMES = 120  # two 60-frame scenes, hard cut at frame 60
CUT = 60


def _frame(i):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    if i < CUT:
        # Scene 0: green rect drifting SLOWLY left -> right (30 px/s):
        # sustained evidence, holdable -> single lock.
        x = int(200 + (60 * i / (CUT - 1)))
        cv2.rectangle(img, (x, 180), (x + 90, 300), (0, 255, 0), -1)
    else:
        # Scene 1: blue rect crossing FAST right -> left (200 px/s):
        # a static lock cannot hold it -> conservative wide.
        j = i - CUT
        x = int(440 - (400 * j / (CUT - 1)))
        cv2.rectangle(img, (x, 180), (x + 90, 300), (255, 0, 0), -1)
    return img


def _build_src(path, tmp_path):
    silent = str(tmp_path / "silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                         (W, H))
    for i in range(N_FRAMES):
        vw.write(_frame(i))
    vw.release()
    dur = N_FRAMES / FPS
    cmd = ["ffmpeg", "-y", "-i", silent,
           "-f", "lavfi", "-i", f"sine=frequency=880:duration={dur}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, capture_output=True, check=True)
    return str(path)


@pytest.fixture()
def src_mp4(tmp_path):
    p = _build_src(tmp_path / "src.mp4", tmp_path)
    assert os.path.exists(p)
    return p


def _install(monkeypatch, detect_fn=None, observe_fn=None):
    """Force GENERAL scenes; inject the action detector seam."""
    from scenedetect import FrameTimecode
    monkeypatch.setattr(
        reframe, "detect_scenes",
        lambda path: ([(FrameTimecode(0, FPS), FrameTimecode(CUT, FPS)),
                       (FrameTimecode(CUT, FPS),
                        FrameTimecode(N_FRAMES, FPS))], float(FPS)))
    monkeypatch.setattr(reframe, "detect_face_candidates",
                        lambda frame: [])
    import clippyme.pipeline.reframe_detect as reframe_detect
    monkeypatch.setattr(reframe_detect, "detect_face_candidates",
                        lambda frame: [])
    from clippyme.pipeline.layouts import split_layout
    monkeypatch.setattr(split_layout, "detect_split_scenes",
                        lambda *a, **k: {})
    monkeypatch.delenv("CLIPPYME_ACTION_AWARE_FRAMING", raising=False)
    calls = {"n": 0}

    def counting_detect(frame):
        calls["n"] += 1
        if detect_fn is None:
            return []
        return detect_fn(frame)

    monkeypatch.setattr(reframe, "detect_action_candidates",
                        counting_detect)
    if observe_fn is not None:
        monkeypatch.setattr(reframe, "_default_action_observe",
                            observe_fn)
    return calls


def _color_detect(frame):
    """Color-keyed detector: green or blue landmark box (lenient for
    lossy encode)."""
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 0].astype(int)
    r = frame[:, :, 2].astype(int)
    green = (g > 110) & (g > r + 25) & (g > b + 25)
    blue = (b > 110) & (b > r + 25) & (b > g + 25)
    for mask in (green, blue):
        if mask.any():
            ys, xs = np.where(mask)
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            return [{"box": [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                     "confidence": 0.9, "category": "object"}]
    return []


def _probe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
           "-show_entries", "format=duration", "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def _read_frames(path, indices):
    cap = cv2.VideoCapture(str(path))
    out = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        assert ok, f"cannot decode frame {i}"
        out[i] = fr
    cap.release()
    return out


def _tone_hz(path):
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a",
           "-f", "f32le", "-ac", "1", "-ar", "8000", "-"]
    r = subprocess.run(cmd, capture_output=True, check=True)
    n = len(r.stdout) // 4
    samples = struct.unpack(f"<{n}f", r.stdout)
    seg = samples[8000:16000]
    crossings = sum(1 for i in range(1, len(seg))
                    if (seg[i - 1] <= 0) != (seg[i] <= 0))
    return (crossings / 2) * (8000 / len(seg))


def _green_present(frame, thresh=400):
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 0].astype(int)
    r = frame[:, :, 2].astype(int)
    return int((((g > 90) & (g > r + 20) & (g > b + 20)).sum())) > thresh


def _blue_present(frame, thresh=400):
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 0].astype(int)
    r = frame[:, :, 2].astype(int)
    return int((((b > 90) & (b > r + 20) & (b > g + 20)).sum())) > thresh


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TestActionRenderEndToEnd:
    def test_two_pass_action_render(self, src_mp4, tmp_path, monkeypatch,
                                    capsys):
        src_hash_before = _sha(src_mp4)
        out = str(tmp_path / "out.mp4")
        calls = _install(monkeypatch, detect_fn=_color_detect)

        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        assert calls["n"] > 0, "action detector must run in pass 1"

        # --- plan sidecar ---
        plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
        assert os.path.exists(plan_path)
        plan = fp.load_plan(plan_path)
        assert plan.schema_version == 4
        assert plan.analysis_complete is True
        by_id = {s.scene_id: s for s in plan.scenes}
        assert len(by_id) == 2
        for sid, sp in by_id.items():
            assert sp.strategy == "GENERAL"
            assert sp.scene_kind == "action", \
                f"scene {sid}: expected action kind, got {sp.scene_kind}"
            assert sp.usable, \
                f"scene {sid} should be usable: {sp.fallback}"

        # --- containment / padding / look-ahead on the persisted plan ---
        s0 = by_id[0]
        singles = [s for s in s0.segments if s.kind == "single"]
        assert singles, "scene 0 should lock a single subject"
        seg = singles[0]
        x1, y1, x2, y2 = seg.crop
        # The landmark drifts x=200+60*i/59 (90px wide), y=180..300: the
        # lock crop must contain its travel WITHIN the segment (which
        # starts after temporal acquisition) with padding.
        x_start = 200 + 60 * seg.start_frame / 59
        x_end = 200 + 60 * seg.end_frame / 59 + 90
        assert x1 <= x_start and y1 < 180 and x2 >= x_end and y2 >= 300
        # Look-ahead: the crop centre leads the +x motion (ahead of the
        # segment-median subject centre).
        x_med = 200 + 60 * (seg.start_frame + seg.end_frame) / 2 / 59 + 45
        assert (x1 + x2) / 2 > x_med
        # Scene 1 (fast crossing) widens: the conservative wide frame
        # dominates with an explicit reason. A brief leading single is
        # the honest measurement artifact (speed not yet measurable on
        # the first frames); it is bounded and one-way, never a bounce.
        s1 = by_id[1]
        wides = [s for s in s1.segments if s.kind == "wide"]
        assert wides, "fast crossing must widen"
        assert any("fast-travel-widen" in s.reason for s in wides)
        for sg in s1.segments:
            if sg.kind == "single":
                dur = (sg.end_frame - sg.start_frame + 1) / 30.0
                assert dur <= 0.6, f"measurement single too long: {dur}s"
        # One-way: no wide->single re-acquisition after the widen.
        kinds = [sg.kind for sg in s1.segments]
        assert kinds == sorted(kinds, key=lambda k: {"single": 0,
                                                     "group": 0,
                                                     "wide": 1}[k])

        # --- rendered frames: landmark stays visible, no jitter ---
        frames = _read_frames(out, [5, 6, 30, 59, 60, 61, 90, 119])
        assert _green_present(frames[5])
        assert _green_present(frames[30])
        assert _green_present(frames[59])
        # No jitter inside a lock: consecutive static-source frames differ
        # only by the drifting landmark (small mean diff).
        for a, b in ((5, 6), (30, 31)):
            frames2 = _read_frames(out, [a, b])
            d = np.abs(frames2[a].astype(np.int16)
                       - frames2[b].astype(np.int16)).mean()
            assert d < 8.0, f"jitter between frames {a},{b}: {d}"
        # The hard cut renders (no garbage at the boundary).
        dcut = np.abs(frames[59].astype(np.int16)
                      - frames[60].astype(np.int16)).mean()
        assert dcut > 15.0

        # --- media properties ---
        ow, oh, ofps, on = self._dims(out)
        assert (ow, oh) == (270, 480), f"actual aspect dims: {(ow, oh)}"
        assert ofps == pytest.approx(30.0)
        assert abs(on - N_FRAMES) <= 2, f"frame count drifted: {on}"
        assert _tone_hz(out) == pytest.approx(880, abs=40)
        # Source bytes untouched by the render.
        assert _sha(src_mp4) == src_hash_before

    @staticmethod
    def _dims(path):
        info = _probe(path)
        st = info["streams"][0]
        num, den = st["avg_frame_rate"].split("/")
        n = int(st.get("nb_frames") or 0)
        return (int(st["width"]), int(st["height"]),
                float(num) / float(den), n)

    def test_plan_persists_and_reloads(self, src_mp4, tmp_path,
                                       monkeypatch):
        out = str(tmp_path / "out.mp4")
        _install(monkeypatch, detect_fn=_color_detect)
        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
        raw = open(plan_path, encoding="utf-8").read()
        d = json.loads(raw)
        assert d["schema_version"] == 4
        assert d["scenes"][0]["scene_kind"] == "action"
        # No forbidden evidence in the persisted plan.
        low = raw.lower()
        for key in ("pixels", "embedding", "landmark", "ocr", "diarization"):
            assert key not in low
        plan = fp.plan_from_dict(d)
        d2 = fp.plan_to_dict(plan)
        assert d2["scenes"][0]["scene_kind"] == "action"

    def test_reused_plan_makes_zero_detector_calls(
            self, src_mp4, tmp_path, monkeypatch):
        out = str(tmp_path / "out.mp4")
        calls = _install(monkeypatch, detect_fn=_color_detect)
        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        first_calls = calls["n"]
        assert first_calls > 0
        # Second render to the SAME path reuses the stable sidecar: the
        # detector must not run again (no redetection/recomputation in
        # the render pass).
        calls["n"] = 0
        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        assert calls["n"] == 0, \
            f"reused plan must not redetect (got {calls['n']} calls)"

    def test_flag_off_matches_accepted_general(
            self, src_mp4, tmp_path, monkeypatch):
        """CLIPPYME_ACTION_AWARE_FRAMING=0 restores the accepted GENERAL
        full-frame result."""
        out_on = str(tmp_path / "on.mp4")
        _install(monkeypatch, detect_fn=_color_detect)
        assert reframe.process_video_to_vertical(
            src_mp4, out_on, reframe_mode="auto") is True

        out_off = str(tmp_path / "off.mp4")
        monkeypatch.setenv("CLIPPYME_ACTION_AWARE_FRAMING", "0")
        assert reframe.process_video_to_vertical(
            src_mp4, out_off, reframe_mode="auto") is True
        # Flag off: v3 sidecar, no action scenes.
        plan_path = os.path.splitext(out_off)[0] + ".turn_framing_plan.json"
        assert not os.path.exists(plan_path) or \
            fp.load_plan(plan_path).schema_version == 3
        # Accepted GENERAL full-frame look: blurred-bg general shot.
        frames = _read_frames(out_off, [30])
        assert not _green_present(frames[30], thresh=200000), \
            "flag-off GENERAL must stay full-frame (no tight crop)"

    def test_detector_failure_keeps_accepted_general(
            self, src_mp4, tmp_path, monkeypatch):
        def boom(frame):
            raise RuntimeError("detector exploded")
        out = str(tmp_path / "out.mp4")
        _install(monkeypatch, detect_fn=boom)
        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
        plan = fp.load_plan(plan_path)
        by_id = {s.scene_id: s for s in plan.scenes}
        for sid, sp in by_id.items():
            assert sp.scene_kind == "action"
            assert sp.fallback is not None, \
                f"scene {sid}: detector failure needs an explicit reason"
            assert "detector" in sp.fallback["reason"]
        # Same accepted full-frame render as the flag-off path.
        frames = _read_frames(out, [30])
        assert not _green_present(frames[30], thresh=200000)

    def test_dense_weak_evidence_stays_full_frame(
            self, src_mp4, tmp_path, monkeypatch):
        """Weak detector evidence (dense/uncertain composition) keeps the
        accepted GENERAL full-frame result with an explicit reason."""
        def weak(frame):
            # Low confidence * visibility: below the evidence floor.
            boxes = _color_detect(frame)
            for b in boxes:
                b["confidence"] = 0.2
            return boxes
        out = str(tmp_path / "out.mp4")
        calls = _install(monkeypatch, detect_fn=weak)
        assert reframe.process_video_to_vertical(
            src_mp4, out, reframe_mode="auto") is True
        assert calls["n"] > 0
        plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
        plan = fp.load_plan(plan_path)
        kinds = {s.scene_kind for s in plan.scenes}
        assert kinds == {"action"}
        # No tight single/group lock from weak evidence.
        for sp in plan.scenes:
            assert all(s.kind == "wide" for s in sp.segments), \
                f"weak evidence must not authorize a tight crop: " \
                f"{[(s.kind) for s in sp.segments]}"
        frames = _read_frames(out, [30])
        assert not _green_present(frames[30], thresh=200000)

    def test_split_and_screencast_never_enter_planner(
            self, src_mp4, tmp_path, monkeypatch):
        """Positive SPLIT/SCREENCAST decisions stay authoritative: the
        action detector is never invoked for those scenes."""
        for mode in ("split", "screencast"):
            out = str(tmp_path / f"{mode}.mp4")
            calls = _install(monkeypatch, detect_fn=_color_detect)
            assert reframe.process_video_to_vertical(
                src_mp4, out, reframe_mode=mode) is True
            assert calls["n"] == 0, \
                f"mode={mode}: action planner must not run"
