"""Integration test for the Task 12 turn-aware framing plan (two-pass route).

Drives ``process_video_to_vertical`` end-to-end on a synthetic audiovisual
fixture with injected deterministic face detections and mouth-aspect-ratio
evidence, exercising the actual automatic two-pass path: pass 1 records
observations, one plan is built + atomically persisted, and pass 2 renders
from the persisted plan verbatim.

The laptop's host python lacks the ML wheels (mediapipe/ultralytics/torch),
so import-time stubs stand in for them — the same pattern as
test_av_acceptance.py's ``_install_heavy_stubs``. The real cv2 decode, scene
detection, plan build, atomic persistence and ffmpeg encode/mux paths all run.
Every detection and MAR value is injected; nothing here validates real
active-speaker accuracy, identities, or non-face action framing.
"""
import hashlib
import json
import os
import subprocess
import sys
import types

import pytest

pytestmark = pytest.mark.integration


def _install_heavy_stubs():
    """Install import-time stubs for the missing ML wheels (host has no
    mediapipe/ultralytics/torch/psutil), then remove them from sys.modules
    after our imports complete.

    The removal is critical for test isolation: other test modules guard
    with ``try: from clippyme.pipeline import reframe except: pytest.skip``.
    If the stubs leaked, those guards would pass and the tests would run
    against broken stubs instead of skipping (they're Docker-only).
    Our module keeps working because the already-imported ``reframe``
    holds direct references to the stub objects.
    """
    if getattr(_install_heavy_stubs, "done", False):
        return
    _stubs = []
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        mod = types.ModuleType("ultralytics")
        mod.YOLO = type("YOLO", (), {})
        sys.modules["ultralytics"] = mod
        _stubs.append("ultralytics")
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_mod = types.ModuleType("torch")
        cuda_mod = types.ModuleType("torch.cuda")
        cuda_mod.is_available = lambda: False
        torch_mod.cuda = cuda_mod
        torch_mod.device = lambda x: x
        sys.modules["torch"] = torch_mod
        sys.modules["torch.cuda"] = cuda_mod
        _stubs.extend(["torch", "torch.cuda"])
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        sys.modules["mediapipe"] = types.ModuleType("mediapipe")
        _stubs.append("mediapipe")
    try:
        import psutil  # noqa: F401
    except ImportError:
        psutil_mod = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil_mod.virtual_memory = lambda: _VM()
        psutil_mod.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil_mod
        _stubs.append("psutil")
    _install_heavy_stubs.done = True
    _install_heavy_stubs.stubs = _stubs


_install_heavy_stubs()

import cv2
import numpy as np

from clippyme.pipeline import reframe
from clippyme.pipeline import framing_plan as fp
from clippyme.pipeline import reframe_detect

# Remove the stubs so other test modules' import guards work correctly.
for _stub_name in getattr(_install_heavy_stubs, "stubs", []):
    sys.modules.pop(_stub_name, None)

# Evict the clippyme modules imported above while the stubs were installed.
# They hold direct references to the stub-bound detector singletons (e.g.
# reframe_detect.YOLO); leaving them in sys.modules would poison other test
# modules -- test_production_render.py installs its own fake
# clippyme.pipeline.reframe_detect via monkeypatch and expects a fresh
# `from clippyme.pipeline import reframe` to pick it up, but a cached
# reframe keeps the stub-bound detector. Our module keeps working through
# its own direct references above.
_EVICT = (
    "clippyme.pipeline.reframe",
    "clippyme.pipeline.reframe_detect",
    "clippyme.pipeline.framing_plan",
)
for _mod_name in _EVICT:
    sys.modules.pop(_mod_name, None)
# A `from clippyme.pipeline import reframe` resolves through the parent
# package's attribute, not just sys.modules: drop those attributes too,
# otherwise the cached stub-bound module would still win.
_pkg = sys.modules.get("clippyme.pipeline")
if _pkg is not None:
    for _mod_name in _EVICT:
        _attr = _mod_name.rpartition(".")[2]
        if getattr(_pkg, _attr, None) is not None:
            try:
                delattr(_pkg, _attr)
            except AttributeError:
                pass
del _EVICT


# --- synthetic audiovisual fixture ------------------------------------------

W, H, FPS = 640, 360, 30
# Scene 1: two equal-size faces, A (left/orange) talks 0-59, B (right/teal)
# talks 60-119 — a speaker turn with NO visual cut (the defect Task 12 fixes).
# Scene 2: hard cut, one face (right/green), stable single subject.
# Scene 3: faceless (GENERAL routing check).
BOX_A = (60, 90, 90, 90)
BOX_B = (490, 90, 90, 90)
BOX_C = (490, 90, 90, 90)
COL_A = (0, 140, 255)     # orange (BGR)
COL_B = (255, 150, 0)     # teal (BGR)
COL_C = (0, 200, 0)       # green (BGR)
BG = [(50, 50, 50), (50, 50, 180), (180, 50, 50)]
N_FRAMES = 240


def _marker_index(frame):
    """Frame number encoded in a 32x32 grayscale block at (0,0).

    Grayscale survives YUV 4:2:0 subsampling (luma is full-resolution);
    the 32x32 mean is robust to single-pixel codec noise. Values 0-239
    fit in one byte.
    """
    return int(round(frame[0:32, 0:32, 0].mean()))


def _build_fixture(path):
    """Lossless 640x360/30fps/8s clip with an 880 Hz sine audio track."""
    import tempfile
    pngdir = tempfile.mkdtemp(prefix="t12frames")
    for i in range(N_FRAMES):
        if i < 120:
            bg, boxes = BG[0], [(BOX_A, COL_A), (BOX_B, COL_B)]
        elif i < 180:
            bg, boxes = BG[1], [(BOX_C, COL_C)]
        else:
            bg, boxes = BG[2], []
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :] = bg
        for (x, y, w, h), col in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, -1)
        frame[0:32, 0:32] = (i % 256, i % 256, i % 256)  # BGR grayscale marker
        cv2.imwrite(f"{pngdir}/f{i:03d}.png", frame)
    cmd = [
        "ffmpeg", "-y", "-start_number", "0",
        "-framerate", str(FPS), "-i", f"{pngdir}/f%03d.png",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return path


def _install_fakes(monkeypatch, turn_at=20):
    """Inject deterministic detections + MAR evidence keyed off the marker."""
    def fake_detect(frame):
        idx = _marker_index(frame)
        if idx < 120:
            return [{"box": [60, 90, 90, 90], "score": 8100.0},
                    {"box": [490, 90, 90, 90], "score": 8100.0}]
        if idx < 180:
            return [{"box": [490, 90, 90, 90], "score": 8100.0}]
        return []

    def fake_mar(frame, box):
        idx = _marker_index(frame)
        x = box[0]
        if idx < 120:
            active_x = 60 if idx < turn_at else 490
        elif idx < 180:
            active_x = 490
        else:
            return 0.25
        if abs(x - active_x) < 5:
            # Alternating MAR with period 4: even sampled frames alternate,
            # giving the tracker real mouth-motion variance.
            return 0.55 if (idx % 4) < 2 else 0.15
        return 0.25

    monkeypatch.setattr(reframe, "detect_face_candidates", fake_detect)
    monkeypatch.setattr(reframe_detect, "detect_face_candidates", fake_detect)
    monkeypatch.setattr(reframe, "compute_mouth_aspect_ratio", fake_mar)
    monkeypatch.setattr(reframe, "detect_person_yolo", lambda frame: None)
    # Task 14: the action detector seam is orthogonal to turn evidence; keep
    # it silent so these tests exercise the accepted GENERAL fallback.
    monkeypatch.setattr(reframe, "detect_action_candidates", lambda frame: [])
    # The SPLIT upgrade is orthogonal to Task 12; keep the scene WIDE.
    import clippyme.pipeline.layouts.split_layout as split_layout
    monkeypatch.setattr(split_layout, "detect_split_scenes",
                        lambda *a, **k: {})
    monkeypatch.delenv("CLIPPYME_TURN_AWARE_FRAMING", raising=False)


def _read_frames(path, indices):
    """Read specific frames by sequential decode (cap.set seeking is
    imprecise with x264; sequential reads are exact)."""
    want = set(indices)
    out = {}
    cap = cv2.VideoCapture(path)
    try:
        i = 0
        while want:
            ok, frame = cap.read()
            assert ok, f"could not decode output frame {i}"
            if i in want:
                out[i] = frame
                want.remove(i)
            i += 1
    finally:
        cap.release()
    assert not want, f"missing frames: {want}"
    return out


def _audio_peak_hz(path, tmp_path):
    wav = str(tmp_path / "audio.raw")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", "-ar", "8000",
         "-f", "f32le", wav],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = np.fromfile(wav, dtype=np.float32)
    assert len(data) > 8000, "no audio decoded"
    seg = data[:8000]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1.0 / 8000)
    return float(freqs[int(np.argmax(spec[1:])) + 1])


def _probe(path):
    cap = cv2.VideoCapture(path)
    try:
        return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(cap.get(cv2.CAP_PROP_FPS)),
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()


# --- the two-pass acceptance test --------------------------------------------

def test_turn_aware_two_pass_end_to_end(tmp_path, monkeypatch, capsys):
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)

    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto") is True
    logged = capsys.readouterr().out
    assert "turn-aware framing plan" in logged

    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    assert os.path.exists(plan_path), "plan sidecar must be atomically persisted"
    assert not list(tmp_path.glob("*.tmp-*")), "no temp leftovers"

    # --- plan identity ---
    plan = fp.load_plan(plan_path)
    assert plan.schema_version == fp.SCHEMA_VERSION
    assert plan.source_fingerprint == fp.source_fingerprint(src)
    assert (plan.frame_w, plan.frame_h) == (W, H)
    assert plan.fps == pytest.approx(30.0)
    assert plan.frame_count == N_FRAMES
    assert len(plan.scene_intervals) == 3, \
        f"expected 3 scenes, got {plan.scene_intervals}"
    # Task 14: with action-aware framing on (the default), the plan binds
    # the combined turn+action config digest, not the turn-only digest.
    import clippyme.pipeline.action_framing as af
    assert plan.config_digest == af.plan_config_digest(
        fp.config_from_env().config_digest(), af.action_config_from_env())

    by_id = {sp.scene_id: sp for sp in plan.scenes}
    assert by_id[0].strategy == "WIDE" and by_id[0].usable
    assert by_id[1].strategy == "TRACK" and by_id[1].usable
    # Task 14: GENERAL scenes are action-planned (v4). With the action
    # detector silenced, this scene carries the accepted no-observations
    # fallback: applies but not usable, no segments, full-frame render.
    assert by_id[2].strategy == "GENERAL" and by_id[2].scene_kind == "action"
    assert by_id[2].applies and not by_id[2].usable
    assert by_id[2].segments == []
    assert by_id[2].fallback and \
        by_id[2].fallback["reason"] == "no-observations"

    # --- the turn: scene 0 reframes mid-scene between the two speakers ---
    singles = [s for s in by_id[0].segments if s.kind == "single"]
    assert len(singles) >= 2, "expected an accepted speaker turn in scene 0"
    assert singles[0].subject_ids != singles[-1].subject_ids
    turn = next(s for s in singles[1:]
                if s.subject_ids != singles[0].subject_ids)
    # Sustained-evidence lag: the tracker needs ~25 MAR samples + the
    # segmenter's 0.8s window after the turn before the switch is accepted.
    assert 50 <= turn.start_frame <= 90
    assert turn.reason == "turn-switch-sustained-evidence"
    first_crop, last_crop = singles[0].crop, singles[-1].crop
    assert first_crop[2] < W / 2 < last_crop[0], \
        "mid-scene reframe: first crop left, later crop right"
    # Subject containment + headroom on the persisted crop.
    x1, y1, x2, y2 = first_crop
    assert x1 <= 60 and y1 < 90 and x2 >= 150 and y2 >= 180

    # --- hard-cut reset: no identity leaks across the cut ---
    ids0 = {t["id"] for t in by_id[0].tracks}
    ids1 = {t["id"] for t in by_id[1].tracks}
    assert ids0 and ids1 and not (ids0 & ids1)

    # --- output media properties ---
    ow, oh, ofps, on = _probe(out)
    assert (ow, oh) == (int(H * 9 / 16), H) and ow % 2 == 0
    assert ofps == pytest.approx(30.0)
    assert abs(on - N_FRAMES) <= 2, f"frame count drifted: {on}"
    assert _audio_peak_hz(out, tmp_path) == pytest.approx(880, abs=8)

    frames = _read_frames(out, [20, 21, 22, 30, 115, 125, 150, 170, 200])

    def _mean_diff(a, b):
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

    # No flicker inside a stable segment (identical crops on static source;
    # lossy encode noise stays tiny).
    assert _mean_diff(frames[20], frames[21]) < 3.0
    assert _mean_diff(frames[21], frames[22]) < 3.0
    # The turn is visible: frames on opposite sides of the accepted turn
    # (A-side vs B-side crops) differ substantially...
    turn_frame = turn.start_frame
    frames_turn = _read_frames(out, [30, turn_frame + 10])
    assert _mean_diff(frames_turn[30], frames_turn[turn_frame + 10]) > 20.0
    # ...and the hard cut renders (no garbage at the boundary).
    assert _mean_diff(frames[115], frames[125]) > 20.0
    # Scene 2 (stable single subject) holds one static crop. The single
    # segment starts at 128; the per-second cap extends its eased transition,
    # so compare two frames well inside the lock.
    assert _mean_diff(frames[150], frames[170]) < 3.0
    # Scene 3 (GENERAL) keeps the existing renderer: blurred-background
    # general shot (not a face crop, not black letterbox). The top-center
    # pixel is the blurred scene bg, close to the (180,50,50) BGR backdrop.
    px = frames[200][0, ow // 2].astype(int)
    assert abs(int(px[0]) - 180) < 30 and px.max() > 100, \
        "expected the GENERAL blurred-bg renderer"

    # Ease keyframes around the turn are monotonic (no oscillation).
    ease = [k for k in by_id[0].keyframes
            if k.kind == "ease" and turn.start_frame <= k.frame < turn.start_frame + 6]
    assert ease, "expected bounded ease keyframes at the accepted turn"
    xs = [k.x1 for k in sorted(ease, key=lambda k: k.frame)]
    assert xs == sorted(xs)


def test_plan_reuse_skips_rebuild(tmp_path, monkeypatch, capsys):
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    mtime1 = os.path.getmtime(plan_path)
    frames1 = _read_frames(out, [30, 100])

    # Second run on the identical source reuses the persisted plan...
    capsys.readouterr()
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    logged = capsys.readouterr().out
    assert "Reusing persisted turn-aware framing plan" in logged
    assert os.path.getmtime(plan_path) == mtime1, "reuse must not rewrite"
    frames2 = _read_frames(out, [30, 100])

    def _mean_diff(a, b):
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

    assert _mean_diff(frames1[30], frames2[30]) < 3.0
    assert _mean_diff(frames1[100], frames2[100]) < 3.0


def test_stale_and_malformed_plans_rebuild(tmp_path, monkeypatch, capsys):
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"

    # Stale: tamper the FPS -> rebuild, never accepted. (Tampering frame_count
    # would fail load_plan validation as malformed; FPS passes validation
    # but fails the identity match.)
    data = json.loads(open(plan_path, encoding="utf-8").read())
    data["fps"] = 25.0
    open(plan_path, "w", encoding="utf-8").write(json.dumps(data))
    capsys.readouterr()
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    logged = capsys.readouterr().out
    assert "stale-fps" in logged
    assert fp.load_plan(plan_path).fps == pytest.approx(30.0)

    # Malformed: garbage bytes -> rebuild, never accepted.
    open(plan_path, "w", encoding="utf-8").write("{not a plan")
    capsys.readouterr()
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    logged = capsys.readouterr().out
    assert "malformed" in logged
    assert fp.load_plan(plan_path).frame_count == N_FRAMES


def test_render_uses_persisted_plan_not_recomputation(tmp_path, monkeypatch):
    # Mutation gate: if pass 2 recomputed decisions instead of rendering the
    # persisted plan, changing the injected evidence would change the output.
    # With the plan reused, the output must be identical despite different
    # injected MAR evidence on the same video bytes.
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=60)
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    frames_turn60 = _read_frames(out, [30, 100])

    _install_fakes(monkeypatch, turn_at=30)  # different evidence, same bytes
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    frames_turn30 = _read_frames(out, [30, 100])

    def _mean_diff(a, b):
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

    assert _mean_diff(frames_turn60[30], frames_turn30[30]) < 3.0
    assert _mean_diff(frames_turn60[100], frames_turn30[100]) < 3.0

    # And the plan genuinely governs rendering: delete it, keep the new
    # evidence, and the turn moves earlier (turn_at=30 vs turn_at=60).
    os.remove(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    plan = fp.load_plan(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    singles = [s for s in plan.scenes[0].segments if s.kind == "single"]
    turn = next(s for s in singles[1:]
                if s.subject_ids != singles[0].subject_ids)
    assert turn.start_frame < 95, \
        "with fresh earlier evidence the turn must move earlier"


def test_disabled_flag_restores_static_comfort(tmp_path, monkeypatch, capsys):
    # Mutation gate: CLIPPYME_TURN_AWARE_FRAMING=0 must restore the accepted
    # whole-scene median-collapse behavior (no mid-scene reframing, no plan).
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    monkeypatch.setenv("CLIPPYME_TURN_AWARE_FRAMING", "0")
    # Task 14: the action flag is independent — disable it too so the
    # accepted "disabled means no plan, static comfort" contract holds.
    monkeypatch.setenv("CLIPPYME_ACTION_AWARE_FRAMING", "0")

    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto") is True
    logged = capsys.readouterr().out
    assert "turn-aware framing plan" not in logged
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    assert not os.path.exists(plan_path), "disabled flag must not persist a plan"

    # Static comfort: the crop cannot reframe mid-scene. The lossy output
    # encode adds noise, so compare with tolerance on the mean: a mid-scene
    # reframe would shift the crop by hundreds of pixels (mean diff >> 20).
    # (The 32x32 frame-number marker legitimately changes per frame, so a
    # max-pixel check would be brittle.)
    frames = _read_frames(out, [30, 100])
    diff = np.abs(frames[30].astype(np.int16) - frames[100].astype(np.int16))
    assert diff.mean() < 5.0, \
        "rollback must pin one static crop per scene"
    ow, oh, ofps, on = _probe(out)
    assert (ow, oh) == (int(H * 9 / 16), H) and abs(on - N_FRAMES) <= 2


def test_explicit_modes_incur_no_plan_cost(tmp_path, monkeypatch, capsys):
    # Explicit disabled/subject modes: no framing-plan analysis, no sidecar.
    src = _build_fixture(str(tmp_path / "src.mp4"))
    _install_fakes(monkeypatch, turn_at=20)
    for mode in ("disabled", "subject"):
        out = str(tmp_path / f"out_{mode}.mp4")
        assert reframe.process_video_to_vertical(
            src, out, reframe_mode=mode) is True
        assert not os.path.exists(
            os.path.splitext(out)[0] + ".turn_framing_plan.json")
    logged = capsys.readouterr().out
    assert "turn-aware framing plan" not in logged


def test_fresh_reframe_import_honors_fake_detect_module(monkeypatch):
    """Isolation regression: no stub-bound clippyme modules may leak.

    This module installs import-time stubs for the missing ML wheels and
    then imports ``reframe``. It must evict those imports from sys.modules
    afterwards: test_production_render.py installs its own fake
    ``clippyme.pipeline.reframe_detect`` via monkeypatch and expects a fresh
    ``from clippyme.pipeline import reframe`` to bind the fake. A leaked
    stub-bound reframe would bypass the fake and fail with
    ``TypeError: YOLO() takes no arguments``.
    """
    import sys

    _names = ("clippyme.pipeline.reframe", "clippyme.pipeline.reframe_detect")
    _pkg = sys.modules.get("clippyme.pipeline")
    saved = {}
    saved_attrs = {}
    for _name in _names:
        saved[_name] = sys.modules.pop(_name, None)
        _attr = _name.rpartition(".")[2]
        saved_attrs[_attr] = getattr(_pkg, _attr, None)
        if _pkg is not None and hasattr(_pkg, _attr):
            delattr(_pkg, _attr)
    try:
        fake = types.ModuleType("clippyme.pipeline.reframe_detect")
        for _fn in (
            "_get_face_detection",
            "_get_face_mesh",
            "_get_yolo_model",
            "compute_mouth_aspect_ratio",
            "detect_face_candidates",
            "detect_person_yolo",
            "detect_action_candidates",
        ):
            setattr(fake, _fn, lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, "clippyme.pipeline.reframe_detect", fake)
        import importlib

        rf2 = importlib.import_module("clippyme.pipeline.reframe")
        assert rf2._get_yolo_model is fake._get_yolo_model
    finally:
        for _name, _mod in saved.items():
            if _mod is not None:
                sys.modules[_name] = _mod
            else:
                sys.modules.pop(_name, None)
        if _pkg is not None:
            for _attr, _val in saved_attrs.items():
                if _val is not None:
                    setattr(_pkg, _attr, _val)
                elif hasattr(_pkg, _attr):
                    delattr(_pkg, _attr)


def test_import_evicts_stub_bound_modules():
    """Mutation gate: importing this module must not leave stub-bound
    clippyme modules behind in the import cache.

    Without the module-level eviction, ``clippyme.pipeline.reframe`` stays
    cached with the stub-bound detector, so test_production_render.py's fake
    ``reframe_detect`` is silently bypassed and its tests fail with
    ``TypeError: YOLO() takes no arguments``. Verified hermetically in a
    subprocess: removing the eviction makes this test fail.
    """
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    _code = (
        "import sys; "
        "sys.path.insert(0, 'tests/pipeline'); "
        "import test_turn_aware_integration; "
        "pkg = sys.modules.get('clippyme.pipeline'); "
        "leaked = [n for n in sys.modules if n in ("
        "'clippyme.pipeline.reframe', "
        "'clippyme.pipeline.reframe_detect', "
        "'clippyme.pipeline.framing_plan')]; "
        "leaked += ['pkg.' + a for a in "
        "('reframe', 'reframe_detect', 'framing_plan') "
        "if pkg is not None and hasattr(pkg, a)]; "
        "print('LEAKED:' + ','.join(sorted(leaked))); "
        "sys.exit(1 if leaked else 0)"
    )
    _env = dict(os.environ)
    _env["PYTHONPATH"] = "src" + os.pathsep + _env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _code],
        capture_output=True,
        text=True,
        cwd=_repo_root,
        env=_env,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --- Task 12 review-01 corrections -------------------------------------------
# Production-route aspect coverage (1:1, 16:9), the monotonic analysis
# deadline with injected clock/detector, and behavioral mutation gates.
# The pure halves of these corrections live in test_turn_aware_review1.py.


@pytest.mark.parametrize("aspect_ratio,exp_w,exp_h", [
    (1.0, 360, 360),
    (16 / 9, 640, 360),
])
def test_production_route_non_916_aspects(tmp_path, monkeypatch,
                                          aspect_ratio, exp_w, exp_h):
    """The real two-pass route for 1:1 and 16:9: the plan binds the actual
    render aspect, every persisted crop honors the requested geometry, and
    the output has correct dimensions, duration/FPS/frame count and 880 Hz
    audio. Landmark frames prove the turn is visible in the render."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)

    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto", aspect_ratio=aspect_ratio) is True
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    plan = fp.load_plan(plan_path)
    assert plan.output_aspect == pytest.approx(aspect_ratio)

    # Correct crop geometry: every persisted crop/keyframe matches the
    # requested aspect (int rounding only).
    for sp in plan.scenes:
        for seg in sp.segments:
            if seg.crop:
                x1, y1, x2, y2 = seg.crop
                assert (x2 - x1) / (y2 - y1) == pytest.approx(
                    aspect_ratio, abs=0.05), seg.crop
        for kf in sp.keyframes:
            assert (kf.x2 - kf.x1) / (kf.y2 - kf.y1) == pytest.approx(
                aspect_ratio, abs=0.05)

    # Output media properties.
    ow, oh, ofps, on = _probe(out)
    assert (ow, oh) == (exp_w, exp_h), (ow, oh)
    assert ofps == pytest.approx(30.0)
    assert abs(on - N_FRAMES) <= 2, f"frame count drifted: {on}"
    assert _audio_peak_hz(out, tmp_path) == pytest.approx(880, abs=8)

    # The turn is visible: landmark frames on the A-lock and B-lock differ
    # substantially. Segment ends sit past the eased transition (any ease
    # occupies the segment start). 16:9 cannot fit both speakers in one crop
    # at min zoom, so the turn resolves single(A) -> wide -> single(B).
    by_id = {sp.scene_id: sp for sp in plan.scenes}
    kinds = [(sg.kind, sg.start_frame, sg.subject_ids)
             for sg in by_id[0].segments]
    singles = [sg for sg in by_id[0].segments if sg.kind == "single"]
    assert len(singles) >= 2, kinds
    assert singles[0].subject_ids != singles[-1].subject_ids, kinds
    turn = next(sg for sg in singles[1:]
                if sg.subject_ids != singles[0].subject_ids)
    a_frame = singles[0].end_frame - 1
    b_frame = turn.end_frame - 1
    assert singles[0].start_frame < a_frame <= 119, kinds
    assert turn.start_frame < b_frame <= 119, kinds
    frames = _read_frames(out, [a_frame, b_frame])

    def _mean_diff(a, b):
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()

    assert _mean_diff(frames[a_frame], frames[b_frame]) > 20.0


def test_production_route_square_crop_geometry_pixels(tmp_path, monkeypatch):
    """1:1 render: decoded landmark frames prove the square crop actually
    frames the active speaker (orange left box vs teal right box)."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto", aspect_ratio=1.0) is True
    plan = fp.load_plan(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    by_id = {sp.scene_id: sp for sp in plan.scenes}
    singles = [s for s in by_id[0].segments if s.kind == "single"]
    turn = next(s for s in singles[1:]
                if s.subject_ids != singles[0].subject_ids)
    # Lock frames: segment ends sit past the eased transition.
    # A-side crop [0,56,225,281]: output (110,180) -> src ~(69,169) orange.
    # B-side crop [415,56,640,281]: output (250,180) -> src ~(571,169) teal.
    a_frame = singles[0].end_frame - 1
    b_frame = turn.end_frame - 1
    assert singles[0].start_frame < a_frame <= 119
    assert turn.start_frame < b_frame <= 119
    frames = _read_frames(out, [a_frame, b_frame])
    px = frames[a_frame][180, 110].astype(int)
    assert px[2] > 150 and px[1] > 80 and px[0] < 100, f"not orange: {px}"
    px = frames[b_frame][180, 250].astype(int)
    assert px[0] > 150 and px[1] > 80 and px[2] < 100, f"not teal: {px}"


def test_plan_reuse_rejected_on_aspect_change(tmp_path, monkeypatch, capsys):
    """A 9:16 sidecar is never reused for a 1:1 render: stale-aspect rebuild."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto")
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    mtime1 = os.path.getmtime(plan_path)
    # Production binds the ACTUAL render aspect (202/360 from the
    # even-rounded output width), not the requested 9/16.
    assert fp.load_plan(plan_path).output_aspect == pytest.approx(202 / 360)

    capsys.readouterr()
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto", aspect_ratio=1.0) is True
    logged = capsys.readouterr().out
    assert "stale-aspect" in logged
    assert os.path.getmtime(plan_path) != mtime1, "aspect change must rebuild"
    assert fp.load_plan(plan_path).output_aspect == pytest.approx(1.0)
    ow, oh, _, _ = _probe(out)
    assert (ow, oh) == (360, 360)


def test_mutation_gate_omitted_aspect_wiring(tmp_path, monkeypatch):
    """The production plan build must receive the actual render aspect.

    If the wiring is omitted, the build silently falls back to the 9:16
    default — this gate fails."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    real_build = reframe.build_framing_plan
    seen = {}

    def spy(**kw):
        seen.update(kw)
        return real_build(**kw)

    monkeypatch.setattr(reframe, "build_framing_plan", spy)
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto", aspect_ratio=1.0) is True
    assert seen.get("aspect") == pytest.approx(1.0), \
        "production must thread the render aspect into the plan build"
    plan = fp.load_plan(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    assert plan.output_aspect == pytest.approx(1.0)


def test_mutation_gate_omitted_aspect_wiring_simulated(tmp_path, monkeypatch):
    """Simulate the omission: drop the aspect kwarg in flight and prove the
    persisted plan exposes the 9:16 default — the defect the review found."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    real_build = reframe.build_framing_plan

    def drop_aspect(**kw):
        kw.pop("aspect", None)
        return real_build(**kw)

    monkeypatch.setattr(reframe, "build_framing_plan", drop_aspect)
    assert reframe.process_video_to_vertical(
        src, out, reframe_mode="auto", aspect_ratio=1.0) is True
    plan = fp.load_plan(os.path.splitext(out)[0] + ".turn_framing_plan.json")
    # The bug: a 9:16 plan persisted for a 1:1 render.
    assert plan.output_aspect == pytest.approx(9 / 16)
    assert plan.output_aspect != pytest.approx(1.0)


# --- analysis deadline: injected clock/detector -------------------------------

class _FakeClock:
    """Monotonic clock stand-in; every call advances ``now`` by ``step``."""

    def __init__(self, step=0.005):
        self.now = 1000.0
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class _FakeCapture:
    """cv2.VideoCapture stand-in: ``n_frames`` synthetic frames, then EOF."""

    def __init__(self, n_frames):
        self.remaining = n_frames
        self.released = 0
        self._closed = False

    def isOpened(self):
        return not self._closed

    def _next(self):
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        return True

    def read(self):
        ok = self._next()
        return (ok, object() if ok else None)

    def grab(self):
        return self._next()

    def release(self):
        self.released += 1
        self._closed = True


def _pass1_with_fakes(n_frames=2000, deadline_s=0.05, clock_step=0.005,
                      max_analysis_frames=10 ** 9, strategies=("TRACK",)):
    """Drive ``reframe._pass1_track`` with a fake capture, fake clock and
    counting detector/MAR stand-ins. Returns (result, calls, cap)."""
    clock = _FakeClock(step=clock_step)
    calls = {"detect": 0, "mar": 0, "person": 0}

    def fake_detect(frame):
        calls["detect"] += 1
        return [{"box": [60, 90, 90, 90], "score": 1.0}]

    def fake_mar(frame, box):
        calls["mar"] += 1
        return 0.5

    def fake_person(frame):
        calls["person"] += 1
        return None

    cfg = fp.TurnFramingConfig(analysis_deadline_s=deadline_s,
                               max_analysis_frames=max_analysis_frames)
    tracker = reframe.SpeakerTracker(cooldown_frames=45)
    smoother = reframe.DetectionSmoother(window_size=5)
    cameraman = reframe.SmoothedCameraman(202, 360, 640, 360,
                                          aspect_ratio=9 / 16)
    holder = {}

    def factory(path):
        cap = _FakeCapture(n_frames)
        holder["cap"] = cap
        return cap

    result = reframe._pass1_track(
        "fake.mp4",
        scene_boundaries=[(0, n_frames)], scene_strategies=list(strategies),
        speaker_tracker=tracker, detection_smoother=smoother,
        cameraman=cameraman, turn_cfg=cfg, turn_aware=True, plan_reused=False,
        original_width=640, original_height=360, total_frames=n_frames,
        fps=30.0, clock=clock, detect_fn=fake_detect, mar_fn=fake_mar,
        person_fn=fake_person, capture_factory=factory, show_progress=False,
    )
    return result, calls, holder["cap"]


def test_analysis_deadline_stops_expensive_calls():
    """Subsecond deadline on a long source: detector and MAR call counts
    freeze at exhaustion — they do not merely get discarded afterwards."""
    (targets, scene_ids, strategies, recorder, budget), calls, cap = \
        _pass1_with_fakes(n_frames=2000, deadline_s=0.05, clock_step=0.005)
    # Deterministic: the 5ms clock exhausts the 50ms deadline after exactly
    # two analyzed frames (see the trace in the test docstring below).
    assert calls["detect"] == 2, calls
    assert calls["mar"] == 2, calls
    assert budget.exhausted
    assert budget.exhaust_reason == "analysis_budget_exhausted"
    assert "analysis_budget_exhausted" in recorder.diagnostics
    # Bookkeeping continued for the whole source; capture released once.
    assert len(targets) == 2000
    assert len(scene_ids) == 2000 and len(strategies) == 2000
    assert cap.released == 1


def test_zero_deadline_makes_no_detector_calls():
    """Deadline 0 with the injected clock: not a single detector/MAR call,
    while frame bookkeeping still completes."""
    (targets, _, _, recorder, budget), calls, cap = _pass1_with_fakes(
        n_frames=500, deadline_s=0.0, clock_step=0.0)
    assert calls == {"detect": 0, "mar": 0, "person": 0}, calls
    assert budget.exhaust_reason == "analysis_budget_exhausted"
    assert "analysis_budget_exhausted" in recorder.diagnostics
    assert len(targets) == 500
    assert cap.released == 1


def test_analysis_frame_budget_stops_calls():
    """Tiny max_analysis_frames: the observation cap stops expensive calls
    and records the frame-budget diagnostic."""
    (targets, _, _, recorder, budget), calls, cap = _pass1_with_fakes(
        n_frames=2000, deadline_s=3600.0, clock_step=0.0,
        max_analysis_frames=4)
    # 4 observations max; the log drops on the 5th analyzed frame and the
    # loop stops analyzing right after.
    assert 0 < calls["detect"] <= 8, calls
    assert calls["mar"] <= calls["detect"], calls
    assert "analysis-frame-budget-exhausted" in recorder.diagnostics
    assert len(targets) == 2000
    assert cap.released == 1


def test_mutation_gate_disabled_deadline_checks(monkeypatch):
    """If the budget checks are disabled (ok() forced True), the tiny
    deadline must NOT stop the detector — proving the stop-tests above are
    sensitive to removed checks."""
    monkeypatch.setattr(fp.AnalysisBudget, "ok", lambda self: True)
    (_, _, _, _, budget), calls, _ = _pass1_with_fakes(
        n_frames=200, deadline_s=0.05, clock_step=0.005)
    assert calls["detect"] == 100, calls  # every even frame analyzed
    assert not budget.exhausted


def test_tiny_deadline_completes_render_safely(tmp_path, monkeypatch):
    """A subsecond wall-time budget on the real route: the render completes,
    the detector call count is bounded, the plan persists, and the output is
    a valid clip (affected scenes keep the accepted static/wide fallback)."""
    src = _build_fixture(str(tmp_path / "src.mp4"))
    out = str(tmp_path / "out.mp4")
    _install_fakes(monkeypatch, turn_at=20)
    calls = {"detect": 0}
    real_detect = reframe.detect_face_candidates

    def counting_detect(frame):
        calls["detect"] += 1
        import time as _t
        _t.sleep(0.002)  # keep the tiny deadline deterministic on any host
        return real_detect(frame)

    monkeypatch.setattr(reframe, "detect_face_candidates", counting_detect)
    monkeypatch.setenv("CLIPPYME_TURN_ANALYSIS_DEADLINE_S", "0.05")
    assert reframe.process_video_to_vertical(src, out, reframe_mode="auto") is True
    # 120 even frames could be analyzed; the deadline must stop well short.
    assert 0 < calls["detect"] < 120, calls
    plan_path = os.path.splitext(out)[0] + ".turn_framing_plan.json"
    assert os.path.exists(plan_path)
    assert not list(tmp_path.glob("*.tmp-*")), "no temp leftovers"
    ow, oh, ofps, on = _probe(out)
    assert (ow, oh) == (int(H * 9 / 16), H) and ow % 2 == 0
    assert ofps == pytest.approx(30.0)
    assert abs(on - N_FRAMES) <= 2
    assert _audio_peak_hz(out, tmp_path) == pytest.approx(880, abs=8)
