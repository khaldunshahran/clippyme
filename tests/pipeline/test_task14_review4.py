"""Task 14 review-04: explicit fit must contain the source at every output aspect.

Promotes coordinator-review4-remaining.py (3 cases) and adds:
- real AV square/landscape tests with distinct corner/edge landmarks
  (containment, FPS/frame count, duration, 880 Hz audio, source hash,
  sidecar reuse),
- load-bearing mutations: restoring width-only fit must fail, and
  producing black output must fail,
- pixel-identity of the new contain-fit with the legacy width-fit
  where the width-fit already fits (9:16 case).
"""
import importlib.util
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "_t14r4_render_fixture",
    _repo / "tests" / "pipeline" / "test_action_framing_render.py",
)
f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f)

np = f.np
cv2 = f.cv2


# ---------------------------------------------------------------------------
# Promoted coordinator probes (verbatim logic, fixture import path adapted)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('source_w,source_h,out_w,out_h', [
    (640, 480, 854, 480), (360, 640, 480, 480)])
def test_fit_contains_whole_source_when_output_is_wider(source_w, source_h,
                                                        out_w, out_h):
    frame = np.zeros((source_h, source_w, 3), dtype=np.uint8)
    frame[10:60, 10:60] = (0, 255, 0)
    frame[-60:-10, -60:-10] = (255, 0, 0)
    result = f.reframe.create_general_fit_frame(frame, out_w, out_h)
    assert result.shape == (out_h, out_w, 3)
    assert f._green_present(result, thresh=100)
    assert f._blue_present(result, thresh=100)


def test_real_explicit_fit_landscape_render(tmp_path, monkeypatch):
    src = f._build_src(tmp_path / 'src.mp4', tmp_path)
    before = f._sha(src)
    f._install(monkeypatch, detect_fn=lambda frame: [
        {'box': [200., 180., 90., 120.], 'confidence': .1, 'category': 'object'}],
        observe_fn=lambda *args: ([.1], [.5], 0.))
    output = str(tmp_path / 'landscape.mp4')
    assert f.reframe.process_video_to_vertical(src, output, reframe_mode='auto',
                                              aspect_ratio=16 / 9)
    w, h, fps, n = f.TestActionRenderEndToEnd._dims(output)
    assert (w, h) == (854, 480)  # production rounds canvas width up to even
    assert fps == pytest.approx(30.)
    assert abs(n - 120) <= 2
    assert f._sha(src) == before
    assert f._tone_hz(output) == pytest.approx(880, abs=40)
    frames = f._read_frames(output, [30, 90])
    assert f._green_present(frames[30]), \
        'landscape fit export lost scene0 landmark (black fallback)'
    assert f._blue_present(frames[90]), \
        'landscape fit export lost scene1 landmark (black fallback)'


# ---------------------------------------------------------------------------
# Square AV test with distinct corner/edge landmarks + reuse
# ---------------------------------------------------------------------------

SQ_W, SQ_H = 360, 640
SQ_FRAMES = 120
SQ_CUT = 60


def _square_frame(i):
    img = np.zeros((SQ_H, SQ_W, 3), dtype=np.uint8)
    if i < SQ_CUT:
        cv2.rectangle(img, (10, 10), (70, 70), (0, 255, 0), -1)      # TL green
        cv2.rectangle(img, (10, 300), (40, 335), (0, 0, 255), -1)     # left red
    else:
        cv2.rectangle(img, (SQ_W - 70, SQ_H - 70),
                      (SQ_W - 10, SQ_H - 10), (255, 0, 0), -1)        # BR blue
        cv2.rectangle(img, (SQ_W - 40, 300), (SQ_W - 10, 335),
                      (0, 0, 255), -1)                                # right red
    return img


def _red_present(frame, thresh=400):
    r = frame[:, :, 2].astype(int)
    g = frame[:, :, 1].astype(int)
    b = frame[:, :, 0].astype(int)
    return int((((r > 90) & (r > g + 20) & (r > b + 20)).sum())) > thresh


def _build_square_src(path, tmp_path):
    silent = str(tmp_path / "sq_silent.mp4")
    vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), f.FPS,
                         (SQ_W, SQ_H))
    for i in range(SQ_FRAMES):
        vw.write(_square_frame(i))
    vw.release()
    dur = SQ_FRAMES / f.FPS
    cmd = ["ffmpeg", "-y", "-i", silent,
           "-f", "lavfi", "-i", f"sine=frequency=880:duration={dur}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", str(path)]
    r = f.subprocess.run(cmd, capture_output=True, check=True)
    assert r.returncode == 0
    return str(path)


def _weak_install(monkeypatch):
    return f._install(
        monkeypatch,
        detect_fn=lambda frame: [
            {'box': [200., 180., 90., 120.], 'confidence': .1,
             'category': 'object'}],
        observe_fn=lambda *args: ([.1], [.5], 0.))


def test_real_explicit_fit_square_render_and_reuse(tmp_path, monkeypatch):
    src = _build_square_src(tmp_path / 'sq_src.mp4', tmp_path)
    before = f._sha(src)
    calls = _weak_install(monkeypatch)
    output = str(tmp_path / 'square.mp4')
    assert f.reframe.process_video_to_vertical(src, output, reframe_mode='auto',
                                              aspect_ratio=1.0)
    w, h, fps, n = f.TestActionRenderEndToEnd._dims(output)
    # Production derives OUTPUT_HEIGHT from the source height (640) and
    # OUTPUT_WIDTH = int(640 * 1.0): exact requested square geometry.
    assert (w, h) == (640, 640)
    assert fps == pytest.approx(30.)
    assert abs(n - SQ_FRAMES) <= 2
    assert f._sha(src) == before
    assert f._tone_hz(output) == pytest.approx(880, abs=40)
    frames = f._read_frames(output, [30, 90])
    assert f._green_present(frames[30]), 'square fit lost TL green landmark'
    assert _red_present(frames[30]), 'square fit lost left-edge red landmark'
    assert f._blue_present(frames[90]), 'square fit lost BR blue landmark'
    assert _red_present(frames[90]), 'square fit lost right-edge red landmark'
    # Sidecar reuse: no redetection, landmarks still contained.
    calls["n"] = 0
    assert f.reframe.process_video_to_vertical(src, output, reframe_mode='auto',
                                              aspect_ratio=1.0)
    assert calls["n"] == 0, "reused plan must not redetect"
    frames2 = f._read_frames(output, [30, 90])
    assert f._green_present(frames2[30])
    assert f._blue_present(frames2[90])


# ---------------------------------------------------------------------------
# Pixel identity: where legacy width-fit already fits, contain-fit is identical
# ---------------------------------------------------------------------------

def test_vertical_output_pixel_identical_to_legacy_width_fit():
    frame = (np.arange(480 * 640 * 3, dtype=np.uint8) % 251
             ).reshape(480, 640, 3)
    new = f.reframe.create_general_fit_frame(frame, 270, 480)
    # Legacy width-only construction for this geometry (width-limited).
    oh2, ow2 = 480, 270
    orig_h, orig_w = frame.shape[:2]
    bg_scale = oh2 / orig_h
    bg_w = int(orig_w * bg_scale)
    bg = cv2.resize(frame, (bg_w, oh2))
    sx = (bg_w - ow2) // 2
    if sx < 0:
        sx = 0
    background = bg[:, sx:sx + ow2]
    if background.shape[1] != ow2:
        background = cv2.resize(background, (ow2, oh2))
    background = cv2.GaussianBlur(background, (51, 51), 0)
    scale = ow2 / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (ow2, fg_h))
    y_off = (oh2 - fg_h) // 2
    old = background.copy()
    old[y_off:y_off + fg_h, :] = foreground
    assert np.array_equal(new, old)


# ---------------------------------------------------------------------------
# Load-bearing mutations
# ---------------------------------------------------------------------------

def _old_width_only_fit(frame, output_width, output_height):
    """The pre-fix implementation: width-only foreground scale."""
    orig_h, orig_w = frame.shape[:2]
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height))
    start_x = (bg_w - output_width) // 2
    if start_x < 0:
        start_x = 0
    background = bg_resized[:, start_x:start_x + output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
    background = cv2.GaussianBlur(background, (51, 51), 0)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h))
    y_offset = (output_height - fg_h) // 2
    final_frame = background.copy()
    final_frame[y_offset:y_offset + fg_h, :] = foreground
    return final_frame


def test_mutation_width_only_fit_fails(monkeypatch):
    """Restoring the width-only foreground must raise on wider outputs."""
    monkeypatch.setattr(f.reframe, "create_general_fit_frame",
                        _old_width_only_fit)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        f.reframe.create_general_fit_frame(frame, 854, 480)
    frame2 = np.zeros((640, 360, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        f.reframe.create_general_fit_frame(frame2, 480, 480)


def test_mutation_black_output_fails(tmp_path, monkeypatch):
    """A fit renderer emitting black frames must fail landmark checks."""
    monkeypatch.setattr(
        f.reframe, "create_general_fit_frame",
        lambda frame, ow, oh: np.zeros((oh, ow, 3), dtype=np.uint8))
    src = f._build_src(tmp_path / 'src.mp4', tmp_path)
    _weak_install(monkeypatch)
    output = str(tmp_path / 'black.mp4')
    assert f.reframe.process_video_to_vertical(src, output, reframe_mode='auto',
                                              aspect_ratio=16 / 9)
    frames = f._read_frames(output, [30, 90])
    # The real gate asserts presence; with black output both are absent,
    # proving the production test catches black-frame regression.
    assert not f._green_present(frames[30])
    assert not f._blue_present(frames[90])
