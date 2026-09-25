"""Pure post-render output QA policy.

``media_qa`` owns ffprobe/ffmpeg I/O; this module turns normalized measurements
into a stable verdict that is cheap to unit-test.  Structural defects are marked
critical so the orchestrator can reject a broken temp render before atomically
publishing it.  Signal-quality findings remain warnings and never destroy a
usable clip.
"""
from __future__ import annotations

import os
import re

DURATION_SHORT_RATIO = 0.25
DURATION_LONG_SLACK = 1.5
MIN_BYTES = 10_000
ASPECT_TOLERANCE = 0.08
BLACK_WARNING_RATIO = 0.45
FREEZE_WARNING_RATIO = 0.70
QUIET_WARNING_DB = -38.0
CLIP_WARNING_DB = -0.1


def evaluate_clip_qa(
    *,
    actual_duration: float | None,
    expected_duration: float | None,
    has_audio: bool,
    size_bytes: int | None,
    smartcut_applied: bool = False,
    has_video: bool = True,
    width: int | None = None,
    height: int | None = None,
    expected_aspect: float | None = None,
    black_ratio: float | None = None,
    freeze_ratio: float | None = None,
    mean_volume_db: float | None = None,
    max_volume_db: float | None = None,
) -> dict:
    """Score a rendered clip against structural and signal expectations.

    Returns ``ok`` for a clean clip, ``critical`` when the file must not replace a
    known-good output, plus separate ``issues`` and ``warnings`` string arrays.
    The historical call shape remains valid because every new argument has a
    conservative default.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if size_bytes is not None and size_bytes < MIN_BYTES:
        issues.append(f"output is effectively empty ({size_bytes} bytes)")
    if not has_video:
        issues.append("output has no video stream")
    if not has_audio:
        issues.append("output has no audio stream")
    if actual_duration is not None and actual_duration <= 0:
        issues.append("output duration is zero")

    if (
        actual_duration is not None
        and expected_duration is not None
        and expected_duration > 0
        and actual_duration > 0
    ):
        ratio = actual_duration / expected_duration
        if ratio > DURATION_LONG_SLACK:
            issues.append(
                f"output ({actual_duration:.1f}s) far longer than expected "
                f"({expected_duration:.1f}s)"
            )
        if ratio < DURATION_SHORT_RATIO and not smartcut_applied:
            issues.append(
                f"output ({actual_duration:.1f}s) far shorter than expected "
                f"({expected_duration:.1f}s)"
            )

    if width and height and expected_aspect and expected_aspect > 0:
        actual_aspect = float(width) / float(height)
        relative_error = abs(actual_aspect - expected_aspect) / expected_aspect
        if relative_error > ASPECT_TOLERANCE:
            issues.append(
                f"wrong aspect ratio ({width}x{height}, {actual_aspect:.3f}; "
                f"expected {expected_aspect:.3f})"
            )

    if black_ratio is not None and black_ratio > BLACK_WARNING_RATIO:
        warnings.append(f"large full-frame black share ({black_ratio * 100:.1f}%)")
    if freeze_ratio is not None and freeze_ratio > FREEZE_WARNING_RATIO:
        warnings.append(f"long frozen-frame share ({freeze_ratio * 100:.1f}%)")
    if mean_volume_db is not None and mean_volume_db < QUIET_WARNING_DB:
        warnings.append(f"audio is very quiet (mean {mean_volume_db:.1f} dB)")
    if max_volume_db is not None and max_volume_db > CLIP_WARNING_DB:
        warnings.append(f"audio peak may clip ({max_volume_db:.2f} dB)")

    return {
        "ok": not issues and not warnings,
        "critical": bool(issues),
        "issues": issues,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Subtitle-presence validation (Phase 1C, OpusClip-level upgrade)
# ---------------------------------------------------------------------------

#: A caption file that survived generation must carry at least this many
#: dialogue events. (The genuine "no speech" case never reaches the burn —
#: compose skips the subtitle layer — so a burn attempt with fewer events
#: means the file is effectively empty.)
MIN_SUBTITLE_EVENTS = 1

#: |delta-gray| floor for counting a caption-band pixel as "changed"
#: between the pre-burn and post-burn frames. Burned karaoke text flips
#: pixels hard (bright text + dark outline over the scene); plain re-encode
#: noise stays well under this.
_SUB_CHANGE_THRESHOLD = 25.0
#: Minimum fraction of caption-band pixels that must *change* between the
#: pre-burn and post-burn frames (median over the sampled frames).
#: Calibrated 2026-09-24 on real clips: burned 0.0041-0.0106,
#: re-encoded-without-subs 0.0000 (see _opusclip_verify/phase1 evidence).
#: The 0.0015 floor keeps ~3x margin under the weakest observed burn while
#: staying far above encode noise.
_SUB_CHANGE_MIN = 0.0015


def _ass_time_to_seconds(value: str):
    """Parse an ASS timestamp (H:MM:SS.cc) to seconds; None if malformed."""
    try:
        h, m, rest = value.strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(rest)
    except (ValueError, AttributeError):
        return None


def _srt_time_to_seconds(value: str):
    """Parse an SRT timestamp (HH:MM:SS,mmm) to seconds; None if malformed."""
    try:
        hms, ms = value.strip().replace(".", ",").split(",")
        h, m, sec = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
    except (ValueError, AttributeError):
        return None


def _parse_subtitle_events(sub_path: str) -> list:
    """Return ``[(start_s, end_s)]`` dialogue events from a .ass or .srt file."""
    events = []
    try:
        with open(sub_path, encoding="utf-8-sig") as handle:
            content = handle.read()
    except OSError:
        return events
    if sub_path.lower().endswith(".ass"):
        for line in content.splitlines():
            if not line.startswith("Dialogue:"):
                continue
            fields = line[len("Dialogue:"):].split(",", 9)
            if len(fields) < 10:
                continue
            start = _ass_time_to_seconds(fields[1])
            end = _ass_time_to_seconds(fields[2])
            if start is not None and end is not None and end > start:
                events.append((start, end))
    else:  # .srt
        for block in re.split(r"\n\s*\n", content):
            match = re.search(
                r"(\d+:\d+:\d+[,.]\d+)\s*-->\s*(\d+:\d+:\d+[,.]\d+)", block
            )
            if not match:
                continue
            start = _srt_time_to_seconds(match.group(1))
            end = _srt_time_to_seconds(match.group(2))
            if start is not None and end is not None and end > start:
                events.append((start, end))
    return events


def _caption_band(width: int, height: int, position: str) -> tuple:
    """Pixel box ``(x0, y0, x1, y1)`` where captions are expected."""
    pos = str(position or "bottom").lower()
    if pos == "top":
        y0, y1 = 0.03, 0.32
    elif pos == "center":
        y0, y1 = 0.38, 0.62
    else:  # bottom (default)
        y0, y1 = 0.70, 0.97
    return (
        int(width * 0.06), int(height * y0),
        int(width * 0.94), int(height * y1),
    )


def _grab_frame(video_path: str, t: float):
    """Best-effort single-frame grab at ``t`` seconds (None on failure).

    Uses ffmpeg with the seek placed AFTER ``-i`` so the returned frame is
    decode-accurate. cv2.VideoCapture's POS_MSEC seek only lands on
    keyframes, which can be seconds off the requested time — comparing the
    before/after videos would then compare different scene content and the
    subtitle-burn check would report nonsense gains.
    """
    import shutil  # lazy: keeps this module import-light for unit tests
    import subprocess
    import tempfile

    import cv2  # lazy: see above

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    try:
        ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{max(0.0, t):.3f}", "-i", video_path,
            "-frames:v", "1", tmp.name,
        ]
        subprocess.run(cmd, check=True)
        return cv2.imread(tmp.name)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _band_change_fraction(before, after, band: tuple) -> float:
    """Fraction of caption-band pixels that changed significantly.

    ``before``/``after`` are BGR frames of the same scene (decode-accurate
    seeks, so scene content matches); burned captions are the only thing
    that should flip pixels hard inside the band. Note this deliberately
    measures *change*, not absolute edge density: on highly textured scenes
    the burn's re-encode pass smooths the background and can even *reduce*
    naive edge density, which made an edge-gain metric report negative gains
    on real burned clips.
    """
    import cv2  # lazy: see _grab_frame
    import numpy as np

    x0, y0, x1, y1 = band
    b = before[y0:y1, x0:x1]
    a = after[y0:y1, x0:x1]
    if b.size == 0 or a.size == 0:
        return 0.0
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(ga - gb) > _SUB_CHANGE_THRESHOLD))


def check_subtitles_burned(
    sub_path: str,
    before_video: str,
    after_video: str,
    *,
    position: str = "bottom",
    num_samples: int = 3,
) -> dict:
    """Validate that captions were actually burned into the video pixels.

    Two layers, matching the Phase 1C spec:

    1. **File check** — the ``.ass``/``.srt`` sent to the burn must exist,
       be non-empty, and carry at least ``MIN_SUBTITLE_EVENTS`` dialogue
       events. Missing/empty here is CRITICAL (captions were requested but
       the file that was "burned" never existed).
    2. **Pixel check** — sample up to ``num_samples`` frames at caption
       timestamps (spread across the clip) from both the pre-burn and
       post-burn videos and measure the fraction of caption-band pixels
       that changed significantly. Burned karaoke text flips pixels hard;
       a silently failed burn only re-encodes, which changes ~nothing.
       Median change fraction below ``_SUB_CHANGE_MIN`` is CRITICAL.

    Returns the same ``{"ok", "critical", "issues", "warnings"}`` shape as
    :func:`evaluate_clip_qa`, plus a ``"detail"`` dict with the measured
    median change fraction for debugging.
    """
    issues: list = []
    warnings: list = []

    def _verdict(ok: bool, critical: bool) -> dict:
        return {"ok": ok, "critical": critical, "issues": issues,
                "warnings": warnings}

    if not sub_path or not os.path.isfile(sub_path):
        issues.append(
            f"subtitle file missing: {sub_path!r} "
            "(captions were requested but no caption file exists)"
        )
        return _verdict(False, True)
    try:
        file_size = os.path.getsize(sub_path)
    except OSError:
        file_size = 0
    if file_size == 0:
        issues.append("subtitle file is empty (0 bytes) — nothing was burned")
        return _verdict(False, True)

    events = _parse_subtitle_events(sub_path)
    if len(events) < MIN_SUBTITLE_EVENTS:
        issues.append(
            f"subtitle file has {len(events)} dialogue event(s) "
            f"(< {MIN_SUBTITLE_EVENTS}) — effectively empty"
        )
        return _verdict(False, True)

    # Sample frames spread across the clip: first / middle / last event.
    spread = sorted({0, len(events) // 2, len(events) - 1})[: max(1, num_samples)]
    changes = []
    for idx in spread:
        start, end = events[idx]
        t = (start + end) / 2.0
        before = _grab_frame(before_video, t)
        after = _grab_frame(after_video, t)
        if before is None or after is None:
            warnings.append(f"could not sample frame at {t:.1f}s — skipped")
            continue
        if before.shape != after.shape:
            warnings.append(
                f"frame size mismatch at {t:.1f}s "
                f"({before.shape} vs {after.shape}) — skipped"
            )
            continue
        band = _caption_band(after.shape[1], after.shape[0], position)
        changes.append(_band_change_fraction(before, after, band))

    if not changes:
        # Frame sampling failed entirely: fail OPEN with a loud warning (an
        # infra problem, not evidence of missing captions), so a decoder
        # hiccup can never nuke a good clip.
        warnings.append(
            "subtitle pixel check inconclusive — no caption frames could be "
            "sampled; file check passed"
        )
        return _verdict(True, False)

    median_change = sorted(changes)[len(changes) // 2]
    detail = {
        "median_change_fraction": round(median_change, 5),
        "samples": len(changes),
        "events": len(events),
    }
    if median_change < _SUB_CHANGE_MIN:
        issues.append(
            f"no captions visibly burned: caption-band change fraction "
            f"{median_change:.4f} < {_SUB_CHANGE_MIN} over {len(changes)} "
            "sampled frame(s)"
        )
        result = _verdict(False, True)
        result["detail"] = detail
        return result

    result = _verdict(True, False)
    result["detail"] = detail
    return result
