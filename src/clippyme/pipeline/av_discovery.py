"""Bounded audiovisual discovery (Task 11) — pure, deterministic, no provider calls.

This module closes the discovery gap left by Tasks 08-10: source time with
little or no speech (reactions, reveals, action, demonstrations, applause,
music/performance beats). It provides:

* a deterministic media-window planner that tiles ``[0, source_duration)``
  with absolute timestamps, coalescing with transcript windows where
  sensible, so the whole source is represented even with no valid words;
* an injectable frame-sampling seam (production default shells out to
  ffmpeg; tests inject fakes and never decode real media);
* audio-event normalization with strict type/time validation;
* opaque local evidence IDs (frames ``vf_<window>_<seq>``, events
  ``ae_<n>``) and local candidate ID / moment fingerprint generation from
  source identity, interval, evidence basis and stable evidence refs —
  never provider IDs;
* strict evidence validation for the provider contract (see
  ``gemini_parser.parse_av_discovery_payload``);
* a source fingerprint helper used to gate checkpoint/resume reuse.

Honest limits (see ``docs/audiovisual-discovery.md``): frames are sparse
observations at listed timestamps, never continuous visual coverage. This
module does not implement active-speaker framing, camera trajectories,
recall/editorial-quality proof on real footage, or OpusClip parity.

Everything here is pure stdlib except the production frame sampler, which
shells out to ffmpeg (already a repo dependency, see ``media_probe.py``).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

AV_DISCOVERY_VERSION = 1

# Evidence bases the provider may declare. "transcript" is the Task 08
# contract; the other three are the Task 11 audiovisual additions.
EVIDENCE_BASES = ("transcript", "visual", "audio_event", "multimodal")

# Defaults. Bounded for ~1h sources; every knob is env-overridable.
AV_WINDOW_TARGET_S = 240.0
AV_WINDOW_OVERLAP_S = 45.0
AV_MAX_MEDIA_WINDOWS = 16
AV_MAX_TOTAL_SAMPLED_FRAMES = 64
AV_FRAMES_PER_WINDOW = 3
AV_FRAME_MAX_WIDTH = 320
AV_FRAME_MAX_PIXELS = 102400
AV_MAX_FRAME_BYTES = 65536
AV_MAX_WINDOW_BYTES = 262144
AV_MAX_INVOCATION_BYTES = 4 * 1024 * 1024
AV_SAMPLING_TIMEOUT_S = 30.0
AV_FINGERPRINT_PROBE_BYTES = 65536
AV_MAX_AUDIO_EVENT_LABEL_CHARS = 64
AV_MAX_VISUAL_DESCRIPTION_CHARS = 500
AV_MAX_MANIFEST_FRAMES = 12
AV_MAX_MANIFEST_EVENTS = 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.getenv(name) or default)
    except (ValueError, TypeError):
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(name) or default)
    except (ValueError, TypeError):
        v = default
    return max(lo, min(hi, v))


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def av_discovery_enabled() -> bool:
    """Master flag; default ON on this branch.

    The audiovisual path additionally requires a readable source file
    (see ``source_media_usable``): with no readable source, or with this
    flag off, discovery runs the accepted Task 10 transcript behavior
    with zero media sampling and no audiovisual provider parts.
    """
    return _env_flag("CLIPPYME_AV_DISCOVERY", True)


def source_media_usable(source_path: Optional[str]) -> bool:
    """True when the path names a readable regular file."""
    if not source_path or not isinstance(source_path, str):
        return False
    try:
        return os.path.isfile(source_path) and os.access(
            source_path, os.R_OK)
    except OSError:
        return False


@dataclass
class AVDiscoveryConfig:
    """Validated, bounded audiovisual discovery configuration (env-driven)."""

    enabled: bool = True
    media_window_target_s: float = AV_WINDOW_TARGET_S
    media_window_overlap_s: float = AV_WINDOW_OVERLAP_S
    max_media_windows: int = AV_MAX_MEDIA_WINDOWS
    max_total_sampled_frames: int = AV_MAX_TOTAL_SAMPLED_FRAMES
    frames_per_window: int = AV_FRAMES_PER_WINDOW
    frame_max_width: int = AV_FRAME_MAX_WIDTH
    frame_max_pixels: int = AV_FRAME_MAX_PIXELS
    max_frame_bytes: int = AV_MAX_FRAME_BYTES
    max_window_bytes: int = AV_MAX_WINDOW_BYTES
    max_invocation_bytes: int = AV_MAX_INVOCATION_BYTES
    sampling_timeout_s: float = AV_SAMPLING_TIMEOUT_S
    fingerprint_probe_bytes: int = AV_FINGERPRINT_PROBE_BYTES
    max_audio_event_label_chars: int = AV_MAX_AUDIO_EVENT_LABEL_CHARS
    max_visual_description_chars: int = AV_MAX_VISUAL_DESCRIPTION_CHARS
    max_manifest_frames: int = AV_MAX_MANIFEST_FRAMES
    max_manifest_events: int = AV_MAX_MANIFEST_EVENTS

    @classmethod
    def from_env(cls) -> "AVDiscoveryConfig":
        target = _env_float("CLIPPYME_AV_WINDOW_TARGET_S",
                            AV_WINDOW_TARGET_S, 30.0, 900.0)
        overlap = _env_float("CLIPPYME_AV_WINDOW_OVERLAP_S",
                             AV_WINDOW_OVERLAP_S, 0.0, target * 0.9)
        return cls(
            enabled=av_discovery_enabled(),
            media_window_target_s=target,
            media_window_overlap_s=overlap,
            max_media_windows=_env_int("CLIPPYME_AV_MAX_WINDOWS",
                                       AV_MAX_MEDIA_WINDOWS, 1, 128),
            max_total_sampled_frames=_env_int(
                "CLIPPYME_AV_MAX_TOTAL_FRAMES",
                AV_MAX_TOTAL_SAMPLED_FRAMES, 1, 4096),
            frames_per_window=_env_int("CLIPPYME_AV_FRAMES_PER_WINDOW",
                                       AV_FRAMES_PER_WINDOW, 1, 12),
            frame_max_width=_env_int("CLIPPYME_AV_FRAME_MAX_WIDTH",
                                     AV_FRAME_MAX_WIDTH, 64, 1280),
            frame_max_pixels=_env_int("CLIPPYME_AV_FRAME_MAX_PIXELS",
                                      AV_FRAME_MAX_PIXELS, 4096, 2073600),
            max_frame_bytes=_env_int("CLIPPYME_AV_MAX_FRAME_BYTES",
                                     AV_MAX_FRAME_BYTES, 4096, 1048576),
            max_window_bytes=_env_int("CLIPPYME_AV_MAX_WINDOW_BYTES",
                                      AV_MAX_WINDOW_BYTES, 16384, 16777216),
            max_invocation_bytes=_env_int(
                "CLIPPYME_AV_MAX_INVOCATION_BYTES",
                AV_MAX_INVOCATION_BYTES, 65536, 134217728),
            sampling_timeout_s=_env_float("CLIPPYME_AV_SAMPLING_TIMEOUT_S",
                                          AV_SAMPLING_TIMEOUT_S, 5.0, 300.0),
            fingerprint_probe_bytes=_env_int(
                "CLIPPYME_AV_FINGERPRINT_PROBE_BYTES",
                AV_FINGERPRINT_PROBE_BYTES, 4096, 1048576),
            max_audio_event_label_chars=_env_int(
                "CLIPPYME_AV_MAX_EVENT_LABEL_CHARS",
                AV_MAX_AUDIO_EVENT_LABEL_CHARS, 8, 256),
            max_visual_description_chars=_env_int(
                "CLIPPYME_AV_MAX_VISUAL_DESC_CHARS",
                AV_MAX_VISUAL_DESCRIPTION_CHARS, 64, 2000),
            max_manifest_frames=_env_int("CLIPPYME_AV_MAX_MANIFEST_FRAMES",
                                         AV_MAX_MANIFEST_FRAMES, 1, 64),
            max_manifest_events=_env_int("CLIPPYME_AV_MAX_MANIFEST_EVENTS",
                                         AV_MAX_MANIFEST_EVENTS, 1, 100),
        )


@dataclass(frozen=True)
class FrameLimits:
    """Byte/pixel/time caps enforced by the frame sampler."""

    max_frame_bytes: int = AV_MAX_FRAME_BYTES
    max_frame_width: int = AV_FRAME_MAX_WIDTH
    max_frame_pixels: int = AV_FRAME_MAX_PIXELS
    max_window_bytes: int = AV_MAX_WINDOW_BYTES
    max_invocation_bytes: int = AV_MAX_INVOCATION_BYTES
    sampling_timeout_s: float = AV_SAMPLING_TIMEOUT_S
    jpeg_quality_steps: Tuple[int, ...] = (5, 12, 20)

    @classmethod
    def from_config(cls, config: AVDiscoveryConfig) -> "FrameLimits":
        return cls(
            max_frame_bytes=config.max_frame_bytes,
            max_frame_width=config.frame_max_width,
            max_frame_pixels=config.frame_max_pixels,
            max_window_bytes=config.max_window_bytes,
            max_invocation_bytes=config.max_invocation_bytes,
            sampling_timeout_s=config.sampling_timeout_s,
        )


# ---------------------------------------------------------------------------
# Source identity
# ---------------------------------------------------------------------------

class SourceIdentityError(OSError):
    """The source file could not be fingerprinted (missing/unreadable)."""


def source_identity_details(source_path: str,
                            probe_bytes: int = AV_FINGERPRINT_PROBE_BYTES
                            ) -> Dict[str, Any]:
    """Return the documented source-identity record for checkpoint gating.

    Format: ``fingerprint = "src-" + sha256(first probe_bytes ||
    size_bytes || mtime_ns).hexdigest()[:16]``. Any change to the file's
    head bytes, size, or mtime invalidates the fingerprint, so a resumed
    record is reusable only when the fingerprint (plus audiovisual
    config/version, window plan, and evidence manifest identities) still
    matches.
    """
    if not source_path or not isinstance(source_path, str):
        raise SourceIdentityError("source path is empty")
    try:
        st = os.stat(source_path)
    except OSError as e:
        raise SourceIdentityError(
            f"source not stat-able: {source_path}: {e}") from e
    if not os.path.isfile(source_path):
        raise SourceIdentityError(
            f"source is not a regular file: {source_path}")
    try:
        with open(source_path, "rb") as fh:
            head = fh.read(probe_bytes)
    except OSError as e:
        raise SourceIdentityError(
            f"source not readable: {source_path}: {e}") from e
    h = hashlib.sha256()
    h.update(head)
    h.update(b"|")
    h.update(str(st.st_size).encode("utf-8"))
    h.update(b"|")
    h.update(str(st.st_mtime_ns).encode("utf-8"))
    return {
        "fingerprint": "src-" + h.hexdigest()[:16],
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "probe_bytes": probe_bytes,
    }


def fingerprint_source(source_path: str,
                       probe_bytes: int = AV_FINGERPRINT_PROBE_BYTES
                       ) -> str:
    """Opaque source fingerprint string; see ``source_identity_details``."""
    return source_identity_details(source_path, probe_bytes)["fingerprint"]


# ---------------------------------------------------------------------------
# Media-window planning
# ---------------------------------------------------------------------------

@dataclass
class MediaWindow:
    """One planned media window with absolute source timestamps.

    ``kind`` is "transcript" when the window coalesces with >= 1 Task 08
    transcript window (their indices are recorded) and "media" otherwise.
    Media windows are their own planned entities even when coalesced:
    the interval below is the media plan, not the transcript plan.
    """

    index: int
    start_s: float
    end_s: float
    kind: str  # "media" | "transcript"
    merged_transcript_window_idxs: Tuple[int, ...] = ()
    planned_frame_count: int = AV_FRAMES_PER_WINDOW


def _as_interval(value: Any) -> Optional[Tuple[float, float]]:
    """Coerce a transcript-window-like value to (start, end) or None."""
    start = end = None
    if isinstance(value, dict):
        start = value.get("start_s", value.get("start"))
        end = value.get("end_s", value.get("end"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value[0], value[1]
    else:
        start = getattr(value, "start_s", None)
        end = getattr(value, "end_s", None)
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    if not (isinstance(start, (int, float))
            and isinstance(end, (int, float))):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if end <= start:
        return None
    return float(start), float(end)


def _is_finite_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def plan_media_windows(
    source_duration: float,
    transcript_windows: Sequence[Any],
    config: Optional[AVDiscoveryConfig] = None,
) -> Dict[str, Any]:
    """Deterministic whole-source media-window plan.

    Tiles ``[0, source_duration)`` with target-length overlapping windows
    using absolute source timestamps. Transcript windows that fall inside
    a media window are coalesced (recorded, kind "transcript"); all other
    media windows are kind "media". The whole source is represented even
    when ``transcript_windows`` is empty.

    Caps: ``max_media_windows`` and ``max_total_sampled_frames`` leave
    trailing windows unattempted; their intervals are reported explicitly
    in ``unattempted_windows`` with exact reasons ("media_window_cap",
    "frame_budget_cap"). This is the planning-time distinction; a
    deadline that later skips windows is reported by the orchestrator
    per window ("planned" vs "provider-attempted" vs sampled-frame
    timestamps are separate lifecycle stages, never conflated here).

    Returns {"windows": [MediaWindow...], "unattempted_windows": [...],
    "coverage": {...}, "transcript_windows_seen": int}.
    """
    cfg = config or AVDiscoveryConfig.from_env()
    if not _is_finite_number(source_duration) or source_duration <= 0:
        raise ValueError(
            f"invalid_source_duration: {source_duration!r}")

    # Normalize transcript windows; clamp to source bounds; ignore junk.
    tw: List[Tuple[float, float]] = []
    ignored = 0
    for entry in transcript_windows or []:
        iv = _as_interval(entry)
        if iv is None:
            ignored += 1
            continue
        s, e = iv
        s = max(0.0, min(s, source_duration))
        e = max(0.0, min(e, source_duration))
        if e > s:
            tw.append((s, e))
        else:
            ignored += 1

    target = cfg.media_window_target_s
    step = target - cfg.media_window_overlap_s
    if step <= 0:
        step = target

    # Deterministic tiling of [0, source_duration). A tiny tail (< half
    # a step) is absorbed into the previous window instead of becoming a
    # sliver window.
    raw: List[Tuple[float, float]] = []
    start = 0.0
    while start < source_duration - 1e-9:
        end = min(start + target, source_duration)
        if end < source_duration and source_duration - end < step / 2.0:
            end = source_duration
        raw.append((start, end))
        if end >= source_duration - 1e-9:
            break
        start += step

    windows: List[MediaWindow] = []
    unattempted: List[Dict[str, Any]] = []
    frames_budget_used = 0
    for i, (ws, we) in enumerate(raw):
        merged = tuple(
            idx for idx, (ts, te) in enumerate(tw)
            if ts >= ws - 1e-6 and te <= we + 1e-6)
        kind = "transcript" if merged else "media"
        if i >= cfg.max_media_windows:
            unattempted.append({
                "window_index": i, "start_s": ws, "end_s": we,
                "reason": "media_window_cap",
            })
            continue
        if (frames_budget_used + cfg.frames_per_window
                > cfg.max_total_sampled_frames):
            unattempted.append({
                "window_index": i, "start_s": ws, "end_s": we,
                "reason": "frame_budget_cap",
            })
            continue
        frames_budget_used += cfg.frames_per_window
        windows.append(MediaWindow(
            index=i, start_s=ws, end_s=we, kind=kind,
            merged_transcript_window_idxs=merged,
            planned_frame_count=cfg.frames_per_window))

    # Coverage: union of attempted windows must cover [0, duration).
    covered: List[Tuple[float, float]] = [
        (w.start_s, w.end_s) for w in windows]
    uncovered: List[List[float]] = []
    cursor = 0.0
    for ws, we in sorted(covered):
        if ws > cursor + 1e-6:
            uncovered.append([cursor, min(ws, source_duration)])
        cursor = max(cursor, we)
    if cursor < source_duration - 1e-6:
        uncovered.append([cursor, source_duration])

    return {
        "windows": windows,
        "unattempted_windows": unattempted,
        "coverage": {
            "source_duration_s": source_duration,
            "windows_planned": len(windows),
            "media_only_windows": sum(
                1 for w in windows if w.kind == "media"),
            "transcript_merged_windows": sum(
                1 for w in windows if w.kind == "transcript"),
            "unattempted_windows": len(unattempted),
            "frames_planned": frames_budget_used,
            "full_coverage": not uncovered,
            "uncovered_ranges": uncovered,
        },
        "transcript_windows_seen": len(tw),
        "transcript_windows_ignored": ignored,
    }


def window_frame_timestamps(window: MediaWindow,
                            n_frames: int) -> List[float]:
    """Deterministic absolute sample timestamps inside a media window.

    Centered samples: for n frames, ``start + (i + 0.5) * length / n``.
    Absolute source timestamps, rounded to milliseconds, strictly inside
    ``[start_s, end_s]``. Never rebased: callers must persist these exact
    values; a rebased (window-relative) timestamp fails validation.
    """
    if n_frames <= 0:
        return []
    length = window.end_s - window.start_s
    if length <= 0:
        return []
    out = []
    for i in range(n_frames):
        t = window.start_s + (i + 0.5) * length / n_frames
        t = min(max(t, window.start_s), window.end_s)
        out.append(round(t, 3))
    # De-duplicate while preserving order (tiny windows).
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ---------------------------------------------------------------------------
# Frame-sampling seam
# ---------------------------------------------------------------------------

class FrameSample(NamedTuple):
    """One sampled frame: (requested_ts, decoded_ts, frame_bytes).

    Timestamps are absolute source seconds. ``frame_bytes`` is the
    encoded (JPEG) frame; it is transient request material and must
    never be persisted in job metadata or checkpoints.
    """
    requested_ts: float
    decoded_ts: float
    frame_bytes: bytes


class SamplingError(Exception):
    """A frame/window could not be sampled within the documented caps."""


# Injectable seam protocol: any callable with this signature works.
# Production default is ``ffmpeg_frame_sampler``; tests inject fakes.
FrameSampler = Callable[
    [str, Sequence[float], Optional[FrameLimits]], List[FrameSample]]


_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")


def ffmpeg_frame_sampler(
    source_path: str,
    timestamps: Sequence[float],
    limits: Optional[FrameLimits] = None,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> List[FrameSample]:
    """Production frame sampler: decode frames with ffmpeg, downscale, MJPEG.

    For each absolute timestamp: fast-seek (``-ss`` before ``-i`` with
    ``-copyts`` so the decoder reports absolute presentation
    timestamps), decode one frame, downscale with a filter chain
    enforcing ``max_frame_width`` and ``max_frame_pixels``, encode MJPEG
    with a quality ladder (``jpeg_quality_steps``) until under
    ``max_frame_bytes``. The decoded presentation timestamp is read from
    ``showinfo`` stderr and is authoritative — it is recorded, never
    assumed equal to the request (on non-intra seeks it may precede the
    requested timestamp slightly; that is reported honestly).

    The source is opened read-only; it is never modified. No temp files:
    frame bytes are captured from stdout in memory. Raises
    ``SamplingError`` on unreadable source, invalid timestamps, decode
    failure, missing decoded timestamps, or cap violations. Raises
    ``FileNotFoundError`` when the ffmpeg binary is missing.
    """
    lim = limits or FrameLimits()
    if not source_media_usable(source_path):
        raise SamplingError(f"source_unreadable: {source_path!r}")
    ts_list = list(timestamps or [])
    for t in ts_list:
        if not _is_finite_number(t) or t < 0:
            raise SamplingError(f"invalid_timestamp: {t!r}")

    scale_filter = (
        f"scale=w='min(iw,{lim.max_frame_width})':h=-2,"
        f"scale=w='iw*min(1,sqrt({lim.max_frame_pixels}/(iw*ih)))':"
        f"h='ih*min(1,sqrt({lim.max_frame_pixels}/(iw*ih)))',"
        "showinfo"
    )
    out: List[FrameSample] = []
    for t in ts_list:
        frame_bytes: Optional[bytes] = None
        decoded_ts: Optional[float] = None
        last_err = ""
        for q in lim.jpeg_quality_steps:
            cmd = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "info",
                "-ss", f"{t:.3f}", "-copyts", "-i", source_path,
                "-frames:v", "1", "-vf", scale_filter,
                "-f", "image2", "-vcodec", "mjpeg", "-q:v", str(q),
                "-",
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True,
                    timeout=lim.sampling_timeout_s)
            except FileNotFoundError:
                raise
            except subprocess.TimeoutExpired:
                last_err = "ffmpeg_timeout"
                break
            except OSError as e:
                last_err = f"ffmpeg_spawn_failed: {e}"
                break
            if proc.returncode != 0 or not proc.stdout:
                last_err = (
                    "frame_decode_failed: "
                    + (proc.stderr or b"")[-300:].decode(
                        "utf-8", "replace"))
                break
            m = _PTS_TIME_RE.search(
                (proc.stderr or b"").decode("utf-8", "replace"))
            if m:
                try:
                    decoded_ts = float(m.group(1))
                except ValueError:
                    decoded_ts = None
            if decoded_ts is None or not math.isfinite(decoded_ts):
                last_err = "decoded_timestamp_unavailable"
                break
            if len(proc.stdout) <= lim.max_frame_bytes:
                frame_bytes = proc.stdout
                break
            last_err = (f"frame_bytes_over_cap: {len(proc.stdout)} > "
                        f"{lim.max_frame_bytes}")
            # Try the next (lower-quality) step.
        if frame_bytes is None or decoded_ts is None:
            raise SamplingError(
                f"sample_failed ts={t:.3f}: {last_err}")
        out.append(FrameSample(
            requested_ts=float(t), decoded_ts=decoded_ts,
            frame_bytes=frame_bytes))
    return out


@dataclass
class WindowSample:
    """Result of sampling one media window.

    ``frames`` holds ``SampledFrame`` records; frame bytes live here
    transiently for request construction only. ``metadata()`` strips the
    bytes for job metadata/checkpoints (timestamps and counts only).
    """

    window_index: int
    status: str  # "ok" | "failed" | "skipped"
    frames: List["SampledFrame"] = field(default_factory=list)
    total_bytes: int = 0
    reason: str = ""
    error: str = ""

    def metadata(self) -> Dict[str, Any]:
        return {
            "window_index": self.window_index,
            "status": self.status,
            "reason": self.reason,
            "error": self.error[:200] if self.error else "",
            "frame_count": len(self.frames),
            "total_bytes": self.total_bytes,
            "frames": [
                {"frame_id": f.frame_id,
                 "requested_ts": f.requested_ts,
                 "decoded_ts": f.decoded_ts,
                 "byte_size": len(f.frame_bytes)}
                for f in self.frames
            ],
        }


@dataclass
class SampledFrame:
    """One sampled frame with its opaque local ID."""

    frame_id: str  # vf_<window_index>_<seq>
    requested_ts: float  # absolute source seconds, as planned
    decoded_ts: float  # absolute source seconds, from the decoder
    frame_bytes: bytes  # transient; never persisted in job metadata
    mime_type: str = "image/jpeg"


def make_frame_id(window_index: int, seq: int) -> str:
    """Opaque local frame ID: ``vf_<window_index>_<seq>``."""
    return f"vf_{int(window_index)}_{int(seq)}"


def sample_media_window(
    source_path: str,
    window: MediaWindow,
    sampler: FrameSampler,
    limits: Optional[FrameLimits] = None,
    n_frames: Optional[int] = None,
) -> WindowSample:
    """Sample one media window through the injectable sampler seam.

    Plans deterministic absolute timestamps, calls the sampler, assigns
    opaque local frame IDs, records requested + decoded timestamps, and
    enforces the per-window byte cap. Any failure (sampler exception,
    contract violation, decoded timestamp outside the window, window
    byte cap) yields status "failed" with an exact reason — the caller
    falls back to transcript-only evidence for that window and reports
    it; other windows are unaffected.

    Temp-file hygiene: neither this function nor the production sampler
    writes temp files; if a custom sampler does, it must clean up after
    itself (documented seam contract).
    """
    lim = limits or FrameLimits()
    count = window.planned_frame_count if n_frames is None else n_frames
    timestamps = window_frame_timestamps(window, count)
    if not timestamps:
        return WindowSample(window_index=window.index, status="skipped",
                            reason="no_timestamps_planned")
    try:
        samples = sampler(source_path, timestamps, lim)
    except FileNotFoundError:
        raise
    except Exception as e:
        return WindowSample(window_index=window.index, status="failed",
                            reason="sampler_error",
                            error=f"{type(e).__name__}: {e}"[:300])

    # Seam contract check: one FrameSample per requested timestamp, in
    # order, with matching requested_ts.
    if not isinstance(samples, list) or len(samples) != len(timestamps):
        return WindowSample(
            window_index=window.index, status="failed",
            reason="sampler_contract_violation",
            error=(f"expected {len(timestamps)} samples, got "
                   f"{len(samples) if isinstance(samples, list) else '?'}"))
    frames: List[SampledFrame] = []
    total = 0
    tol = 1e-6
    for seq, (sample, want_ts) in enumerate(zip(samples, timestamps)):
        try:
            rts = float(sample.requested_ts)
            dts = float(sample.decoded_ts)
            fbytes = bytes(sample.frame_bytes)
        except (TypeError, ValueError, AttributeError) as e:
            return WindowSample(
                window_index=window.index, status="failed",
                reason="sampler_contract_violation",
                error=f"bad sample shape at seq {seq}: {e}"[:200])
        if abs(rts - want_ts) > 1e-3:
            return WindowSample(
                window_index=window.index, status="failed",
                reason="sampler_contract_violation",
                error=(f"requested_ts mismatch at seq {seq}: "
                       f"{rts} != {want_ts}"))
        if not math.isfinite(dts) or dts < 0:
            return WindowSample(
                window_index=window.index, status="failed",
                reason="decoded_timestamp_invalid",
                error=f"seq {seq}: decoded_ts={dts!r}")
        if not (window.start_s - tol <= dts <= window.end_s + tol):
            # A rebased (window-relative) or wrong-window timestamp is a
            # hard failure: absolute source timestamps are the contract.
            return WindowSample(
                window_index=window.index, status="failed",
                reason="decoded_timestamp_out_of_window",
                error=(f"seq {seq}: decoded_ts={dts:.3f} outside "
                       f"[{window.start_s:.3f}, {window.end_s:.3f}]"))
        if len(fbytes) > lim.max_frame_bytes:
            return WindowSample(
                window_index=window.index, status="failed",
                reason="frame_bytes_over_cap",
                error=f"seq {seq}: {len(fbytes)} > {lim.max_frame_bytes}")
        total += len(fbytes)
        frames.append(SampledFrame(
            frame_id=make_frame_id(window.index, seq),
            requested_ts=rts, decoded_ts=dts, frame_bytes=fbytes))
    if total > lim.max_window_bytes:
        return WindowSample(
            window_index=window.index, status="failed",
            reason="window_bytes_over_cap",
            error=f"{total} > {lim.max_window_bytes}")
    return WindowSample(window_index=window.index, status="ok",
                        frames=frames, total_bytes=total)


class SamplingBudget:
    """Local sampling budget: frames, bytes, and a sampling deadline.

    Pure helper for the orchestration layer; mirrors the
    ``DiscoveryBudget`` shape from Task 08 without sharing state.
    ``deadline_s`` anchors ONE sampling deadline across frames/retries:
    production passes the shared discovery deadline so sampling never
    outlives it; when omitted, the historical sampling-timeout scale is
    kept for direct unit use.
    """

    def __init__(self, config: AVDiscoveryConfig,
                 time_fn: Callable[[], float] = time.monotonic,
                 deadline_s: Optional[float] = None):
        self.config = config
        self.time_fn = time_fn
        self.start = time_fn()
        self.deadline_s = (float(deadline_s)
                           if deadline_s is not None
                           else config.sampling_timeout_s * 10)
        self.frames_used = 0
        self.bytes_used = 0

    def remaining_s(self) -> float:
        return max(0.0, self.deadline_s - (self.time_fn() - self.start))

    def exhausted(self) -> bool:
        return (self.frames_used >= self.config.max_total_sampled_frames
                or self.bytes_used >= self.config.max_invocation_bytes
                or self.remaining_s() <= 0)

    def charge(self, frames: int, byte_count: int) -> bool:
        """Charge frames/bytes against the invocation caps.

        Strict: declines (returns False) BEFORE a charge would exceed
        ``max_total_sampled_frames`` or ``max_invocation_bytes``;
        ``frames_used``/``bytes_used`` are left unchanged on decline.
        A charge is never partially applied — the caller must not send
        declined evidence.
        """
        if frames < 0 or byte_count < 0:
            return False
        if (self.frames_used + frames
                > self.config.max_total_sampled_frames):
            return False
        if (self.bytes_used + byte_count
                > self.config.max_invocation_bytes):
            return False
        self.frames_used += frames
        self.bytes_used += byte_count
        return True


# ---------------------------------------------------------------------------
# Audio-event normalization
# ---------------------------------------------------------------------------

def make_audio_event_id(seq: int, prefix: str = "ae") -> str:
    """Opaque local audio-event ID: ``<prefix>_<seq>`` (default ``ae_<n>``)."""
    return f"{prefix}_{int(seq)}"


def normalize_audio_events(
    raw_events: Optional[Sequence[Any]],
    source_duration: float,
    id_prefix: str = "ae",
    max_label_chars: int = AV_MAX_AUDIO_EVENT_LABEL_CHARS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Normalize transcript ``audio_events``; return (events, issues).

    An event is valid only when its label is a bounded nonempty string
    and start/end are finite, non-boolean, ordered, and within source
    bounds. Valid events get opaque local IDs (``ae_<n>`` by default).
    Invalid events are isolated as issue records, never raised on.

    Never infers a sound event from loudness alone: there is no
    loudness input here at all — only explicitly supplied events.
    """
    events: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    if not raw_events:
        return events, issues
    if not _is_finite_number(source_duration) or source_duration <= 0:
        raise ValueError(
            f"invalid_source_duration: {source_duration!r}")
    seq = 0
    for idx, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            issues.append({"index": idx, "reason": "event_not_object"})
            continue
        # ElevenLabs emits audio events as {text, start, end}; accept the
        # "text" label shape alongside label/name/type.
        label = raw.get("label", raw.get("name", raw.get("type",
                                                        raw.get("text"))))
        if (not isinstance(label, str) or isinstance(label, bool)
                or not label.strip()
                or len(label.strip()) > max_label_chars):
            issues.append({"index": idx, "reason": "invalid_event_label"})
            continue
        start = raw.get("start_s", raw.get("start"))
        end = raw.get("end_s", raw.get("end"))
        if isinstance(start, bool) or isinstance(end, bool):
            issues.append({"index": idx, "reason": "boolean_event_time"})
            continue
        if not (_is_finite_number(start) and _is_finite_number(end)):
            issues.append({"index": idx, "reason": "invalid_event_time"})
            continue
        start_f, end_f = float(start), float(end)
        if not end_f > start_f:
            issues.append({"index": idx, "reason": "event_time_reversed"})
            continue
        if not (0.0 <= start_f and end_f <= source_duration + 1e-6):
            issues.append(
                {"index": idx, "reason": "event_out_of_source_bounds"})
            continue
        events.append({
            "event_id": make_audio_event_id(seq, id_prefix),
            "label": label.strip(),
            "start_s": start_f,
            "end_s": end_f,
        })
        seq += 1
    return events, issues


# ---------------------------------------------------------------------------
# Local candidate identity (never provider IDs)
# ---------------------------------------------------------------------------

def _normalize_evidence_text(text: str) -> str:
    # Mirrors the Task 08 normalization (candidate_discovery): lowercase,
    # collapse whitespace, strip punctuation. Kept local so this module
    # has no import dependency on candidate_discovery.
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _evidence_ref_key(evidence_refs: Sequence[Any]) -> str:
    parts = []
    for r in evidence_refs or []:
        if isinstance(r, (list, tuple)):
            parts.append(":".join(str(x) for x in r))
        else:
            parts.append(str(r))
    return "|".join(sorted(parts))


def make_av_moment_fingerprint(
    source_fingerprint: str,
    start_s: float,
    end_s: float,
    evidence_basis: str,
    evidence_refs: Sequence[Any],
    evidence_text: str = "",
) -> str:
    """Stable moment fingerprint across windows; for semantic dedupe.

    Built from canonical ABSOLUTE evidence identity: source identity,
    interval, evidence basis, normalized model observation / transcript
    evidence, and stable absolute evidence references — ``("frame",
    "<decoded_ts:.3f>")`` for visual frames, ``("audio", "<label>",
    "<start_s:.3f>", "<end_s:.3f>")`` for audio events. Window-local IDs
    (``vf_0_0``, ``ae_w0_0``) must NEVER be hashed here: overlapping
    windows assign different local IDs to the same absolute evidence,
    which would make genuine cross-window repeats impossible to
    collapse. Two windows observing the same moment with the same
    absolute evidence share this fingerprint; different model
    observations sharing time remain distinct.
    """
    h = hashlib.sha256()
    h.update(str(source_fingerprint).encode("utf-8"))
    h.update(f"{float(start_s):.3f}-{float(end_s):.3f}".encode("utf-8"))
    h.update(str(evidence_basis or "").encode("utf-8"))
    h.update(_evidence_ref_key(evidence_refs).encode("utf-8"))
    h.update(_normalize_evidence_text(evidence_text).encode("utf-8"))
    return "avmoment-" + h.hexdigest()[:16]


def make_av_candidate_id(
    source_fingerprint: str,
    start_s: float,
    end_s: float,
    evidence_basis: str,
    evidence_refs: Sequence[Any],
    window_index: int = 0,
    evidence_text: str = "",
) -> str:
    """Distinct trace ID per window occurrence; never model-provided.

    Includes ``window_index`` so the same moment in overlapping windows
    gets distinct IDs. Use ``make_av_moment_fingerprint`` for the stable
    dedupe key.
    """
    h = hashlib.sha256()
    h.update(str(source_fingerprint).encode("utf-8"))
    h.update(f"w{int(window_index)}:".encode("utf-8"))
    h.update(f"{float(start_s):.3f}-{float(end_s):.3f}".encode("utf-8"))
    h.update(str(evidence_basis or "").encode("utf-8"))
    h.update(_evidence_ref_key(evidence_refs).encode("utf-8"))
    h.update(_normalize_evidence_text(evidence_text).encode("utf-8"))
    return "avcand-" + h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Strict evidence validation
# ---------------------------------------------------------------------------

class EvidenceValidationError(Exception):
    """One provider item failed the strict evidence contract.

    ``reason`` is the exact structured rejection reason (stable string
    for funnel accounting); ``detail`` carries human context.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class ValidatedAVCandidate:
    """One validated audiovisual discovery candidate."""

    candidate_id: str
    moment_fingerprint: str
    window_index: int
    start_s: float
    end_s: float
    evidence_basis: str
    evidence_text: str
    visual_description: Optional[Dict[str, Any]]  # {"text", "model_produced"}
    visual_frame_ids: Tuple[str, ...]
    audio_event_ids: Tuple[str, ...]
    context: str
    expected_payoff: str
    event_type: str
    confidence: float
    recommendation_score: float
    speaker_ids: List[Any]
    event_key: Optional[str] = None


def _word_text(w: Dict[str, Any]) -> str:
    for k in ("w", "text", "word"):
        t = w.get(k)
        if isinstance(t, str) and t.strip():
            return t
        if t is not None and not isinstance(t, bool):
            s = str(t).strip()
            if s:
                return s
    return ""


def _word_start(w: Dict[str, Any]) -> Optional[float]:
    for k in ("s", "start", "start_time", "start_s"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _word_end(w: Dict[str, Any]) -> Optional[float]:
    for k in ("e", "end", "end_time", "end_s"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _word_speaker(w: Dict[str, Any]) -> Optional[int]:
    for k in ("sp", "speaker"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
    return None


def _grounded_transcript_evidence(evidence: str,
                                 window_words: Sequence[Dict[str, Any]],
                                 start: float, end: float) -> bool:
    """Contiguous token-slice grounding (strict Task 11 rule).

    The normalized evidence token sequence must appear as a CONTIGUOUS
    slice of complete normalized interval tokens. Character substrings
    ("state" inside "statement") do not count; token boundaries are
    respected. Unlike Task 08 there is no ordered-subsequence fallback:
    audiovisual transcript evidence must be an exact contiguous quote.
    """
    ev_tokens = _normalize_evidence_text(evidence).split()
    if not ev_tokens:
        return False
    interval_tokens: List[str] = []
    for w in window_words:
        if not isinstance(w, dict):
            continue
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end and we > start:
            wt = _normalize_evidence_text(_word_text(w))
            if wt:
                interval_tokens.extend(wt.split())
    n = len(ev_tokens)
    if n > len(interval_tokens):
        return False
    return any(interval_tokens[i:i + n] == ev_tokens
               for i in range(len(interval_tokens) - n + 1))


def _coerce_id_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise EvidenceValidationError(
            f"invalid_{field}_type",
            f"{field} must be a list of local IDs")
    out = []
    for v in value:
        if not isinstance(v, str) or not v.strip():
            raise EvidenceValidationError(
                f"invalid_{field}_entry",
                f"{field} entries must be nonempty strings")
        out.append(v.strip())
    return out


def validate_av_evidence(
    item: Any,
    *,
    window_start_s: float,
    window_end_s: float,
    window_index: int = 0,
    source_duration: Optional[float] = None,
    frames_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    audio_events_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    window_words: Sequence[Dict[str, Any]] = (),
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    source_fingerprint: str = "src",
    max_visual_description_chars: int = AV_MAX_VISUAL_DESCRIPTION_CHARS,
) -> ValidatedAVCandidate:
    """Validate one audiovisual provider item; raise on any violation.

    ``frames_by_id`` is scoped to THIS window: a visual reference must
    name a supplied frame from the same window. ``audio_events_by_id``
    holds validated (normalized) events; each referenced event must
    overlap the candidate interval. ``min_duration``/``max_duration``
    are the resolved Task 04 policy bounds — this function applies
    them; it does not re-resolve the policy (use
    ``duration_policy.resolve_duration_policy`` upstream).

    Rejection reasons are exact stable strings (see docs).

    Visual grounding (review-01): every referenced frame's finite
    authoritative ``decoded_ts`` must lie inside the candidate interval
    under the documented half-open boundary
    ``start_s <= decoded_ts < end_s`` — reason
    ``visual_frame_outside_candidate:<frame_id>`` otherwise. Whenever a
    candidate uses visual frame evidence (basis ``visual``, or
    ``multimodal`` with frame refs), a bounded nonempty model-produced
    ``visual_description`` is required — reason
    ``missing_visual_description`` when absent.
    """
    if not isinstance(item, dict):
        raise EvidenceValidationError("candidate_not_object")
    frames = frames_by_id or {}
    events = audio_events_by_id or {}

    # --- interval -----------------------------------------------------
    start = item.get("start_s", item.get("start"))
    end = item.get("end_s", item.get("end"))
    if start is None or end is None:
        raise EvidenceValidationError("missing_start_end")
    if isinstance(start, bool) or isinstance(end, bool):
        raise EvidenceValidationError("boolean_time",
                                      "start/end must not be booleans")
    if not (_is_finite_number(start) and _is_finite_number(end)):
        raise EvidenceValidationError("non_finite_time",
                                      "start/end must be finite numbers")
    start_f, end_f = float(start), float(end)
    if not end_f > start_f:
        raise EvidenceValidationError("interval_reversed")
    tol = 1e-6
    if not (window_start_s - tol <= start_f
            and end_f <= window_end_s + tol):
        raise EvidenceValidationError(
            "interval_out_of_window",
            f"[{start_f:.3f}, {end_f:.3f}] outside window "
            f"[{window_start_s:.3f}, {window_end_s:.3f}]")
    if source_duration is not None and source_duration > 0:
        if not (0.0 - tol <= start_f
                and end_f <= source_duration + tol):
            raise EvidenceValidationError(
                "interval_out_of_source",
                f"[{start_f:.3f}, {end_f:.3f}] outside source "
                f"[0, {source_duration:.3f}]")
    # Resolved Task 04 duration policy bounds (applied, not re-resolved).
    dur = end_f - start_f
    if min_duration is not None and dur < min_duration - 1e-6:
        raise EvidenceValidationError(
            "duration_below_minimum",
            f"{dur:.1f}s < minimum {min_duration:.1f}s")
    if max_duration is not None and dur > max_duration + 1e-6:
        raise EvidenceValidationError(
            "duration_above_maximum",
            f"{dur:.1f}s > maximum {max_duration:.1f}s")

    # --- text fields --------------------------------------------------
    context = item.get("context") or ""
    payoff = item.get("expected_payoff") or item.get("payoff") or ""
    if not str(context).strip():
        raise EvidenceValidationError("empty_context")
    if not str(payoff).strip():
        raise EvidenceValidationError("empty_expected_payoff")

    # --- evidence basis ------------------------------------------------
    basis = str(item.get("evidence_basis") or "").strip().lower()
    if basis not in EVIDENCE_BASES:
        raise EvidenceValidationError(
            "unsupported_evidence_basis",
            f"{item.get('evidence_basis')!r} not in {EVIDENCE_BASES}")

    # --- references ----------------------------------------------------
    visual_ids = _coerce_id_list(
        item.get("visual_frame_ids"), "visual_frame_ids")
    audio_ids = _coerce_id_list(
        item.get("audio_event_ids"), "audio_event_ids")
    for fid in visual_ids:
        if fid not in frames:
            # Scoped to this window: a foreign-window ID is unknown here.
            raise EvidenceValidationError(
                f"unknown_frame_id:{fid}",
                "visual reference must name a supplied frame from the "
                "same window")
    for fid in visual_ids:
        dts = frames[fid].get("decoded_ts")
        if not _is_finite_number(dts):
            raise EvidenceValidationError(
                f"visual_frame_outside_candidate:{fid}",
                "referenced frame has no finite authoritative decoded "
                "timestamp")
        dts_f = float(dts)
        # Documented half-open boundary: the frame's authoritative
        # decoded timestamp must satisfy start_s <= decoded_ts < end_s.
        # A same-window frame decoded outside the proposed clip cannot
        # ground it.
        if not (start_f <= dts_f < end_f):
            raise EvidenceValidationError(
                f"visual_frame_outside_candidate:{fid}",
                f"frame decoded at {dts_f:.3f}s lies outside the "
                f"candidate [{start_f:.3f}, {end_f:.3f})")
    for eid in audio_ids:
        ev = events.get(eid)
        if ev is None:
            raise EvidenceValidationError(
                f"unknown_audio_event_id:{eid}",
                "audio reference must name a validated audio event")
        es, ee = float(ev["start_s"]), float(ev["end_s"])
        if not (es < end_f and ee > start_f):
            raise EvidenceValidationError(
                f"audio_event_not_overlapping:{eid}",
                f"event [{es:.3f}, {ee:.3f}] does not overlap candidate "
                f"[{start_f:.3f}, {end_f:.3f}]")

    # --- basis/reference agreement -------------------------------------
    evidence = item.get("evidence_text") or item.get("evidence") or ""
    evidence_s = str(evidence).strip()
    visual_desc = item.get("visual_description")
    visual_desc_rec: Optional[Dict[str, Any]] = None
    if visual_desc is not None:
        vd = str(visual_desc).strip()
        if not vd:
            raise EvidenceValidationError("empty_visual_description")
        if len(vd) > max_visual_description_chars:
            raise EvidenceValidationError(
                "visual_description_too_long",
                f"{len(vd)} > {max_visual_description_chars}")
        # Labeled as model-produced, never verified fact.
        visual_desc_rec = {"text": vd, "model_produced": True}

    has_words = False
    for w in window_words:
        if not isinstance(w, dict):
            continue
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end_f and we > start_f and _word_text(w).strip():
            has_words = True
            break

    has_words = False
    for w in window_words:
        if not isinstance(w, dict):
            continue
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end_f and we > start_f and _word_text(w).strip():
            has_words = True
            break

    # --- wordless eligibility ------------------------------------------
    # A wordless range is eligible only with >= 1 valid visual or
    # audio-event reference. This gate runs before the per-basis
    # reference checks so a range with neither words nor media
    # evidence gets the precise reason (rather than, e.g.,
    # missing_visual_reference when the real problem is the range).
    if not has_words and not (visual_ids or audio_ids):
        raise EvidenceValidationError(
            "wordless_without_evidence",
            "wordless range needs >= 1 valid visual or audio-event "
            "reference")

    if basis == "transcript":
        if visual_ids:
            raise EvidenceValidationError("unexpected_visual_reference",
                                          "transcript basis takes no frames")
        if audio_ids:
            raise EvidenceValidationError("unexpected_audio_reference",
                                          "transcript basis takes no events")
        if not evidence_s:
            raise EvidenceValidationError("missing_evidence_text")
        if not _grounded_transcript_evidence(
                evidence_s, window_words, start_f, end_f):
            raise EvidenceValidationError(
                "evidence_not_grounded",
                f"{evidence_s[:60]!r} is not a contiguous token slice "
                "of the candidate's words")
    elif basis == "visual":
        if not visual_ids:
            raise EvidenceValidationError(
                "missing_visual_reference",
                "visual basis requires >= 1 supplied frame")
        if audio_ids:
            raise EvidenceValidationError("unexpected_audio_reference",
                                          "visual basis takes no events")
        if visual_desc_rec is None:
            # A candidate that uses visual frame evidence must carry
            # the model-produced observation: the store must never
            # hold bare frame IDs with no auditable description.
            raise EvidenceValidationError(
                "missing_visual_description",
                "visual frame evidence requires a bounded nonempty "
                "model-produced visual_description")
    elif basis == "audio_event":
        if not audio_ids:
            raise EvidenceValidationError(
                "missing_audio_reference",
                "audio_event basis requires >= 1 overlapping event")
        if visual_ids:
            raise EvidenceValidationError("unexpected_visual_reference",
                                          "audio_event basis takes no frames")
    else:  # multimodal
        if not evidence_s or not _grounded_transcript_evidence(
                evidence_s, window_words, start_f, end_f):
            raise EvidenceValidationError(
                "multimodal_missing_transcript_evidence",
                "multimodal basis requires grounded transcript evidence")
        if not (visual_ids or audio_ids):
            raise EvidenceValidationError(
                "multimodal_missing_media_reference",
                "multimodal basis requires >= 1 visual or audio reference")
        if visual_ids and visual_desc_rec is None:
            raise EvidenceValidationError(
                "missing_visual_description",
                "visual frame evidence requires a bounded nonempty "
                "model-produced visual_description")

    # --- confidence / score --------------------------------------------
    # (wordless eligibility was decided above, before basis agreement)
    confidence = item.get("confidence", 0.5)
    score = item.get("recommendation_score",
                     item.get("viral_score", item.get("score", 50)))
    if isinstance(confidence, bool) or not _is_finite_number(confidence):
        raise EvidenceValidationError("non_finite_confidence")
    if isinstance(score, bool) or not _is_finite_number(score):
        raise EvidenceValidationError("non_finite_score")
    confidence_f, score_f = float(confidence), float(score)
    if not 0.0 <= confidence_f <= 1.0:
        raise EvidenceValidationError("confidence_out_of_range")
    if not 1.0 <= score_f <= 100.0:
        raise EvidenceValidationError("score_out_of_range")

    event_key = item.get("event_key") or item.get("continuity_key")
    if event_key is not None and not str(event_key).strip():
        event_key = None
    speakers: List[Any] = []
    for w in window_words:
        if not isinstance(w, dict):
            continue
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end_f and we > start_f:
            sid = _word_speaker(w)
            if sid is not None and sid not in speakers:
                speakers.append(sid)

    # Local identity from source identity, interval, evidence basis,
    # and stable evidence references — never provider IDs. A provider
    # "id"/"candidate_id" field is ignored on purpose. The occurrence
    # (candidate) ID stays window-specific; the moment fingerprint is
    # built from canonical ABSOLUTE evidence identity (stable decoded
    # timestamps for frames, label/start/end for audio events — never
    # window-local IDs), plus the normalized model observation, so the
    # same absolute evidence in overlapping windows shares one
    # fingerprint while different observations stay distinct.
    stable_refs: List[Tuple[str, str]] = (
        [("vf", fid) for fid in visual_ids]
        + [("ae", eid) for eid in audio_ids])
    absolute_refs: List[Tuple[str, ...]] = []
    for fid in visual_ids:
        absolute_refs.append(
            ("frame", f"{float(frames[fid]['decoded_ts']):.3f}"))
    for eid in audio_ids:
        _ev = events[eid]
        absolute_refs.append(
            ("audio", str(_ev.get("label", "")),
             f"{float(_ev['start_s']):.3f}",
             f"{float(_ev['end_s']):.3f}"))
    observation = visual_desc_rec["text"] if visual_desc_rec else ""
    moment_evidence = "\n".join(
        part for part in (evidence_s, observation) if part)
    cid = make_av_candidate_id(
        source_fingerprint, start_f, end_f, basis, stable_refs,
        window_index, evidence_s)
    mfp = make_av_moment_fingerprint(
        source_fingerprint, start_f, end_f, basis, absolute_refs,
        moment_evidence)
    return ValidatedAVCandidate(
        candidate_id=cid,
        moment_fingerprint=mfp,
        window_index=window_index,
        start_s=start_f, end_s=end_f,
        evidence_basis=basis,
        evidence_text=evidence_s,
        visual_description=visual_desc_rec,
        visual_frame_ids=tuple(visual_ids),
        audio_event_ids=tuple(audio_ids),
        context=str(context).strip(),
        expected_payoff=str(payoff).strip(),
        event_type=str(item.get("event_type") or "moment"),
        confidence=confidence_f,
        recommendation_score=score_f,
        speaker_ids=speakers,
        event_key=str(event_key).strip() if event_key else None,
    )


__all__ = [
    "AV_DISCOVERY_VERSION",
    "EVIDENCE_BASES",
    "AVDiscoveryConfig",
    "FrameLimits",
    "MediaWindow",
    "FrameSample",
    "SampledFrame",
    "WindowSample",
    "SamplingBudget",
    "SamplingError",
    "SourceIdentityError",
    "EvidenceValidationError",
    "ValidatedAVCandidate",
    "FrameSampler",
    "av_discovery_enabled",
    "source_media_usable",
    "source_identity_details",
    "fingerprint_source",
    "plan_media_windows",
    "window_frame_timestamps",
    "ffmpeg_frame_sampler",
    "sample_media_window",
    "make_frame_id",
    "make_audio_event_id",
    "normalize_audio_events",
    "make_av_candidate_id",
    "make_av_moment_fingerprint",
    "validate_av_evidence",
]
