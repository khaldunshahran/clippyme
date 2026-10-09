"""Checkpointed backend entrypoint for the ClippyMe video pipeline.

The historical :mod:`clippyme.pipeline.main` remains the owner of the heavy CV,
transcription and Gemini functions. This module imports those functions and
replaces only the CLI orchestration used by queued backend jobs, adding atomic
phase checkpoints, per-clip resume, preflight, QA and operational progress.
"""
from __future__ import annotations

import sys
if sys.stdout is not None:
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr is not None:
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from clippyme.domain.runtime_state import RuntimeState
from clippyme.pipeline.media_qa import inspect_clip, probe_media
from clippyme.pipeline.project_emit import emit_clip_project, project_has_user_edits
from clippyme.pipeline.preflight import (
    PreflightInputs,
    PreflightRejected,
    build_preflight,
    enforce_preflight,
    format_preflight_log,
)
from clippyme.pipeline.reframe_ops import normalize_letterbox_zoom
from clippyme.pipeline.run_ops import (
    build_cut_command,
    clip_output_basename,
    find_source_video_candidate,
    is_same_video_source,
    resolve_output_dir,
    sanitize_windows_basename,
    should_use_fallback,
    slice_provenance_valid,
    source_fingerprint,
    validate_cut_output,
    write_slice_provenance,
)

logger = logging.getLogger("clippyme")
_FALSE_VALUES = {"0", "false", "no", "off"}


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pipeline-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        for attempt in range(10):
            try:
                if os.name == "nt" and os.path.exists(path):
                    try:
                        import stat
                        os.chmod(path, stat.S_IWRITE)
                    except OSError:
                        pass
                os.replace(tmp, path)
                tmp = ""
                break
            except (PermissionError, OSError):
                if attempt == 9:
                    raise
                time.sleep(0.05 * (1.5 ** attempt))
        if tmp == "" and os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _safe_output_artifact(path: str | None, output_dir: str) -> str | None:
    """Accept checkpoint paths only when contained by this job directory."""
    if not path:
        return None
    try:
        candidate = Path(path).resolve()
        root = Path(output_dir).resolve()
        candidate.relative_to(root)
    except (OSError, ValueError):
        return None
    return str(candidate)


def _valid_file(path: str | None, minimum: int = 1) -> bool:
    try:
        return bool(path and os.path.isfile(path) and os.path.getsize(path) >= minimum)
    except OSError:
        return False


def _valid_media_file(path: str | None, minimum: int = 1) -> bool:
    if not _valid_file(path, minimum):
        return False
    try:
        import subprocess, json
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return False
        data = json.loads(proc.stdout)
        dur = float(data.get("format", {}).get("duration", 0))
        if dur <= 0:
            return False
        if not any(s.get("codec_type") == "video" for s in data.get("streams", [])):
            return False
        return True
    except Exception:
        return False


def _enabled(name: str, default: str = "1") -> bool:
    return str(os.getenv(name, default)).strip().lower() not in _FALSE_VALUES


def _nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default)) or default))
    except (TypeError, ValueError):
        return max(0, int(default))


def _source_fingerprint(path: str) -> str | None:
    """Cheap stable identity: size + first/last MiB, not a full multi-GB hash."""
    try:
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        digest.update(str(size).encode("ascii"))
        with open(path, "rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if size > 1024 * 1024:
                handle.seek(max(0, size - 1024 * 1024))
                digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()
    except OSError:
        return None


def _safe_clip_filename(value: str | None) -> str | None:
    if not value or "/" in value or "\\" in value or os.path.basename(value) != value:
        return None
    if value.startswith(".") or not value.lower().endswith(".mp4"):
        return None
    return value


def _save_metadata(path: str, data: dict[str, Any]) -> None:
    _atomic_json(path, data)
    print(f"   💾 checkpoint metadata: {os.path.basename(path)}", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ClippyMe checkpointed pipeline orchestrator")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("-i", "--input", type=str)
    inputs.add_argument("-u", "--url", type=str)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("-c", "--cookies", type=str)
    parser.add_argument("--instructions", type=str)
    parser.add_argument("--no-zoom", action="store_true")
    parser.add_argument(
        "--reframe-mode",
        choices=["auto", "disabled", "subject", "object"],
        default="auto",
    )
    parser.add_argument("--letterbox-zoom", type=float, default=0.0)
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--aspect", choices=["9:16", "1:1", "16:9"], default="9:16")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument("--min-clips", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--clip-type", type=str, default=None)
    parser.add_argument("--duration-mode", type=str, default=None)
    parser.add_argument("--highlights", action="store_true", help="Generate full-video multi-tier highlight reels")
    parser.add_argument("--caption-style-default", type=str, default=None,
                        help="Caption preset applied at compose time for every clip")
    parser.add_argument("--source-timeframe", type=str, default=None,
                        help="Process only this source window, as 'start,end' seconds")
    return parser.parse_args(argv)


def _configure_overrides(args: argparse.Namespace) -> None:
    if args.model:
        if not re.match(r"^gemini-[A-Za-z0-9.\-]{1,64}$", args.model):
            raise ValueError(f"invalid --model: {args.model!r}")
        os.environ["GEMINI_MODEL"] = args.model
        print(f"🤖 Gemini model override: {args.model}", flush=True)
    if args.language:
        if not re.match(r"^[A-Za-z]{2,8}(-[A-Za-z0-9]{2,8})?$", args.language):
            raise ValueError(f"invalid --language: {args.language!r}")
        os.environ["DEEPGRAM_LANGUAGE"] = args.language
        os.environ["ELEVENLABS_LANGUAGE"] = args.language
        os.environ["CLIPPYME_LANGUAGE"] = args.language
        print(f"🌐 Language override: {args.language}", flush=True)


def _expected_aspect(label: str) -> float:
    return {"9:16": 9 / 16, "1:1": 1.0, "16:9": 16 / 9}.get(label, 9 / 16)


def _max_clips_from_env() -> int | None:
    try:
        value = int(os.getenv("CLIPPYME_MAX_CLIPS", "0") or 0)
        return value if value > 0 else None
    except ValueError:
        return None


def _transcript_cache_key(args: argparse.Namespace) -> str | None:
    """Cache slot for a URL job: the URL, plus the head trim when there is one."""
    if not args.url:
        return None
    try:
        offset = float(getattr(args, "start_offset", 0.0) or 0.0)
    except (TypeError, ValueError):
        offset = 0.0
    window = str(getattr(args, "source_timeframe", "") or "").strip()
    key = args.url
    if offset > 0:
        key += f"#start={int(offset)}"
    if window:
        key += f"#window={window}"
    return key


def _trim_head(input_video: str, output_dir: str, offset: float) -> str:
    """Drop the first ``offset`` seconds of the source, stream-copied.

    This is the VOD counterpart of the live monitor's prelive skip: the
    recording of a stream starts with the waiting screen, which is worth
    nothing and costs transcription. Every downstream timestamp is relative to
    the trimmed file, so nothing else needs to know about the offset.

    Falls back to the untouched source on any failure (a truncated container, a
    missing ffmpeg, an offset past the end) — a bad skip must not fail a job.
    """
    try:
        seconds = float(offset or 0.0)
    except (TypeError, ValueError):
        return input_video
    if seconds <= 0:
        return input_video

    trimmed = os.path.join(output_dir, f"trimmed_{os.path.basename(input_video)}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{seconds:.3f}", "-i", input_video,
             "-c", "copy", "-avoid_negative_ts", "make_zero", trimmed],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"⚠️ start-offset trim failed ({exc}); using the untrimmed source", flush=True)
        return input_video
    if not _valid_file(trimmed, 10_000):
        print("⚠️ start-offset trim produced nothing; using the untrimmed source", flush=True)
        return input_video
    print(f"✂️ Skipped the first {seconds:.0f}s of the source", flush=True)
    return os.path.abspath(trimmed)


def _trim_window(input_video: str, output_dir: str, timeframe: str | None) -> str:
    """Limit processing to the [start, end) source window, stream-copied.

    Mechanism: a fast ffmpeg ``-ss start -i <src> -t (end-start) -c copy``
    pre-trim of the downloaded source into ``windowed_<basename>``.
    Stream copy cuts at keyframes only, so window edges are approximate
    to +/- a GOP -- fine for "process this section" semantics, and it
    avoids a full re-encode of long sources. ``end`` is clamped to the
    probed source duration (never fails, never extends). Any failure
    falls back to the untrimmed source so a bad window never fails a job.
    Every downstream timestamp (transcript words, clip bounds, caption
    timings) is relative to the trimmed file, so nothing else needs to
    know about the window.
    """
    raw = (timeframe or "").strip()
    if not raw:
        return input_video
    try:
        start_s, end_s = raw.split(",", 1)
        start, end = float(start_s), float(end_s)
    except (TypeError, ValueError):
        print(f"⚠️ Ignoring malformed --source-timeframe {timeframe!r}", flush=True)
        return input_video
    if not (end > start >= 0):
        print(f"⚠️ Ignoring invalid --source-timeframe {timeframe!r}", flush=True)
        return input_video
    try:
        probe = probe_media(input_video) or {}
        duration = float(probe.get("duration") or 0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration > 0:
        end = min(end, duration)
        if not (end > start):
            print(
                f"⚠️ --source-timeframe [{start:.1f},{end:.1f}] is empty after "
                "clamping to the source duration; using the full source",
                flush=True,
            )
            return input_video
    windowed = os.path.join(output_dir, f"windowed_{os.path.basename(input_video)}")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", input_video,
             "-t", f"{end - start:.3f}", "-c", "copy",
             "-avoid_negative_ts", "make_zero", windowed],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"⚠️ source-timeframe trim failed ({exc}); using the full source", flush=True)
        return input_video
    if not _valid_file(windowed, 10_000):
        print("⚠️ source-timeframe trim produced nothing; using the full source", flush=True)
        return input_video
    print(f"✂️ Limited source to [{start:.1f}s, {end:.1f}s]", flush=True)
    return os.path.abspath(windowed)


def _prepare_input(args: argparse.Namespace, output_dir: str, state: RuntimeState, legacy):
    state.start("acquiring", "acquiring source media")
    if args.input:
        input_video = os.path.abspath(args.input)
        if not _valid_file(input_video):
            raise FileNotFoundError(f"Input file not found or empty: {input_video}")
        video_title = os.path.splitext(os.path.basename(input_video))[0]
    else:
        prior = _safe_output_artifact(state.artifact("input_video"), output_dir)
        same_url = is_same_video_source(state.artifact("source_url"), args.url)
        if not _valid_file(prior, 10_000):
            prior = find_source_video_candidate(output_dir, min_size=10_000)
        if state.completed("acquiring") and same_url and _valid_file(prior, 10_000):
            input_video = prior
            video_title = str(state.artifact("video_title") or Path(prior).stem)
            print(f"♻️ Resume: reusing downloaded source {os.path.basename(input_video)}", flush=True)
        else:
            input_video, video_title = legacy.download_youtube_video(
                args.url,
                output_dir,
                args.cookies,
            )
            input_video = os.path.abspath(input_video)
    if not _valid_file(input_video):
        raise FileNotFoundError(f"Input file not found or empty: {input_video}")
    input_video = _trim_head(input_video, output_dir, getattr(args, "start_offset", 0.0))
    input_video = _trim_window(input_video, output_dir, getattr(args, "source_timeframe", None))
    state.complete_stage(
        "acquiring",
        artifacts={
            "input_video": input_video,
            "video_title": video_title,
            "source_url": args.url,
            "source_fingerprint": _source_fingerprint(input_video),
        },
        detail="source media ready",
    )
    return input_video, video_title


def _run_preflight(args, input_video: str, output_dir: str, state: RuntimeState, legacy):
    state.start("preflight", "checking duration, cost and capacity")
    probe = probe_media(input_video)
    duration = float(probe.get("duration") or 0.0)
    if duration <= 0:
        cap = legacy.cv2.VideoCapture(input_video)
        try:
            fps = float(cap.get(legacy.cv2.CAP_PROP_FPS) or 0)
            frames = int(cap.get(legacy.cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            cap.release()
        duration = frames / fps if fps > 0 and frames > 0 else 0.0
    try:
        free_disk = shutil.disk_usage(output_dir).free
    except OSError:
        free_disk = None
    model = args.model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
    report = build_preflight(
        PreflightInputs(
            duration_seconds=duration,
            input_bytes=int(probe.get("size_bytes") or os.path.getsize(input_video)),
            width=probe.get("width"),
            height=probe.get("height"),
            model=model,
            aspect=args.aspect,
            has_gpu=bool(getattr(legacy, "CUDA_AVAILABLE", False)),
            max_clips=_max_clips_from_env(),
            analysis_enabled=not args.skip_analysis,
        ),
        pricing=getattr(legacy, "MODEL_PRICING", {}),
        free_disk_bytes=free_disk,
    )
    state.set_preflight(report)
    print(format_preflight_log(report), flush=True)
    enforce_preflight(report)
    state.complete_stage("preflight", detail="preflight passed")
    return report, duration


def _load_or_transcribe(args, input_video: str, state: RuntimeState, legacy):
    transcript_path = os.path.join(state.checkpoint_dir, "transcript.json")
    if args.skip_analysis:
        transcript = {"segments": [], "skipped": True}
        state.start("transcribing", "transcription skipped by request")
        _atomic_json(transcript_path, transcript)
        state.complete_stage(
            "transcribing",
            artifacts={"transcript": transcript_path},
            detail="transcription skipped",
        )
        return transcript
    if state.completed("transcribing"):
        transcript = _load_json(transcript_path)
        if transcript and isinstance(transcript.get("segments"), list):
            print("♻️ Resume: reusing transcript checkpoint", flush=True)
            return transcript

    state.start("transcribing", "transcribing speech")
    # A trimmed source has its own timeline, so it needs its own cache slot —
    # otherwise a later run with a different skip reuses shifted timestamps.
    cache_key = _transcript_cache_key(args)
    transcript = legacy._load_cached_transcript(cache_key) if cache_key else None
    if transcript:
        print("♻️ Reusing shared URL transcript cache", flush=True)
    else:
        transcript = legacy.transcribe_video(input_video)
        if cache_key:
            legacy._save_transcript_cache(cache_key, transcript)
    if not isinstance(transcript, dict):
        raise RuntimeError("transcription returned no structured result")
    _atomic_json(transcript_path, transcript)
    state.complete_stage(
        "transcribing",
        artifacts={"transcript": transcript_path},
        detail="transcript ready",
    )
    return transcript


def _whole_video_fallback(video_title: str, duration: float) -> dict[str, Any]:
    return {
        "shorts": [{
            "start": 0.0,
            "end": max(0.1, duration),
            "video_title_for_youtube_short": video_title,
            "video_description_for_tiktok": "",
            "video_description_for_instagram": "",
            "viral_hook_text": "",
            "viral_score": 0,
            "viral_reason": "Whole-video fallback because no valid candidate clips were available.",
        }]
    }


def _analysis_checkpoint_reusable(saved, input_video, duration,
                                   transcript, run_analysis=True):
    """Task 11: is a resumed analysis checkpoint reusable?

    A resumed record is reusable only when the audiovisual identity
    still matches (source fingerprint, AV config/version, window plan,
    evidence manifest, and the provider contract: discovery model name,
    fallback chain, AV prompt/schema version); otherwise the caller
    re-runs and marks the stale record superseded -- never silently
    reused. The CURRENT discovery config is rebuilt from the environment
    exactly like the fresh run (main._get_viral_clips_windowed), so a
    model/config change between the checkpoint and the resume is
    detected instead of being masked by the saved config.
    """
    if not run_analysis:
        return True, "analysis_skipped"
    try:
        from clippyme.pipeline.candidate_discovery import (
            av_resume_reusable, DiscoveryConfig)
        _model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        _fb_env = os.getenv("GEMINI_FALLBACK_MODELS")
        _fallbacks = (
            [m.strip() for m in _fb_env.split(",") if m.strip()]
            if _fb_env else None
        )
        current_dcfg = DiscoveryConfig.from_env(
            model_name=_model, fallback_models=_fallbacks)
        return av_resume_reusable(
            (saved or {}).get("candidate_discovery"),
            source_path=input_video, source_duration=duration,
            transcript_result=transcript,
            discovery_config=current_dcfg)
    except Exception as exc:  # noqa: BLE001 - fail closed, never reuse
        return False, f"resume_check_failed:{type(exc).__name__}"


def _supersede_analysis_checkpoint(metadata_file, reason):
    """Rename a stale analysis checkpoint so it is never silently reused.

    Returns the new path (or None). The fresh analysis run records this
    under clips_data["analysis_checkpoint_superseded"].
    """
    try:
        if metadata_file and os.path.isfile(metadata_file):
            stamped = (f"{metadata_file}.superseded-"
                       f"{int(time.time())}.json")
            os.replace(metadata_file, stamped)
            print(f"   🗄️ stale analysis checkpoint superseded "
                  f"→ {os.path.basename(stamped)} ({reason})",
                  flush=True)
            return stamped
    except OSError as exc:
        print(f"⚠️ could not supersede stale analysis checkpoint: "
              f"{exc}", flush=True)
    return None


def _load_or_analyze(
    args,
    input_video: str,
    video_title: str,
    output_dir: str,
    duration: float,
    transcript: dict,
    state: RuntimeState,
    legacy,
):
    _superseded_checkpoint = None
    prior_metadata = _safe_output_artifact(state.artifact("metadata"), output_dir)
    safe_title = sanitize_windows_basename(video_title, max_len=100) or "video"
    default_metadata = os.path.join(output_dir, f"{safe_title}_metadata.json")
    metadata_file = prior_metadata or default_metadata
    if state.completed("analyzing"):
        # Phase B: analysis is checkpointed -- any steering that arrived
        # after the checkpoint (e.g. queued before a backend restart)
        # can never apply. Record it loudly instead of dropping it.
        _missed = state.consume_steering()
        if _missed:
            state.note_missed_steering(
                _missed, "job resumed from the analysis checkpoint; "
                "steering arrived too late to affect moment selection")
        saved = _load_json(metadata_file)
        if saved and isinstance(saved.get("shorts"), list):
            _ok, _why = _analysis_checkpoint_reusable(
                saved, input_video, duration, transcript,
                run_analysis=not args.skip_analysis)
            if _ok:
                print("♻️ Resume: reusing analysis metadata checkpoint", flush=True)
                return saved, metadata_file
            print(f"♻️ Resume: analysis checkpoint not reusable "
                  f"({_why}); re-running analysis", flush=True)
            _superseded_path = _supersede_analysis_checkpoint(
                metadata_file, _why)
            _superseded_checkpoint = (
                {"previous": _superseded_path, "reason": _why}
                if _superseded_path else {"previous": None,
                                          "reason": _why})

    state.start("analyzing", "selecting and validating clip candidates")
    # Phase B: mid-job steering. Consume pending user prompts here --
    # this marks the steering checkpoint as passed, so the API rejects
    # later prompts (409) instead of silently ignoring them.
    _steer_prompts = state.consume_steering()
    if args.skip_analysis:
        if _steer_prompts:
            state.note_missed_steering(
                _steer_prompts,
                "analysis skipped for this job (skip_analysis); "
                "steering could not be applied")
        clips_data = _whole_video_fallback(video_title, duration)
    else:
        from clippyme.pipeline.audience_intel import load_audience_intel
        audience_intel = load_audience_intel(output_dir)
        _instructions = args.instructions
        if _steer_prompts:
            _extra = "\n\n".join(f"[User steering] {p}" for p in _steer_prompts)
            _instructions = f"{_instructions}\n\n{_extra}" if _instructions else _extra
            print(f"\U0001F9ED steering: applying {len(_steer_prompts)} user "
                  f"prompt(s) to moment selection", flush=True)
        clips_data = legacy.get_viral_clips(
            transcript,
            duration,
            instructions=_instructions,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_clips=args.min_clips,
            max_clips=args.max_clips,
            clip_type=args.clip_type,
            duration_mode=getattr(args, "duration_mode", None),
            audience_intel=audience_intel,
            # Task 11: the queued path discovers from the real source
            # video (optional; keeps Task 10 behavior when omitted).
            source_path=input_video,
        )
        # Preserve discovery funnel through fallbacks (Task 08 Fix 6).
        _discovery_funnel = (clips_data.get("candidate_discovery")
                             if isinstance(clips_data, dict) else None)
        _candidate_store = (clips_data.get("candidate_store")
                            if isinstance(clips_data, dict) else None)
        # Task 09: preserve the editorial funnel + plans through fallbacks.
        _editorial_funnel = (clips_data.get("editorial_refinement")
                             if isinstance(clips_data, dict) else None)
        _editorial_plans = (clips_data.get("editorial_plans")
                            if isinstance(clips_data, dict) else None)
        # Task 10: preserve the application record through fallbacks.
        _application_record = (clips_data.get("editorial_application")
                               if isinstance(clips_data, dict) else None)
        def _preserve_funnel(data, strategy=None):
            # strategy names the fallback ("texttiling"/"whole_video");
            # None preserves the application record verbatim (no
            # fallback shortlist was produced).
            if isinstance(data, dict):
                if _discovery_funnel is not None:
                    data["candidate_discovery"] = _discovery_funnel
                    data["candidate_discovery"]["fallback_used"] = True
                if _candidate_store is not None:
                    data["candidate_store"] = _candidate_store
                if _editorial_funnel is not None:
                    data["editorial_refinement"] = _editorial_funnel
                if _editorial_plans is not None:
                    data["editorial_plans"] = _editorial_plans
                if _application_record is not None:
                    if strategy is None:
                        data["editorial_application"] = _application_record
                    else:
                        from clippyme.pipeline import (
                            editorial_application as _ea)
                        _ea.preserve_application_through_fallback(
                            data, _application_record, strategy)
            return data
        if not clips_data or "shorts" not in clips_data:
            if should_use_fallback(args.monitor):
                clips_data = _preserve_funnel(
                    legacy.build_texttiling_fallback(transcript, video_title),
                    "texttiling")
                if _discovery_funnel is not None and isinstance(clips_data, dict) and "candidate_discovery" in clips_data:
                    clips_data["candidate_discovery"]["fallback_strategy"] = "texttiling"
        if not clips_data or not clips_data.get("shorts"):
            from clippyme.pipeline.editorial_application import (
                is_intentional_editorial_empty)
            if is_intentional_editorial_empty(clips_data):
                # Enabled editorial application honestly reduced a
                # non-empty selection to zero: persist zero shorts and
                # render nothing. Not a failure -- no fallback.
                clips_data = _preserve_funnel({"shorts": []})
            elif not should_use_fallback(args.monitor):
                clips_data = _preserve_funnel({
                    "shorts": [],
                    "gemini_exhausted": bool(
                        getattr(legacy.get_viral_clips, "_last_gemini_exhausted", False)
                    ),
                })
            else:
                print("⚠️ No valid AI/topic clips; using whole-video fallback", flush=True)
                clips_data = _preserve_funnel(
                    _whole_video_fallback(video_title, duration),
                    "whole_video")
                if _discovery_funnel is not None and isinstance(clips_data, dict) and "candidate_discovery" in clips_data:
                    clips_data["candidate_discovery"]["fallback_strategy"] = "whole_video"
        # Phase B: steering that arrived while moment selection was
        # running missed the checkpoint -- record it, never drop it.
        _late = state.consume_steering()
        if _late:
            state.note_missed_steering(
                _late, "arrived after moment selection ran; "
                "steering applies to the next job")

    clips_data["transcript"] = transcript
    clips_data["aspect"] = args.aspect
    # Task 08: fail-soft validation of windowed discovery metadata.
    # The candidate_discovery dict (if present) flows through the
    # existing atomic checkpoint unchanged; we only verify its version.
    _cd = clips_data.get("candidate_discovery")
    if isinstance(_cd, dict):
        try:
            from clippyme.pipeline.candidate_discovery import (
                DISCOVERY_VERSION)
            if _cd.get("version") != DISCOVERY_VERSION:
                print(
                    f"?? candidate_discovery version mismatch "
                    f"(got {_cd.get('version')}, want {DISCOVERY_VERSION}); "
                    f"keeping metadata as-is (fail-soft).",
                    flush=True)
        except Exception:
            pass
    # Task 09: fail-soft validation of editorial refinement metadata.
    # The editorial_refinement dict (if present) flows through the
    # existing atomic checkpoint unchanged; we only verify its version.
    _er = clips_data.get("editorial_refinement")
    if isinstance(_er, dict):
        try:
            from clippyme.pipeline.editorial_refinement import (
                EDITORIAL_VERSION)
            if _er.get("version") != EDITORIAL_VERSION:
                print(
                    f"?? editorial_refinement version mismatch "
                    f"(got {_er.get('version')}, want {EDITORIAL_VERSION}); "
                    f"keeping metadata as-is (fail-soft).",
                    flush=True)
        except Exception:
            pass
    # Task 10: fail-soft validation of the editorial application record.
    # It flows through the existing atomic checkpoint unchanged; we only
    # verify its version.
    _ea_rec = clips_data.get("editorial_application")
    if isinstance(_ea_rec, dict):
        try:
            from clippyme.pipeline.editorial_application import (
                APPLICATION_VERSION)
            if _ea_rec.get("version") != APPLICATION_VERSION:
                print(
                    f"?? editorial_application version mismatch "
                    f"(got {_ea_rec.get('version')}, want "
                    f"{APPLICATION_VERSION}); keeping metadata as-is "
                    f"(fail-soft).",
                    flush=True)
        except Exception:
            pass
    if getattr(args, "caption_style_default", None):
        clips_data["caption_style_default"] = args.caption_style_default
    shorts = clips_data.get("shorts") or []
    max_clips = _max_clips_from_env()
    if max_clips and len(shorts) > max_clips:
        print(
            f"✂️ CLIPPYME_MAX_CLIPS={max_clips}: keeping the highest-ranked "
            f"{max_clips} of {len(shorts)} candidates",
            flush=True,
        )
        shorts = shorts[:max_clips]
        clips_data["shorts"] = shorts

    if shorts and not args.skip_analysis:
        from clippyme.pipeline.cut_ops import flatten_words, snap_clips_to_transcript
        from clippyme.pipeline.duration_policy import resolve_duration_policy

        # Resolve once via shared policy; same bounds as prompt generation.
        _policy = resolve_duration_policy(
            min_duration=getattr(args, "min_duration", None),
            max_duration=getattr(args, "max_duration", None),
            duration_mode=getattr(args, "duration_mode", None),
            clip_type=getattr(args, "clip_type", None),
            video_duration=duration,
        )
        words = flatten_words(transcript)
        silences: list = []
        if _enabled("CLIPPYME_SILENCE_SNAP"):
            try:
                from clippyme.pipeline.media_probe import detect_silences

                silences = detect_silences(input_video)
            except Exception as exc:  # noqa: BLE001
                print(f"⚠️ Silence snap unavailable: {exc}", flush=True)
        for event in snap_clips_to_transcript(
            shorts,
            words,
            source_duration=duration or None,
            silences=silences,
            default_reframe_mode=args.reframe_mode,
            max_duration=_policy.max_duration,
            min_duration=_policy.min_duration,
        ):
            print(
                f"🎯 snap[{event.path}]: [{event.old_start:.2f},{event.old_end:.2f}] "
                f"→ [{event.new_start:.2f},{event.new_end:.2f}]",
                flush=True,
            )

    source_info = _load_json(os.path.join(output_dir, "source_info.json"))
    if source_info:
        clips_data["source_info"] = source_info
    if _superseded_checkpoint is not None:
        clips_data["analysis_checkpoint_superseded"] = (
            _superseded_checkpoint)
    _save_metadata(metadata_file, clips_data)
    state.set_clip_total(len(shorts))
    state.complete_stage(
        "analyzing",
        artifacts={"metadata": metadata_file},
        detail="clip plan ready",
    )
    return clips_data, metadata_file


def _qa_status(report: dict[str, Any]) -> str:
    if report.get("critical"):
        return "failed"
    if report.get("warnings"):
        return "warning"
    return "ready"


def _clip_bounds(clip: dict[str, Any]) -> tuple[float, float]:
    start = float(clip["start"])
    end = float(clip["end"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
        raise ValueError(f"invalid clip interval: start={start!r}, end={end!r}")
    return start, end



def _sidecar_for(legacy, video_path: str) -> str:
    """Task 13: sidecar path for a video, tolerant of faked ``legacy`` modules.

    Production ``legacy`` is the real ``reframe`` module (which defines the
    canonical ``turn_framing_sidecar_for``); tests inject fakes without it,
    so fall back to the identical basename rule.
    """
    fn = getattr(legacy, "turn_framing_sidecar_for", None)
    if callable(fn):
        return fn(video_path)
    return os.path.splitext(video_path)[0] + ".turn_framing_plan.json"


def _resolve_project_for_preservation(output_dir, index):
    """Task 01 project resolver shared by the pre-render rerender guard and
    the emission guard (review-03 correction 4).

    Returns the working ``clip-project-<index>.json`` when present, else
    the latest version-map-backed archived project via
    ``project_render.get_project_path`` (the Task 01 working/archived
    fallback contract), else None. Callers apply ``project_has_user_edits``
    to the resolved path -- one resolver, so the two guards can never
    drift apart (an archived-only user project pins its accepted pair
    before rendering exactly as it pins its sidecar at emission).
    """
    from clippyme.pipeline.project_emit import PROJECT_FILENAME_TEMPLATE

    working = os.path.join(
        output_dir, PROJECT_FILENAME_TEMPLATE.format(index=index)
    )
    if os.path.isfile(working):
        return working
    try:
        from clippyme.domain.project_render import get_project_path

        return get_project_path(output_dir, index)
    except Exception:
        return None


def _maybe_emit_clip_project(**kwargs):
    """Emit the default-edit project unless the user already edited it.

    Preservation guard (Task 01): a completed clip's project may carry
    user edits. Overwriting it with a fresh version-1 would silently
    discard their work or hide an archived user version.

    Resolution contract: the shared working/archived resolver -- working
    file first, else the latest version-map-backed archived project.
    Preserve any resolved project with user edits -- including
    archived-only state -- by not emitting a new default.
    """
    from clippyme.pipeline.project_emit import PROJECT_FILENAME_TEMPLATE

    output_dir = kwargs.get("output_dir")
    index = kwargs.get("index")
    if output_dir is not None and index is not None:
        resolved = _resolve_project_for_preservation(output_dir, index)
        if resolved and project_has_user_edits(resolved):
            archived = os.path.basename(resolved) != (
                PROJECT_FILENAME_TEMPLATE.format(index=index)
            )
            print(
                f"clip-project: preserving "
                f"{'archived user' if archived else 'user-edited'} "
                f"project for clip {index + 1}",
                flush=True,
            )
            return resolved
    return emit_clip_project(**kwargs)



def _render_one_clip(
    *,
    index: int,
    total: int,
    clip: dict[str, Any],
    input_video: str,
    video_title: str,
    output_dir: str,
    metadata_file: str,
    clips_data: dict[str, Any],
    args: argparse.Namespace,
    aspect_ratio: float,
    state: RuntimeState,
    legacy,
) -> bool:
    start, end = _clip_bounds(clip)
    expected_duration = end - start
    title = clip.get("video_title_for_youtube_short") or clip.get("title")
    existing_name = _safe_clip_filename(clip.get("clip_filename"))
    clip_filename = existing_name or f"{clip_output_basename(title, index, video_title)}.mp4"
    clip["clip_filename"] = clip_filename
    clip_source = os.path.join(output_dir, f"source_{clip_filename}")
    clip_final = os.path.join(output_dir, clip_filename)
    _save_metadata(metadata_file, clips_data)

    if _valid_file(clip_final, 10_000):
        report = inspect_clip(
            clip_final,
            expected_duration=expected_duration,
            expected_aspect=aspect_ratio,
            run_signal_checks=_enabled("CLIPPYME_QA_SIGNAL"),
        )
        if not report.get("critical"):
            clip["qa"] = report
            state.mark_clip(index, _qa_status(report), report)
            _save_metadata(metadata_file, clips_data)
            # Phase 1 (clip editor): emit the default-edit project -- but
            # NEVER overwrite an existing working project that carries user
            # edits (Task 01 preservation guarantee). Additive only;
            # emit_clip_project never raises.
            # Task 13 (resume): the stable sidecar should sit next to the
            # reused final clip. Old clips predate Task 13 -- informational
            # only; never rerender or fail for a missing sidecar.
            _stable_sidecar = _sidecar_for(legacy, clip_final)
            if not os.path.isfile(_stable_sidecar):
                print(
                    f"clip-project: no stable turn-framing sidecar for resumed "
                    f"clip {index + 1}/{total} (pre-Task 13)",
                    flush=True,
                )
            _maybe_emit_clip_project(
                output_dir=output_dir,
                job_id=os.getenv("CLIPPYME_JOB_ID") or os.path.basename(output_dir),
                index=index,
                start=start,
                end=end,
                clip=clip,
                clips_data=clips_data,
                clip_source=clip_source,
                clip_final=clip_final,
            )
            print(f"♻️ Resume: clip {index + 1}/{total} already valid", flush=True)
            return True
        try:
            os.remove(clip_final)
        except OSError:
            pass

    # Task 13 review-02/03: decline the rerender for a preserved project.
    # The Task 01 preservation contract (project_has_user_edits) pins
    # the accepted video+sidecar pair the preserved project references;
    # replacing either would invalidate its reframed_base lineage
    # (invalid_asset). Review-03: the guard uses the SAME working/archived
    # resolver as the emission guard, so a version-map-backed
    # archived-only user project also declines the rerender BEFORE its
    # pixel base changes (it stays lineage-eligible and renderable).
    # When an accepted pair was snapshotted, keep it -- restoring the
    # pair from the snapshot when it is gone -- and return success rather
    # than rendering a replacement. Without a snapshot there is no
    # accepted pair to keep, so the render proceeds (the project file
    # itself stays preserved).
    _preserved_project = _resolve_project_for_preservation(
        output_dir, index)
    if _preserved_project and project_has_user_edits(_preserved_project):
        from clippyme.domain import project_framing as _pf13decline
        if _pf13decline.accepted_snapshot_exists(
                job_dir=output_dir, clip_final=clip_final):
            print(
                f"clip-project: declining rerender for preserved project "
                f"(clip {index + 1}/{total}) -- keeping the accepted pair",
                flush=True,
            )
            _pf13decline.restore_accepted_snapshot(
                job_dir=output_dir,
                clip_final=clip_final,
                stable_sidecar=_sidecar_for(legacy, clip_final),
            )
            clip["qa"] = {"ok": True, "critical": False, "issues": [],
                          "warnings": []}
            state.mark_clip(index, "complete", clip["qa"])
            _save_metadata(metadata_file, clips_data)
            return True

    state.start(
        "cutting",
        f"cutting clip {index + 1}/{total}",
        progress=62 + int(index / max(1, total) * 8),
    )
    slice_reusable = _valid_media_file(clip_source, 10_000) and slice_provenance_valid(
        clip_source, start=start, end=end, source_path=input_video
    )
    if not slice_reusable:
        if os.path.exists(clip_source) and not _valid_media_file(clip_source, 10_000):
            # Corrupt/partial slice: safe to drop, it was never valid.
            try:
                os.remove(clip_source)
            except OSError:
                pass
        elif os.path.exists(clip_source):
            # Valid file but stale provenance (legacy stream-copy slice,
            # changed interval, or changed source): re-cut. The old file
            # stays until the replacement is verified (atomic os.replace).
            print(f"Resume: source slice provenance mismatch for clip {index + 1}/{total} -- re-cutting", flush=True)
        if not _valid_media_file(input_video, 10_000):
            candidate = find_source_video_candidate(output_dir, min_size=10_000)
            if candidate and _valid_media_file(candidate, 10_000):
                input_video = candidate
                print(f"♻️ Re-linked source video: {os.path.basename(input_video)}", flush=True)
        if not _valid_media_file(input_video, 10_000):
            raise FileNotFoundError(f"Input source video file not found on disk: {input_video}")
        
        # Fingerprint the source BEFORE the cut: if it changes mid-cut,
        # the provenance check will invalidate the result.
        src_fp = source_fingerprint(input_video)
        temp_dest = clip_source + ".tmp.mp4"
        command = build_cut_command(input_video, start, end, temp_dest)
        print(f"✂️ Clip {index + 1}/{total}: {start:.2f}s → {end:.2f}s", flush=True)
        
        clip_duration = float(end) - float(start)
        timeout_factor = int(os.getenv("CLIPPYME_CUT_TIMEOUT_FACTOR", "8"))
        timeout = max(600, int(clip_duration * timeout_factor))
        
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if process.returncode != 0:
            tail = (process.stderr or b"").decode("utf-8", errors="replace")[-1000:]
            try:
                os.remove(temp_dest)
            except OSError:
                pass
            raise RuntimeError(f"ffmpeg cut failed for clip {index + 1}: {tail}")
        # Validate BEFORE swapping: a successful ffmpeg exit code does not
        # guarantee usable media. Never replace a known-good slice with a
        # broken temp file.
        ok, reason = validate_cut_output(temp_dest, clip_duration)
        if not ok:
            try:
                os.remove(temp_dest)
            except OSError:
                pass
            raise RuntimeError(f"ffmpeg cut produced invalid output for clip {index + 1}: {reason}")
        # Re-compare the source fingerprint BEFORE publishing: a source
        # that changed mid-cut must not replace the old slice.
        if source_fingerprint(input_video) != src_fp:
            try:
                os.remove(temp_dest)
            except OSError:
                pass
            raise RuntimeError(
                f"source changed during cut for clip {index + 1}; "
                "discarding temp slice, keeping previous"
            )
        os.replace(temp_dest, clip_source)
        write_slice_provenance(clip_source, start=start, end=end, source_fp=src_fp)
    else:
        print(f"♻️ Resume: reusing source slice for clip {index + 1}", flush=True)

    state.start(
        "reframing",
        f"rendering clip {index + 1}/{total}",
        progress=70 + int(index / max(1, total) * 20),
    )
    retries = _nonnegative_int("CLIPPYME_RENDER_QA_RETRIES", 1)
    last_report: dict[str, Any] | None = None
    for render_attempt in range(1, retries + 2):
        temp_output = clip_final + ".render.tmp.mp4"
        try:
            os.remove(temp_output)
        except FileNotFoundError:
            pass
        # Task 13: drop a stale temp sidecar from a crashed/failed attempt so
        # this attempt cannot leak it or promote another attempt's plan.
        try:
            _attempt_sidecar = _sidecar_for(legacy, temp_output)
            if os.path.isfile(_attempt_sidecar):
                os.remove(_attempt_sidecar)
        except OSError:
            pass
        success = legacy.process_video_to_vertical(
            clip_source,
            temp_output,
            reframe_mode=args.reframe_mode,
            zoom_end=None if args.no_zoom else 1.05,
            aspect_ratio=aspect_ratio,
            letterbox_zoom=normalize_letterbox_zoom(args.letterbox_zoom),
        )
        if not success or not _valid_file(temp_output, 10_000):
            last_report = {
                "ok": False,
                "critical": True,
                "issues": ["reframe produced no valid file"],
                "warnings": [],
            }
        else:
            legacy.normalize_audio(temp_output)
            state.start(
                "quality",
                f"verifying clip {index + 1}/{total}",
                progress=90 + int((index + 1) / max(1, total) * 6),
            )
            last_report = inspect_clip(
                temp_output,
                expected_duration=expected_duration,
                expected_aspect=aspect_ratio,
                run_signal_checks=_enabled("CLIPPYME_QA_SIGNAL"),
            )
        if last_report and not last_report.get("critical"):
            # Task 13 review-01: coordinated video + sidecar publication.
            # The video is published first; the sidecar half is coordinated
            # by publish_attempt_sidecar (promotion / retire /
            # preserved-project skip).
            # Task 13 review-02/03: stage the previous accepted pair BEFORE
            # publication. Review-03: staging is FAIL-CLOSED -- if the
            # previous pair cannot be backed up AND verified, publication
            # is aborted here and the accepted pair stays untouched (no
            # os.replace, no half-published state). A promotion/retirement
            # OSError is NEVER swallowed: BOTH the video and the sidecar
            # are restored from the staged pair and the attempt fails, so
            # an accepted clip can never pair a new video with a stale
            # plan and a failed publish can never destroy the previous
            # accepted pair.
            from clippyme.domain import project_framing as _pf13
            _staged_pair = None
            try:
                _staged_pair = _pf13.stage_previous_pair(
                    clip_final=clip_final,
                    stable_sidecar=_sidecar_for(legacy, clip_final),
                )
            except OSError as _stage_exc:
                print(
                    f"   Task 13: pair staging failed ({_stage_exc}); "
                    f"keeping previous pair for clip {index + 1}/{total}",
                    flush=True,
                )
                last_report = {
                    "ok": False,
                    "critical": True,
                    "issues": ["pair staging failed; previous pair kept"],
                    "warnings": [],
                }
            else:
                os.replace(temp_output, clip_final)
                _promotion_ok = True
                _pub_outcome = _pf13.PUBLISH_NO_SIDECAR
                try:
                    _pub_outcome = _pf13.publish_attempt_sidecar(
                        attempt_sidecar=_sidecar_for(legacy, temp_output),
                        stable_sidecar=_sidecar_for(legacy, clip_final),
                        source_path=clip_source,
                        job_dir=output_dir,
                    )
                except OSError as _pub_exc:
                    _promotion_ok = False
                    print(
                        f"   Task 13: sidecar publish failed ({_pub_exc}); "
                        f"restoring staged pair for clip {index + 1}/{total}",
                        flush=True,
                    )
                    try:
                        _pf13.restore_staged_pair(
                            _staged_pair,
                            clip_final=clip_final,
                            stable_sidecar=_sidecar_for(legacy, clip_final),
                        )
                    except OSError as _restore_exc:
                        # Fail loudly: the pair state is unverifiable.
                        # The attempt still fails; nothing is reported
                        # as successfully published.
                        print(
                            f"   Task 13: CRITICAL staged-pair restore "
                            f"failed ({_restore_exc}); pair state uncertain "
                            f"for clip {index + 1}/{total}",
                            flush=True,
                        )
                    try:
                        _abandoned = _sidecar_for(legacy, temp_output)
                        if os.path.isfile(_abandoned):
                            os.remove(_abandoned)
                    except OSError:
                        pass
                if not _promotion_ok:
                    # Fall through to the attempt-failure handling below: the
                    # retry loop may try again; exhaustion returns False.
                    last_report = {
                        "ok": False,
                        "critical": True,
                        "issues": ["sidecar promotion failed; video rolled back"],
                        "warnings": [],
                    }
                else:
                    _pf13.discard_staged_pair(_staged_pair)
                    # Task 13 review-02: snapshot the accepted pair so a
                    # later declined rerender (preserved project) can restore
                    # the exact video+sidecar the project references.
                    # Review-03: atomic per-clip generations. A False return
                    # means durable recovery is unavailable -- warn, but do
                    # not fail the accepted clip over it.
                    if not _pf13.snapshot_accepted_pair(
                        job_dir=output_dir,
                        clip_final=clip_final,
                        stable_sidecar=_sidecar_for(legacy, clip_final),
                    ):
                        print(
                            f"   Task 13: WARNING accepted-pair snapshot "
                            f"not durable for clip {index + 1}/{total}; "
                            f"declined-rerender recovery unavailable",
                            flush=True,
                        )
                    if _pub_outcome != _pf13.PUBLISH_PROMOTED:
                        print(
                            f"   Task 13: sidecar publish outcome "
                            f"'{_pub_outcome}' for clip {index + 1}/{total}",
                            flush=True,
                        )
                    # The cover must be based on the stable final basename; generating it
                    # from the temporary render would orphan `*.render.tmp_cover.jpg`.
                    legacy.select_cover_frame(clip_final)
                    clip["qa"] = last_report
                    status = _qa_status(last_report)
                    state.mark_clip(index, status, last_report)
                    _save_metadata(metadata_file, clips_data)
                    # Phase 1 (clip editor): emit the default-edit project. Additive
                    # only -- emit_clip_project never raises.
                    _maybe_emit_clip_project(
                        output_dir=output_dir,
                        job_id=os.getenv("CLIPPYME_JOB_ID") or os.path.basename(output_dir),
                        index=index,
                        start=start,
                        end=end,
                        clip=clip,
                        clips_data=clips_data,
                        clip_source=clip_source,
                        clip_final=clip_final,
                    )
                    warning_text = (
                        f" with {len(last_report.get('warnings') or [])} warning(s)"
                        if status == "warning"
                        else ""
                    )
                    print(
                        f"✅ Clip {index + 1}/{total} ready{warning_text}: {clip_filename}",
                        flush=True,
                    )
                    return True
        try:
            os.remove(temp_output)
        except OSError:
            pass
        # Task 13: a failed render/QA/retry must not leak its temp sidecar;
        # a retry must not promote another attempt's plan.
        try:
            _failed_sidecar = _sidecar_for(legacy, temp_output)
            if os.path.isfile(_failed_sidecar):
                os.remove(_failed_sidecar)
        except OSError:
            pass
        if render_attempt <= retries:
            print(
                f"🔁 QA rejected clip {index + 1}; render retry {render_attempt}/{retries}",
                flush=True,
            )

    clip["qa"] = last_report or {
        "ok": False,
        "critical": True,
        "issues": ["unknown render failure"],
        "warnings": [],
    }
    state.mark_clip(index, "failed", clip["qa"])
    _save_metadata(metadata_file, clips_data)
    print(f"❌ Clip {index + 1}/{total} failed QA; source slice preserved", flush=True)
    return False


def _cleanup_completed(
    *,
    output_dir: str,
    state: RuntimeState,
    input_video: str,
    is_url: bool,
    keep_original: bool,
    all_clips_ready: bool,
) -> None:
    if is_url and not keep_original and _valid_file(input_video):
        try:
            os.remove(input_video)
            print("🗑️ Cleaned downloaded source after successful completion", flush=True)
        except OSError as exc:
            print(f"⚠️ Could not clean downloaded source: {exc}", flush=True)
    if not all_clips_ready or _enabled("CLIPPYME_KEEP_CHECKPOINTS", "0"):
        return
    # Preserve source_<clip>.mp4: POST /api/reframe needs these slices
    # to change framing without downloading/transcribing again.
    try:
        shutil.rmtree(state.checkpoint_dir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"⚠️ Could not clean checkpoint cache: {exc}", flush=True)


def run(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = os.path.abspath(resolve_output_dir(args.output, default="."))
    os.makedirs(output_dir, exist_ok=True)
    state = RuntimeState(output_dir, job_id=os.getenv("CLIPPYME_JOB_ID"))
    state.begin_attempt(
        int(os.getenv("CLIPPYME_ATTEMPT", "1") or 1),
        int(os.getenv("CLIPPYME_JOB_MAX_ATTEMPTS", "3") or 3),
    )
    started = time.time()

    try:
        _configure_overrides(args)
        from clippyme.pipeline import main as legacy

        aspect_ratio = _expected_aspect(args.aspect)
        input_video, video_title = _prepare_input(args, output_dir, state, legacy)
        _preflight, duration = _run_preflight(args, input_video, output_dir, state, legacy)
        transcript = _load_or_transcribe(args, input_video, state, legacy)
        if getattr(args, "highlights", False):
            state.start("analyzing", "AI scoring viral moments across full video", progress=55)
            from clippyme.domain.highlight_service import generate_multi_tier_highlights_sync
            output_root = os.path.dirname(output_dir) or "output"
            job_id = os.getenv("CLIPPYME_JOB_ID") or os.path.basename(output_dir)
            print("✨ Synthesizing full-video multi-tier highlight reels...", flush=True)
            res = generate_multi_tier_highlights_sync(
                job_id=job_id,
                aspect=args.aspect,
                reframe_mode=args.reframe_mode,
                output_root=output_root,
                model_name=args.model,
            )
            hl_list = res.get("highlights", [])
            state.start("finalizing", f"completed {len(hl_list)} highlight reels", progress=98)
            state.finish(f"completed: {len(hl_list)} highlight reels ready")
            print(f"🎉 Generated {len(hl_list)} multi-tier highlight reels for full video in {time.time() - started:.2f}s", flush=True)
            _cleanup_completed(
                output_dir=output_dir,
                state=state,
                input_video=input_video,
                is_url=args.url is not None,
                keep_original=bool(args.keep_original),
                all_clips_ready=bool(hl_list),
            )
            return 0

        clips_data, metadata_file = _load_or_analyze(
            args,
            input_video,
            video_title,
            output_dir,
            duration,
            transcript,
            state,
            legacy,
        )
        clips = clips_data.get("shorts") or []
        if not clips:
            state.start("finalizing", "no valid clips for this source", progress=99)
            state.finish("completed with zero clips")
            _cleanup_completed(
                output_dir=output_dir,
                state=state,
                input_video=input_video,
                is_url=bool(args.url),
                keep_original=args.keep_original,
                all_clips_ready=True,
            )
            print("🚫 No valid clips generated for this job", flush=True)
            return 0

        ready = 0
        for index, clip in enumerate(clips):
            try:
                if _render_one_clip(
                    index=index,
                    total=len(clips),
                    clip=clip,
                    input_video=input_video,
                    video_title=video_title,
                    output_dir=output_dir,
                    metadata_file=metadata_file,
                    clips_data=clips_data,
                    args=args,
                    aspect_ratio=aspect_ratio,
                    state=state,
                    legacy=legacy,
                ):
                    ready += 1
            except Exception as exc:  # one bad candidate must not lose siblings
                clip["qa"] = {
                    "ok": False,
                    "critical": True,
                    "issues": [str(exc)],
                    "warnings": [],
                }
                state.mark_clip(index, "failed", clip["qa"])
                _save_metadata(metadata_file, clips_data)
                print(f"❌ Clip {index + 1}/{len(clips)} failed: {exc}", flush=True)
                logger.exception("clip %d failed", index + 1)

        state.start("finalizing", "generating situational platform captions & metadata", progress=98)
        try:
            from clippyme.domain.metadata_generator import generate_all_clips_metadata
            jid = os.getenv("CLIPPYME_JOB_ID") or os.path.basename(output_dir)
            generate_all_clips_metadata(jid, output_dir=output_dir, force=False)
            print("✨ Generated situational platform captions & descriptions for all clips", flush=True)
            refreshed = _load_json(metadata_file)
            if refreshed and isinstance(refreshed.get("shorts"), list):
                clips_data["shorts"] = refreshed["shorts"]
        except Exception as exc:
            logger.warning("Auto metadata generation failed: %s", exc)
            print(f"⚠️ Platform metadata generation notice: {exc}", flush=True)

        _save_metadata(metadata_file, clips_data)
        if ready == 0:
            raise RuntimeError("all candidate clips failed rendering or QA")

        state.finish(f"completed: {ready}/{len(clips)} clips ready")
        _cleanup_completed(
            output_dir=output_dir,
            state=state,
            input_video=input_video,
            is_url=bool(args.url),
            keep_original=args.keep_original,
            all_clips_ready=ready == len(clips),
        )
        print(f"⏱️ Total execution time: {time.time() - started:.2f}s", flush=True)
        return 0
    except PreflightRejected as exc:
        state.fail(str(exc), resumable=False)
        print(f"❌ Preflight rejected job: {exc}", flush=True)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        state.fail(str(exc), resumable=False)
        print(f"❌ {exc}", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001
        state.fail(str(exc), resumable=True)
        logger.exception("checkpointed pipeline failed")
        print(f"❌ Pipeline failed: {exc}", flush=True)
        try:
            from clippyme.domain.job_artifacts import cleanup_failed_job_artifacts
            j_id = state.data.get("job_id") or os.path.basename(output_dir)
            cleanup_failed_job_artifacts(j_id, os.path.dirname(output_dir))
        except Exception:
            pass
        return 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
