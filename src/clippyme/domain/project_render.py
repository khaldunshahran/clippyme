"""Phase 2 (clip editor): render a ClipProject to a versioned mp4.

``render_project()`` is the single orchestrator for project-based renders:

1. Resolve the base timeline:
   - ``project.base_clip`` set (legacy-mapped projects): use that 9:16 master
     directly -- no re-encode, so legacy renders stay byte-identical.
   - Otherwise: cut/crop/concat ``project.segments`` from ``source.file``
     into a 9:16 timeline (one ffmpeg filtergraph, frame-accurate).
2. Map caption words (with ``edits`` applied) from source-file coordinates
   to timeline coordinates; synthesize the transcript/clip_info the layer
   pipeline expects (clip-relative, start=0).
3. Derive compose toggles+params from project content -- or use the stashed
   ``legacy_params`` verbatim for byte-identical legacy renders.
4. Run ``compose_layers`` (the ONE layer implementation -- grade, subtitles,
   smartcut, hook, logo). Layer order is machine-checked (canonical).
5. Mix music (v1: null file -> skip; non-null -> amix at volume/offset).
6. QA via ``inspect_clip``; write a NEW versioned file per render
   (``<stem>.edited_v<version>.mp4``); record version->file mapping.

Never overwrites: masters, previous composes, or previous versions.
All times in a ClipProject are source-file coordinates (0 = first frame of
``source.file``); see ``clip_project.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid

from clippyme.domain.clip_project import ClipProject, project_to_json
from clippyme.domain.compose import compose_layers, validate_layer_order
from clippyme.domain.encode import x264_video_args
from clippyme.domain.errors import ConflictError, ValidationError
from clippyme.pipeline.media_qa import inspect_clip, probe_media

logger = logging.getLogger(__name__)


class ProjectRenderError(ValidationError):
    """A project render request is invalid or cannot be fulfilled.

    Maps to HTTP 400 via the ClippyMeError handler -- never silently corrupt.
    """

    status_code = 400


class ProjectRenderResult:
    """Outcome of render_project()."""

    def __init__(self, output_basename: str, output_path: str,
                 version: int | None, deduped: bool = False):
        self.output_basename = output_basename
        self.output_path = output_path
        self.version = version
        self.deduped = deduped


VERSION_MAP_FILENAME = "project-versions.json"
"""Per-job version map: {str(clip_index): {"versions": {str(v): {...}}, "latest": v}}."""


# ---------------------------------------------------------------------------
# Version map + idempotency
# ---------------------------------------------------------------------------

def _version_map_path(job_dir: str) -> str:
    return os.path.join(job_dir, VERSION_MAP_FILENAME)


def _load_version_map(job_dir: str) -> dict:
    path = _version_map_path(job_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_version_map(job_dir: str, data: dict) -> None:
    path = _version_map_path(job_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _find_by_idempotency_key(version_map: dict, clip_index: int,
                             idempotency_key: str) -> dict | None:
    """Return the recorded version entry if this key was already rendered."""
    entry = version_map.get(str(clip_index), {})
    for _ver, rec in (entry.get("versions") or {}).items():
        if rec.get("idempotency_key") == idempotency_key and rec.get("output_file"):
            return rec
    return None


def _record_version(*, job_dir: str, clip_index: int, version: int,
                    project_file: str, output_file: str,
                    idempotency_key: str, origin: str) -> None:
    version_map = _load_version_map(job_dir)
    key = str(clip_index)
    entry = version_map.setdefault(key, {"versions": {}, "latest": 0})
    entry["versions"][str(version)] = {
        "version": int(version),
        "project_file": project_file,
        "output_file": output_file,
        "idempotency_key": idempotency_key,
        "origin": origin,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    entry["latest"] = max(int(entry.get("latest") or 0), int(version))
    _save_version_map(job_dir, version_map)


def _project_filename(clip_index: int, version: int | None = None) -> str:
    if version is None:
        return f"clip-project-{clip_index}.json"
    return f"clip-project-{clip_index}.v{version}.json"


def get_project_path(job_dir: str, clip_index: int,
                     version: int | None = None) -> str | None:
    """Resolve the project file for a clip.

    ``version=None`` -> the latest: the working copy
    (``clip-project-<i>.json``) if present, else the highest versioned file
    from the version map. Otherwise the archived
    ``clip-project-<i>.v<version>.json``. Returns None if nothing exists.
    """
    if version is not None:
        path = os.path.join(job_dir, _project_filename(clip_index, version))
        return path if os.path.isfile(path) else None
    latest = os.path.join(job_dir, _project_filename(clip_index))
    if os.path.isfile(latest):
        return latest
    # Fall back to the highest recorded version (e.g. after a POST render
    # without a prior PUT).
    ver = latest_project_version(job_dir, clip_index)
    if ver:
        path = os.path.join(job_dir, _project_filename(clip_index, ver))
        if os.path.isfile(path):
            return path
    return None


def latest_project_version(job_dir: str, clip_index: int) -> int:
    """Highest version recorded for a clip (0 when none)."""
    entry = _load_version_map(job_dir).get(str(clip_index), {})
    return int(entry.get("latest") or 0)


# ---------------------------------------------------------------------------
# Phase 3b: crossfade transitions + shared draft-save primitive
# ---------------------------------------------------------------------------

FADE_SECONDS = 0.5
"""Crossfade duration (seconds) at a segment boundary whose later segment has
``transition_in="fade"``. Clamped per boundary to half of each adjacent
segment's duration so short segments can't collapse."""


def _boundary_fades(segs: list[dict]) -> list[float]:
    """Fade duration at each segment boundary.

    Returns one float per segment; index 0 is always 0.0 (no previous
    segment). ``segs`` are the materialization dicts (keys: start, end,
    transition_in).
    """
    fades = [0.0]
    for i in range(1, len(segs)):
        if segs[i].get("transition_in") == "fade":
            prev_dur = segs[i - 1]["end"] - segs[i - 1]["start"]
            cur_dur = segs[i]["end"] - segs[i]["start"]
            fades.append(round(min(FADE_SECONDS, prev_dur / 2, cur_dur / 2), 3))
        else:
            fades.append(0.0)
    return fades


def _timeline_layout(segs: list[dict]) -> tuple[list[dict], float]:
    """Map segments to final-timeline coordinates, accounting for crossfades.

    Returns ([{seg, offset, dur, fade_in}], total_duration). A faded segment
    starts ``fade_in`` seconds before the previous segment ends (the fade
    overlaps the boundary); cut segments butt up against the previous end.
    """
    fades = _boundary_fades(segs)
    layout: list[dict] = []
    t = 0.0
    for i, seg in enumerate(segs):
        dur = seg["end"] - seg["start"]
        offset = t - fades[i]
        layout.append({"seg": seg, "offset": round(offset, 3),
                       "dur": round(dur, 3), "fade_in": fades[i]})
        t = offset + dur
    return layout, round(t, 3)


def save_project_version(*, job_dir: str, clip_index: int,
                         project: ClipProject,
                         expected_version: int | None = None) -> tuple[int, ClipProject]:
    """Archive-and-bump: the shared draft-save primitive.

    Archives the pre-existing working copy (e.g. the pipeline-emitted AI
    cut) as a recorded version the first time it is superseded, then writes
    the new working copy + versioned file and records the version. Returns
    (version, project). Used by PUT /api/project/{job}/{clip} and POST
    .../batch-apply -- one implementation, no drift.

    Optimistic concurrency: when expected_version is given, the save is
    rejected with ConflictError (409) unless it matches the latest stored
    version -- the editor's undo/redo and serialized mutation queue rely on
    this to never clobber a newer edit.
    """
    if expected_version is not None:
        latest = latest_project_version(job_dir, clip_index)
        if int(expected_version) != latest:
            raise ConflictError(
                f"stale project version: expected {expected_version}, "
                f"latest is {latest} -- refresh and retry")
    cur_path = get_project_path(job_dir, clip_index)
    if cur_path:
        try:
            with open(cur_path, "r", encoding="utf-8") as f:
                cur_data = json.load(f)
            cur_ver = int(cur_data.get("version") or 0)
            known = (_load_version_map(job_dir)
                     .get(str(clip_index), {}).get("versions", {}))
            if cur_ver and str(cur_ver) not in known:
                ver_path = os.path.join(
                    job_dir, _project_filename(clip_index, cur_ver))
                if not os.path.isfile(ver_path):
                    with open(ver_path, "w", encoding="utf-8") as f:
                        json.dump(cur_data, f, indent=2, ensure_ascii=False)
                _record_version(
                    job_dir=job_dir, clip_index=clip_index,
                    version=cur_ver,
                    project_file=os.path.basename(ver_path),
                    output_file="",
                    idempotency_key=cur_data.get("idempotency_key") or "",
                    origin=cur_data.get("origin") or "auto")
        except (OSError, ValueError):
            pass
    version = latest_project_version(job_dir, clip_index) + 1
    project.version = version
    if not project.idempotency_key:
        project.idempotency_key = str(uuid.uuid4())
    payload = project_to_json(project)
    latest_path = os.path.join(job_dir, _project_filename(clip_index))
    versioned_path = os.path.join(
        job_dir, _project_filename(clip_index, version))
    for target in (latest_path, versioned_path):
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, target)
    _record_version(
        job_dir=job_dir, clip_index=clip_index, version=version,
        project_file=os.path.basename(versioned_path),
        output_file="",  # draft: no render yet
        idempotency_key=project.idempotency_key,
        origin=project.origin,
    )
    return version, project


# ---------------------------------------------------------------------------
# Timeline materialization
# ---------------------------------------------------------------------------

def _has_audio_stream(path: str) -> bool:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=index", "-of", "csv=p=0", path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return bool(r.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _clamp_crop(crop, src_w: int, src_h: int) -> tuple[int, int, int, int]:
    """Clamp a crop box into the source frame; return ints."""
    x = max(0, min(int(crop.x), src_w - 1))
    y = max(0, min(int(crop.y), src_h - 1))
    w = max(2, min(int(crop.w), src_w - x))
    h = max(2, min(int(crop.h), src_h - y))
    # keep even for yuv420p
    w -= w % 2
    h -= h % 2
    return x, y, w, h


def _build_timeline_filter(segments, render_w: int, render_h: int,
                           render_fps: int, has_audio: bool) -> tuple[str, list[str]]:
    """Build a single filtergraph: trim+crop+scale each segment, then join.

    Cut boundaries use concat (byte-identical to the pre-3b graph). A
    boundary whose later segment has ``transition_in="fade"`` uses an
    xfade/acrossfade chain instead (~0.5s, clamped per boundary).
    """
    layout, _total = _timeline_layout(segments)
    fades = [item["fade_in"] for item in layout]
    use_fades = any(f > 0 for f in fades)
    n = len(segments)

    filt: list[str] = []
    for i, seg in enumerate(segments):
        x, y, w, h = seg["_crop_int"]
        v = (f"[0:v]trim=start={seg['start']:.3f}:end={seg['end']:.3f},"
             f"setpts=PTS-STARTPTS,"
             f"crop={w}:{h}:{x}:{y},"
             f"scale={render_w}:{render_h}:flags=lanczos")
        if use_fades:
            # xfade needs CFR + identical pix fmt/timebase on both inputs.
            v += f",fps={render_fps},format=yuv420p,settb=AVTB"
        filt.append(v + f"[v{i}]")
        if has_audio:
            a = (f"[0:a]atrim=start={seg['start']:.3f}:end={seg['end']:.3f},"
                 f"asetpts=PTS-STARTPTS")
            filt.append(a + f"[a{i}]")

    if not use_fades:
        # Cut-only path: exactly the pre-3b filtergraph.
        vcat = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[vcat]"
        # Original order: video parts, vcat, audio parts, acat.
        parts = [f for f in filt if "[v" in f and "[a" not in f]
        parts.append(vcat)
        maps = ["-map", "[vcat]"]
        if has_audio:
            acat = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[acat]"
            parts += [f for f in filt if "[a" in f] + [acat]
            maps += ["-map", "[acat]"]
        return ";".join(parts), maps

    # Fade path: merge runs of faded segments with xfade/acrossfade; join the
    # resulting pieces (and any cut-isolated segments) with concat.
    v_pieces: list[str] = []
    a_pieces: list[str] = []
    vcur = "[v0]"
    acur = "[a0]" if has_audio else None
    run_start = layout[0]["offset"]
    for i in range(1, n):
        f = fades[i]
        if f > 0:
            off = layout[i]["offset"] - run_start
            vnew = f"[vx{i}]"
            filt.append(
                f"{vcur}[v{i}]xfade=offset={off:.3f}:duration={f:.3f}"
                f":transition=fade{vnew}")
            vcur = vnew
            if has_audio:
                anew = f"[ax{i}]"
                filt.append(
                    f"{acur}[a{i}]acrossfade=d={f:.3f}:c1=tri:c2=tri{anew}")
                acur = anew
        else:
            v_pieces.append(vcur)
            if has_audio:
                a_pieces.append(acur)
            vcur = f"[v{i}]"
            acur = f"[a{i}]" if has_audio else None
            run_start = layout[i]["offset"]
    v_pieces.append(vcur)
    if has_audio:
        a_pieces.append(acur)
    vcat = "".join(v_pieces) + f"concat=n={len(v_pieces)}:v=1:a=0[vcat]"
    filt.append(vcat)
    maps = ["-map", "[vcat]"]
    if has_audio:
        acat = "".join(a_pieces) + f"concat=n={len(a_pieces)}:v=0:a=1[acat]"
        filt.append(acat)
        maps += ["-map", "[acat]"]
    return ";".join(filt), maps


async def _materialize_timeline(project: ClipProject, job_dir: str,
                                clip_index: int) -> tuple[str, float]:
    """Cut/crop/concat segments from source.file -> 9:16 timeline mp4.

    Returns (timeline_path, timeline_duration). One ffmpeg generation,
    frame-accurate trims.
    """
    src_path = os.path.join(job_dir, project.source.file)
    if not os.path.isfile(src_path):
        raise ProjectRenderError(
            f"project source file not found: {project.source.file!r}")

    src_probe = probe_media(src_path)
    src_w = int(src_probe.get("width") or project.source.width)
    src_h = int(src_probe.get("height") or project.source.height)
    src_dur = float(src_probe.get("duration") or 0)

    segs = []
    for seg in project.segments:
        if src_dur and seg.start >= src_dur:
            raise ProjectRenderError(
                f"segment {seg.id!r} start {seg.start}s is beyond source "
                f"duration {src_dur:.2f}s")
        end = min(seg.end, src_dur) if src_dur else seg.end
        if end <= seg.start:
            raise ProjectRenderError(
                f"segment {seg.id!r} has non-positive duration after "
                f"clamping to source")
        segs.append({
            "start": seg.start,
            "end": end,
            "transition_in": seg.transition_in,
            "_crop_int": _clamp_crop(seg.crop, src_w, src_h),
        })

    render_w = project.render.width
    render_h = project.render.height
    has_audio = await asyncio.to_thread(_has_audio_stream, src_path)
    filt, maps = _build_timeline_filter(
        segs, render_w, render_h, project.render.fps, has_audio)

    timeline_path = os.path.join(
        job_dir, f".timeline_{clip_index}_{project.version}.mp4")
    cmd = (["ffmpeg", "-y", "-i", src_path,
            "-filter_complex", filt] + maps +
           ["-r", str(project.render.fps)] +
           x264_video_args(faststart=False) +
           (["-c:a", "aac", "-ar", "44100", "-ac", "2"] if has_audio
            else ["-an"]) +
           [timeline_path])
    logger.info("project_render: materializing %d segment(s) -> %s",
                len(segs), os.path.basename(timeline_path))
    r = await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(timeline_path):
        raise ProjectRenderError(
            f"timeline materialization failed: {(r.stderr or '')[-800:]}")

    _layout, duration = _timeline_layout(segs)
    return timeline_path, duration


# ---------------------------------------------------------------------------
# Word / overlay mapping (source-file coords -> timeline coords)
# ---------------------------------------------------------------------------

def _map_words_to_timeline(project: ClipProject) -> tuple[list[dict], float]:
    """Apply caption edits, then map words from source coords to timeline.

    Returns (mapped_words, timeline_duration). Words outside all segments
    are dropped. Edits apply to the original words-list indices first.
    """
    words = [w.model_dump() for w in project.captions.words]
    # Edits are keyed by stable word ID; legacy word-index strings still work.
    by_id = {w.get("id"): i for i, w in enumerate(words) if w.get("id")}
    for key, corrected in (project.captions.edits or {}).items():
        idx = by_id.get(key)
        if idx is None:
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if not 0 <= idx < len(words):
                continue
        words[idx]["w"] = corrected

    # 3b: timeline offsets account for crossfade overlaps (a faded segment
    # starts its fade_in before the previous segment ends).
    seg_dicts = [{"start": s.start, "end": s.end,
                  "transition_in": s.transition_in}
                 for s in project.segments]
    layout, timeline_duration = _timeline_layout(seg_dicts)

    mapped: list[dict] = []
    for w in words:
        ws, we = float(w["start"]), float(w["end"])
        text = (w.get("w") or "").strip()
        if not text or we <= ws:
            continue
        for item in layout:
            s0 = item["seg"]["start"]
            s1 = item["seg"]["end"]
            off = item["offset"]
            if s0 <= ws < s1:
                mapped.append({
                    "word": text,
                    "start": round(off + (ws - s0), 3),
                    "end": round(off + (min(we, s1) - s0), 3),
                })
                break
    # drop zero-length after rounding
    mapped = [w for w in mapped if w["end"] > w["start"]]
    return mapped, timeline_duration


# ---------------------------------------------------------------------------
# Project -> compose inputs
# ---------------------------------------------------------------------------

def _derive_compose_inputs(project: ClipProject, timeline_duration: float,
                           metadata: dict, clip_info: dict):
    """Derive (toggles, params, synth_metadata, synth_clip_info) from a project.

    Words are mapped to timeline coordinates and injected as a synthetic
    transcript so the existing subtitle burner works unchanged (it windows
    by clip_info start/end).
    """
    mapped_words, _ = _map_words_to_timeline(project)
    language = (metadata.get("transcript") or {}).get("language")
    synth_metadata = {
        **metadata,
        "transcript": {
            "segments": [{"words": mapped_words}],
            "language": language,
        },
    }
    synth_clip_info = {**clip_info, "start": 0.0, "end": timeline_duration}

    caps = project.captions
    toggles: dict = {}

    has_words = bool(mapped_words)
    toggles["subtitles"] = has_words
    subtitle_params = {
        "preset": caps.style,
        "position": caps.position,
        "mode": "karaoke",
    } if has_words else {}

    hook_overlays = [o for o in project.overlays
                     if o.type == "hook" and (o.text or "").strip()]
    toggles["hook"] = bool(hook_overlays)
    hook_params = ({
        "text": hook_overlays[0].text.strip(),
        "style": hook_overlays[0].style,
        "position": hook_overlays[0].position,
    } if hook_overlays else {})

    grade_active = project.grade.preset not in ("default", "none", "")
    toggles["grade"] = grade_active
    grade_params = {"preset": project.grade.preset} if grade_active else {}

    banner_overlays = [o for o in project.overlays
                       if o.type == "banner" and (o.text or "").strip()]
    toggles["banner"] = bool(banner_overlays)
    banner_params = ({"enabled": True, "text": banner_overlays[0].text}
                     if banner_overlays else {})

    toggles["logo"] = False
    toggles["smartcut"] = False
    logo_params: dict = {}

    return (toggles, hook_params, subtitle_params, logo_params,
            grade_params, banner_params, synth_metadata, synth_clip_info)


# ---------------------------------------------------------------------------
# Music mix
# ---------------------------------------------------------------------------

async def _mix_music(composed_path: str, project: ClipProject,
                     job_dir: str, clip_index: int) -> str:
    """Mix project.audio[0] under the composed clip. Returns final path.

    v1: a null file skips silently. A missing non-null file is a 400.
    """
    track = project.audio[0] if project.audio else None
    if not track or not track.file:
        return composed_path
    music_path = (track.file if os.path.isabs(track.file)
                  else os.path.join(job_dir, track.file))
    if not os.path.isfile(music_path):
        raise ProjectRenderError(f"music file not found: {track.file!r}")
    offset_ms = int(track.offset * 1000)
    out_path = os.path.join(
        job_dir, f".musicmix_{clip_index}_{project.version}.mp4")
    # 3b: the composed clip may have no audio stream (silent source) -- then
    # the music becomes the audio track instead of being mixed under it.
    if _has_audio_stream(composed_path):
        filt = (f"[1:a]adelay={offset_ms}|{offset_ms},"
                f"volume={track.volume}[mus];"
                f"[0:a][mus]amix=inputs=2:duration=first:dropout_transition=0[a]")
    else:
        filt = (f"[1:a]adelay={offset_ms}|{offset_ms},"
                f"volume={track.volume}[a]")
    cmd = (["ffmpeg", "-y", "-i", composed_path, "-i", music_path,
            "-filter_complex", filt, "-map", "0:v", "-map", "[a]"] +
           x264_video_args(faststart=False) +
           ["-c:a", "aac", "-shortest", out_path])
    r = await asyncio.to_thread(
        subprocess.run, cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not os.path.isfile(out_path):
        raise ProjectRenderError(
            f"music mix failed: {(r.stderr or '')[-800:]}")
    return out_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def render_project(*, job_id: str, clip_index: int,
                         project: ClipProject, job_dir: str,
                         metadata: dict, clip_info: dict,
                         drop_ranges=None,
                         naming: str = "versioned") -> ProjectRenderResult:
    """Render a ClipProject to a NEW versioned mp4.

    ``naming="versioned"``: ``<composed_stem>.edited_v<version>.mp4`` + version
    map entry (project path). ``naming="legacy"``: the existing
    ``composed_clip_{i}.mp4`` convention (legacy toggles path -- byte-identical
    behavior preserved).

    Idempotency: a re-POST with a previously-seen ``idempotency_key`` returns
    the recorded output without re-rendering.
    """
    if naming not in ("versioned", "legacy"):
        raise ProjectRenderError(f"unknown output naming: {naming!r}")

    # Effective layer order is always canonical for projects (the schema has
    # no custom order field); the check exists so a future field can't
    # silently violate timing constraints.
    validate_layer_order(None)

    version_map = _load_version_map(job_dir)
    if naming == "versioned" and project.idempotency_key:
        dup = _find_by_idempotency_key(version_map, clip_index,
                                       project.idempotency_key)
        if dup:
            out_path = os.path.join(job_dir, dup["output_file"])
            if os.path.isfile(out_path):
                logger.info("project_render: idempotent hit for key %s",
                            project.idempotency_key)
                return ProjectRenderResult(
                    dup["output_file"], out_path,
                    version=int(dup.get("version") or project.version),
                    deduped=True)

    timeline_tmp: list[str] = []
    try:
        # NOTE: no clip_lock here -- compose_layers() acquires it internally,
        # and asyncio.Lock is not reentrant (would deadlock). Materialization
        # uses unique temp filenames; version recording is atomic.
        if True:
            # ---- 1. base timeline ----
            if project.base_clip:
                base_path = os.path.join(job_dir, project.base_clip)
                if not os.path.isfile(base_path):
                    raise ProjectRenderError(
                        f"project base_clip not found: {project.base_clip!r}")
                probe = probe_media(base_path)
                timeline_duration = float(probe.get("duration") or 0)
                if not timeline_duration:
                    raise ProjectRenderError(
                        f"cannot probe duration of {project.base_clip!r}")
                eff_metadata, eff_clip_info = metadata, clip_info
                legacy = project.legacy_params or {}
                toggles = legacy.get("toggles", {})
                hook_params = legacy.get("hook_params", {})
                subtitle_params = legacy.get("subtitle_params", {})
                logo_params = legacy.get("logo_params", {})
                grade_params = legacy.get("grade_params", {})
                banner_params = legacy.get("banner_params", {})
                eff_drop_ranges = (legacy["drop_ranges"]
                                   if "drop_ranges" in legacy else drop_ranges)
            else:
                base_path, timeline_duration = await _materialize_timeline(
                    project, job_dir, clip_index)
                timeline_tmp.append(base_path)
                (toggles, hook_params, subtitle_params, logo_params,
                 grade_params, banner_params,
                 eff_metadata, eff_clip_info) = _derive_compose_inputs(
                    project, timeline_duration, metadata, clip_info)
                eff_drop_ranges = drop_ranges

            # ---- 2. compose layers (the ONE layer implementation) ----
            composed_basename = await compose_layers(
                base_clip=base_path,
                job_dir=job_dir,
                clip_index=clip_index,
                metadata=eff_metadata,
                clip_info=eff_clip_info,
                toggles=toggles,
                hook_params=hook_params,
                subtitle_params=subtitle_params,
                logo_params=logo_params,
                grade_params=grade_params,
                banner_params=banner_params,
                drop_ranges=eff_drop_ranges,
            )
            composed_path = os.path.join(job_dir, composed_basename)

            # ---- 3. music (project path only) ----
            if not project.base_clip:
                mixed = await _mix_music(
                    composed_path, project, job_dir, clip_index)
                if mixed != composed_path:
                    timeline_tmp.append(mixed)
                    composed_path = mixed

            # ---- 4. QA ----
            report = inspect_clip(
                composed_path,
                expected_duration=None,
                expected_aspect=9 / 16,
                run_signal_checks=False,
            )
            if report.get("critical"):
                raise ProjectRenderError(
                    f"rendered clip failed QA: {report.get('critical')}")

            # ---- 5. versioned output ----
            if naming == "versioned":
                from clippyme.domain.clip_resolve import composed_clip_basename
                stem = composed_clip_basename(
                    clip_info, clip_index).rsplit(".mp4", 1)[0]
                out_basename = f"{stem}.edited_v{project.version}.mp4"
                out_path = os.path.join(job_dir, out_basename)
                # atomic publish
                tmp = out_path + ".tmp"
                with open(composed_path, "rb") as fsrc, \
                        open(tmp, "wb") as fdst:
                    fdst.write(fsrc.read())
                os.replace(tmp, out_path)
                project_file = _project_filename(clip_index, project.version)
                with open(os.path.join(job_dir, project_file), "w",
                          encoding="utf-8") as f:
                    f.write(project_to_json(project))
                _record_version(
                    job_dir=job_dir, clip_index=clip_index,
                    version=project.version, project_file=project_file,
                    output_file=out_basename,
                    idempotency_key=project.idempotency_key,
                    origin=project.origin)
            else:
                out_basename = composed_basename
                out_path = composed_path

            logger.info("project_render: job %s clip %d -> %s",
                        job_id, clip_index, out_basename)
            return ProjectRenderResult(out_basename, out_path,
                                       version=project.version)
    finally:
        for p in timeline_tmp:
            try:
                if p and os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
