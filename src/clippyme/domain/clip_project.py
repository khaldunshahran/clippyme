"""clip-project.json schema v1 — the editor/server render contract.

``clip-project.json`` is the single language the (future) clip editor UI and the
server-side render farm speak. The auto-pipeline emits one per clip as the
"default edit" (Phase 1); the editor loads it, the user tweaks it, and
``POST /api/compose`` renders it deterministically (Phase 2+).

Design principles (EDITOR_PLAN.md §1):
- Same JSON in -> same pixels out. The schema is versioned (``schema`` tag +
  ``version``) so the renderer can evolve without breaking saved projects.
- v1 is purpose-built and minimal. OpenTimelineIO was considered; we are not
  paying the abstraction tax until interchange is actually needed.
- The browser decides, the server renders: nothing in here requires
  client-side encoding.

Validation: ``validate_project()`` raises pydantic ``ValidationError`` with
field-level detail. A JSON Schema twin (``clip_project.schema.json``) is
emitted via ``ClipProject.model_json_schema()`` for the future UI to validate
against before POSTing.
"""

from __future__ import annotations

import json
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

SCHEMA_TAG = "nugget.clip-project/1"


class CropBox(BaseModel):
    """9:16 crop window inside the 16:9 source slice (source pixel coords)."""

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    w: float = Field(gt=0)
    h: float = Field(gt=0)


class ProjectSegment(BaseModel):
    """One timeline segment. v1: editor works single-segment; schema allows many."""

    id: str
    start: float = Field(ge=0, description="source seconds, inclusive")
    end: float = Field(gt=0, description="source seconds, exclusive")
    crop: CropBox
    transition_in: Literal["cut", "fade"] = "cut"  # 3b: "fade" = 0.5s crossfade

    @field_validator("end")
    @classmethod
    def _end_after_start(cls, v: float, info) -> float:
        start = (info.data or {}).get("start")
        if start is not None and v <= start:
            raise ValueError(f"segment end ({v}) must be after start ({start})")
        return v


class CaptionWord(BaseModel):
    w: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class Captions(BaseModel):
    style: str = "hormozi"  # mirrors dashboard seedClipParams styles
    position: str = "lower-third"
    words: list[CaptionWord] = Field(default_factory=list)
    edits: dict[str, str] = Field(
        default_factory=dict,
        description="word-index (as string) -> corrected text",
    )


class Overlay(BaseModel):
    """Text overlays: hooks, banners."""

    type: Literal["hook", "banner"]
    text: str
    style: str = "viral"
    start: float = Field(ge=0, description="clip-relative seconds")
    end: float = Field(ge=0)
    position: str = "top"


class AudioTrack(BaseModel):
    file: Optional[str] = Field(
        default=None, description="music file; null = no music (v1)"
    )
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    offset: float = Field(default=0.0, ge=0.0)


class Grade(BaseModel):
    preset: str = "default"


class SourceRef(BaseModel):
    file: str = Field(description="16:9 source slice on disk, e.g. source_<clip>.mp4")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration: Optional[float] = Field(
        default=None, ge=0,
        description="source slice duration in seconds (informational)")


class RenderSpec(BaseModel):
    width: int = Field(default=608, gt=0)
    height: int = Field(default=1080, gt=0)
    fps: int = Field(default=30, gt=0)
    encoder: Literal["nvenc", "libx264"] = "nvenc"


class ClipProject(BaseModel):
    # NOTE: the field name "schema" intentionally matches the plan's contract
    # ("nugget.clip-project/1"). Pydantic v2 emits a harmless UserWarning that
    # it shadows the deprecated BaseModel.schema attribute; the field works
    # as intended. Do NOT rename it.
    schema: Literal["nugget.clip-project/1"]  # required: format-version contract; no default
    version: int = Field(default=1, ge=1, description="increments per save")
    origin: Literal["auto", "user", "legacy"] = "auto"
    job_id: str
    clip_index: int = Field(ge=0)
    idempotency_key: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: SourceRef
    segments: list[ProjectSegment] = Field(min_length=1)
    captions: Captions = Field(default_factory=Captions)
    overlays: list[Overlay] = Field(default_factory=list)
    audio: list[AudioTrack] = Field(default_factory=lambda: [AudioTrack()])
    grade: Grade = Field(default_factory=Grade)
    render: RenderSpec = Field(default_factory=RenderSpec)
    # Phase 2: legacy-mapped projects only. When set, the renderer uses
    # base_clip directly (no timeline materialization) and the exact stashed
    # params -- byte-identical to the pre-Phase-2 render path. The segments /
    # captions / overlays above remain informational (full-range segment,
    # transcript words, toggle-derived flags) so the editor can display and
    # evolve the project.
    base_clip: Optional[str] = Field(default=None)
    legacy_params: Optional[dict] = Field(default=None)


def validate_project(data: dict) -> ClipProject:
    """Validate raw dict data against the v1 schema.

    Raises:
        pydantic.ValidationError: with field-level detail on bad input.
    """
    if not isinstance(data, dict):
        raise TypeError(f"clip project must be a dict, got {type(data).__name__}")
    return ClipProject.model_validate(data)


def project_to_json(project: ClipProject) -> str:
    """Serialise a validated project to canonical JSON text."""
    return project.model_dump_json(indent=2)


def project_from_json(text: str) -> ClipProject:
    """Parse + validate JSON text into a ClipProject."""
    return ClipProject.model_validate_json(text)


def project_json_schema() -> dict:
    """Return the JSON Schema twin (for the future editor UI)."""
    return ClipProject.model_json_schema()
def _invert_ranges(drop_ranges, duration: float) -> list[tuple[float, float]]:
    """Invert drop ranges into kept segments (source-file coordinates)."""
    drops = []
    for rng in drop_ranges or []:
        try:
            a, b = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            drops.append((a, b))
    drops.sort()
    kept: list[tuple[float, float]] = []
    cur = 0.0
    for a, b in drops:
        if a > cur:
            kept.append((cur, min(a, duration)))
        cur = max(cur, b)
    if cur < duration:
        kept.append((cur, duration))
    return [(s, e) for s, e in kept if e > s]


def toggles_to_project(
    *,
    job_id: str,
    clip_index: int,
    base_clip: str,
    source_file: str,
    source_width: int,
    source_height: int,
    source_duration: float,
    render_width: int = 608,
    render_height: int = 1080,
    fps: int = 30,
    toggles: dict | None = None,
    hook_params: dict | None = None,
    subtitle_params: dict | None = None,
    logo_params: dict | None = None,
    grade_params: dict | None = None,
    banner_params: dict | None = None,
    drop_ranges: list | None = None,
    metadata: dict | None = None,
    version: int = 1,
) -> ClipProject:
    """Build the ClipProject equivalent of a legacy /api/compose toggles call.

    The returned project carries ``base_clip`` + ``legacy_params`` so
    ``render_project`` renders it byte-identically to the pre-Phase-2 path
    (no timeline materialization, exact original params). The segments /
    captions / overlays are informational -- a full-range segment (minus drop
    ranges), the transcript words, and toggle-derived layer flags -- so the
    editor can display and evolve the project.
    """
    toggles = dict(toggles or {})
    hook_params = dict(hook_params or {})
    subtitle_params = dict(subtitle_params or {})
    logo_params = dict(logo_params or {})
    grade_params = dict(grade_params or {})
    banner_params = dict(banner_params or {})
    drop_ranges = list(drop_ranges or [])

    kept = _invert_ranges(drop_ranges, source_duration)
    if not kept:
        kept = [(0.0, source_duration)]
    segments = [
        ProjectSegment(
            id=f"seg{i}",
            start=round(s, 3),
            end=round(e, 3),
            crop=CropBox(x=0.0, y=0.0, w=float(source_width),
                         h=float(source_height)),
            transition_in="cut",
        )
        for i, (s, e) in enumerate(kept)
    ]

    words: list[CaptionWord] = []
    language = None
    transcript = (metadata or {}).get("transcript") or {}
    if isinstance(transcript, dict):
        language = transcript.get("language")
        for tseg in transcript.get("segments") or []:
            for w in tseg.get("words") or []:
                text = (w.get("word") or "").strip()
                if not text:
                    continue
                try:
                    ws, we = float(w.get("start", 0)), float(w.get("end", 0))
                except (TypeError, ValueError):
                    continue
                if we > ws >= 0:
                    words.append(CaptionWord(w=text, start=ws, end=we))

    overlays: list[Overlay] = []
    if toggles.get("hook") and (hook_params.get("text") or "").strip():
        overlays.append(Overlay(
            id="ov-hook",
            type="hook",
            text=hook_params["text"].strip(),
            style=hook_params.get("style", "viral"),
            start=0.0,
            end=round(min(3.0, source_duration), 3),
            position="top",
        ))
    if toggles.get("banner") and (banner_params.get("text") or "").strip():
        overlays.append(Overlay(
            id="ov-banner",
            type="banner",
            text=banner_params["text"].strip(),
            style="viral",
            start=0.0,
            end=round(source_duration, 3),
            position="top",
        ))

    grade_preset = (grade_params.get("preset", "default")
                    if toggles.get("grade") else "default")

    return ClipProject(
        schema=SCHEMA_TAG,
        version=version,
        origin="legacy",
        job_id=job_id,
        clip_index=clip_index,
        source=SourceRef(
            file=source_file,
            width=int(source_width),
            height=int(source_height),
            duration=float(source_duration),
        ),
        render=RenderSpec(
            width=int(render_width),
            height=int(render_height),
            fps=int(fps),
            encoder="nvenc",
        ),
        segments=segments,
        captions=Captions(
            style=subtitle_params.get("preset", "classic_white"),
            position=subtitle_params.get("position", "bottom"),
            words=words,
            edits={},
        ),
        overlays=overlays,
        audio=[AudioTrack()],
        grade=Grade(preset=grade_preset),
        base_clip=base_clip,
        legacy_params={
            "toggles": toggles,
            "hook_params": hook_params,
            "subtitle_params": subtitle_params,
            "logo_params": logo_params,
            "grade_params": grade_params,
            "banner_params": banner_params,
            "drop_ranges": drop_ranges,
        },
    )


# ---------------------------------------------------------------------------
# Phase 3b: batch-apply patch support
# ---------------------------------------------------------------------------

CAPTION_STYLES = frozenset({
    "classic_white", "hormozi_bold", "neon_glow",
    "mrbeast_box", "minimal_clean", "fire_impact",
})
"""Caption style ids (mirrors dashboard SUBTITLE_PRESETS in data.js)."""

CAPTION_POSITIONS = frozenset({"top", "middle", "bottom"})

GRADE_PRESETS = frozenset({
    "default", "warm_cinematic", "cool_crisp",
    "neutral_punch", "vivid_pop",
})
"""Grade preset ids (mirrors dashboard GRADE_PRESETS in data.js)."""

BATCH_PATCH_KEYS = frozenset({
    "caption_style", "caption_position", "hook_text", "grade_preset",
})


def apply_batch_patch(project: "ClipProject", patch: dict) -> "ClipProject":
    """Apply a batch-edit patch to a project (mutates + returns it).

    Supported keys: caption_style, caption_position, hook_text, grade_preset.
    hook_text: non-empty -> set (or create) the hook overlay; empty/missing
    -> no-op (batch never deletes hooks silently).
    Raises ValueError on unknown keys or invalid values -- never silently
    corrupts.
    """
    if not isinstance(patch, dict) or not patch:
        raise ValueError("patch must be a non-empty dict")
    unknown = set(patch) - BATCH_PATCH_KEYS
    if unknown:
        raise ValueError(f"unknown patch keys: {sorted(unknown)}")

    if "caption_style" in patch:
        style = patch["caption_style"]
        if style not in CAPTION_STYLES:
            raise ValueError(f"unknown caption style: {style!r}")
        project.captions.style = style

    if "caption_position" in patch:
        pos = patch["caption_position"]
        if pos not in CAPTION_POSITIONS:
            raise ValueError(f"unknown caption position: {pos!r}")
        project.captions.position = pos

    if "hook_text" in patch:
        text = patch["hook_text"]
        if not isinstance(text, str):
            raise ValueError("hook_text must be a string")
        text = text.strip()
        if text:
            hook = next((o for o in project.overlays if o.type == "hook"), None)
            if hook is not None:
                hook.text = text
            else:
                dur = project.source.duration or 10.0
                project.overlays.append(Overlay(
                    type="hook", text=text, style="viral",
                    start=0.0, end=round(min(3.0, dur), 3), position="top"))

    if "grade_preset" in patch:
        preset = patch["grade_preset"]
        if preset not in GRADE_PRESETS:
            raise ValueError(f"unknown grade preset: {preset!r}")
        project.grade.preset = preset

    return project
