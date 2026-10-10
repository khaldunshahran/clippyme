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
import logging
import re
import uuid
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_TAG = "nugget.clip-project/1"

logger = logging.getLogger(__name__)


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
    # Stable word ID (e.g. "w12"): assigned at project creation from
    # Captions.next_word_id, persisted, never reused. Splits/drops never
    # renumber; re-transcription carries IDs over by (rounded start, text).
    # All word-keyed patch data (edits, and later slices' keys) uses IDs.
    id: str = Field(default="", description="stable word ID, e.g. 'w12'")
    w: str
    start: float = Field(ge=0, description="source seconds")
    end: float = Field(ge=0, description="source seconds")
    removed: bool = Field(
        default=False,
        description="word carved out of the timeline; kept with its ID for "
                    "strikethrough/restore (ID-keyed data like colors/emoji "
                    "survives)",
    )


class Captions(BaseModel):
    style: str = "hormozi"  # mirrors dashboard seedClipParams styles
    position: str = "lower-third"
    words: list[CaptionWord] = Field(default_factory=list)
    edits: dict[str, str] = Field(
        default_factory=dict,
        description="word ID (or legacy word-index as string) -> corrected text",
    )
    next_word_id: int = Field(
        default=0,
        ge=0,
        description="persisted counter; fresh word IDs are f'w{next_word_id}', then it increments",
    )


class Overlay(BaseModel):
    """Text overlays: hooks, banners."""

    type: Literal["hook", "banner"]
    text: str
    style: str = "viral"
    start: float = Field(ge=0, description="clip-relative seconds")
    end: float = Field(ge=0)
    position: str = "top"


class OpeningEvidenceRef(BaseModel):
    """One cited evidence item behind a grounded opening option."""

    basis: Literal["speech", "visual", "audio"]
    quote: Optional[str] = Field(default=None)
    label: Optional[str] = Field(default=None)
    start: Optional[float] = Field(default=None)
    end: Optional[float] = Field(default=None)


class OpeningSelection(BaseModel):
    """Task 15: grounded opening (hook) identity carried by the project.

    Additive optional field. ``status="selected"`` carries the validated
    selected option (identity, evidence, clip-relative display timing,
    packet identity, versions). ``status="user_authored"`` marks an
    explicit user-authored override. ``status="none"`` records an honest
    no-selection with the reason. ``legacy_unverified_hook`` keeps the
    old automatic hook text only as clearly unverified legacy metadata —
    never promoted as grounded.
    """

    status: Literal["selected", "user_authored", "none"]
    option_id: Optional[str] = Field(default=None)
    text: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    display_start: Optional[float] = Field(default=None, ge=0)
    display_end: Optional[float] = Field(default=None, ge=0)
    evidence: list[OpeningEvidenceRef] = Field(default_factory=list)
    evidence_basis: Optional[Literal["speech", "visual", "audio"]] = Field(
        default=None)
    packet_id: Optional[str] = Field(default=None)
    schema_version: int = Field(default=1, ge=1)
    selection: Optional[dict] = Field(default=None)
    no_selection_reason: Optional[str] = Field(default=None)
    legacy_unverified_hook: Optional[str] = Field(default=None)


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


def _backfill_word_ids(data: dict, *, repair_duplicates: bool = False) -> dict:
    """Migration for stable word IDs.

    - Assigns missing word IDs as f'w{counter}', advancing the counter.
    - Advances the counter past any numeric IDs already present (w{N}).
    - Duplicate word IDs: raises ValueError by default (write path --
      validate_project rejects). When repair_duplicates=True (read path),
      reassigns fresh IDs from the counter to the duplicates, bumps the
      counter past them, and logs a warning instead of raising.

    Idempotent: already-migrated projects pass through unchanged.
    """
    if not isinstance(data, dict):
        return data
    caps = data.get("captions")
    if not isinstance(caps, dict):
        return data
    words = caps.get("words") or []
    try:
        counter = int(caps.get("next_word_id") or 0)
    except (TypeError, ValueError):
        counter = 0
    if counter < 0:
        counter = 0
    for w in words:
        wid = (w.get("id") or "") if isinstance(w, dict) else ""
        if isinstance(wid, str) and wid.startswith("w"):
            try:
                counter = max(counter, int(wid[1:]) + 1)
            except ValueError:
                pass
    seen: set[str] = set()
    repaired: list[str] = []
    for w in words:
        if not isinstance(w, dict):
            continue
        wid = (w.get("id") or "").strip()
        if not wid:
            wid = f"w{counter}"
            counter += 1
            w["id"] = wid
        if wid in seen:
            if not repair_duplicates:
                raise ValueError(f"duplicate word id: {wid!r}")
            # Read-path repair: give the duplicate a fresh ID from the
            # counter (already advanced past every w{N} above, so this
            # cannot collide) and keep going.
            repaired.append(wid)
            wid = f"w{counter}"
            counter += 1
            w["id"] = wid
        seen.add(wid)
    if repaired:
        logger.warning(
            "repaired %d duplicate word id(s) on read, reassigned fresh ids: %s",
            len(repaired), ", ".join(sorted(set(repaired))),
        )
    caps["next_word_id"] = counter
    return data


class ReframedBase(BaseModel):
    """Lineage for automatic framing preservation (Task 05).

    Optional. When present on an auto-emitted project, records the
    original reframed clip (with tracked-speaker/layout framing) plus
    the geometry/timeline baseline and file identity needed to detect
    missing, stale, or replaced assets.

    Eligibility is semantic equivalence to the baseline; see
    ``domain.project_lineage.check_lineage_eligible``.
    """

    base_ref: str = Field(
        description="job-relative path to the reframed base clip")
    identity: dict = Field(
        default_factory=dict,
        description="file identity: file/size/mtime for staleness detection")
    source: dict = Field(
        default_factory=dict,
        description="source baseline: file/width/height")
    segments: list[dict] = Field(
        default_factory=list,
        description="ordered segment geometry baseline")
    render: dict = Field(
        default_factory=dict,
        description="render geometry baseline: width/height/fps")
    # Task 13 (pipeline integration): framing trajectory baseline identity
    # (the 8-key subset produced by
    # domain.project_framing.framing_baseline_identity). Optional; None for
    # old emissions. Strictly validated by
    # domain.project_lineage._validate_baseline at eligibility time.
    framing: Optional[dict] = Field(
        default=None,
        description="Task 13 framing baseline identity: plan_ref, sha256, "
        "source_fingerprint, config_digest, tracker_digest, output_aspect, "
        "analysis_complete, timeline_digest")


# ---------------------------------------------------------------------------
# Task 13: stable framing artifact bridge (additive, optional).
# ---------------------------------------------------------------------------

_TRACK_ID_RE = re.compile(r"^t-\d+$")


def _reject_bool_numeric(v):
    """Shared mode="before" guard: JSON true/false is never a number."""
    if isinstance(v, bool):
        raise ValueError("boolean is not a valid numeric value")
    return v


class FramingKeyframe(BaseModel):
    """One crop target in clip-relative inclusive frame coordinates."""

    model_config = ConfigDict(allow_inf_nan=False)

    frame: int = Field(ge=0)
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)
    kind: Literal["lock", "ease"]

    @field_validator("frame", "x1", "y1", "x2", "y2", mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)

    @model_validator(mode="after")
    def _box_order(self):
        if not self.x1 < self.x2:
            raise ValueError(f"x1 ({self.x1}) must be < x2 ({self.x2})")
        if not self.y1 < self.y2:
            raise ValueError(f"y1 ({self.y1}) must be < y2 ({self.y2})")
        return self


class FramingFitRange(BaseModel):
    """One conservative full-frame fit range (Task 14 rev4).

    Clip-relative INCLUSIVE ``[start_frame, end_frame]``. Additive: old
    projects without fit ranges validate unchanged (the ``fit`` bool on
    the scene timeline is still honored for legacy timelines).
    """

    model_config = ConfigDict(allow_inf_nan=False)

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)

    @field_validator("start_frame", "end_frame", mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)

    @model_validator(mode="after")
    def _order(self):
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be <= end_frame "
                f"({self.end_frame})")
        return self


class FramingSceneTimeline(BaseModel):
    """Normalized Task 12 scene on the clip-relative clock (inclusive frames)."""

    model_config = ConfigDict(allow_inf_nan=False)

    scene_id: int = Field(ge=0)
    strategy: str = Field(min_length=1)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    applies: bool
    # Task 14: explicit versioned scene contract ("face" | "action" |
    # "none"). Additive optional — old projects without it validate
    # unchanged.
    scene_kind: Optional[str] = None
    subject_ids: list[str] = Field(default_factory=list)
    keyframes: list[FramingKeyframe] = Field(default_factory=list)
    fallback_reason: Optional[str] = None
    fallback_detail: Optional[str] = None
    # Task 14 rev3: explicit conservative full-frame fit (action scenes).
    # When true, ranges without keyframes render as full-frame letterbox
    # (not as crops); keyframes still give crop targets where the plan
    # tightens. Additive optional -- old projects validate unchanged.
    fit: bool = False
    #: Lossless ordered clip-relative fit intervals (Task 14 rev4).
    #: Additive: old projects without this field validate unchanged.
    fit_ranges: list[FramingFitRange] = Field(default_factory=list)

    @field_validator("scene_id", "start_frame", "end_frame", mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)

    @field_validator("fit", mode="before")
    @classmethod
    def _fit_is_bool(cls, v):
        if not isinstance(v, bool):
            raise ValueError(f"fit must be a bool, got {v!r}")
        return v

    @field_validator("scene_kind")
    @classmethod
    def _kind_allowed(cls, v):
        if v is not None and v not in ("face", "action", "none"):
            raise ValueError(
                f"scene_kind must be face/action/none, got {v!r}")
        return v

    @field_validator("subject_ids")
    @classmethod
    def _anonymous_ids(cls, v):
        for sid in v:
            if not isinstance(sid, str) or not _TRACK_ID_RE.match(sid):
                raise ValueError(
                    "subject_ids must be anonymous local ids "
                    f"(t-<int>), got {sid!r}")
        return v

    @model_validator(mode="after")
    def _check_order(self):
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be <= end_frame "
                f"({self.end_frame})")
        frames = [k.frame for k in self.keyframes]
        if any(b <= a for a, b in zip(frames, frames[1:])):
            raise ValueError("keyframes must be ordered with unique frames")
        # Task 14 rev4: fit ranges are ordered, non-overlapping, inside
        # the scene, and never cover a keyframe (a keyframe is a crop
        # target; fit renders full-frame letterbox, the renderer gives
        # fit precedence).
        prev_end = self.start_frame - 1
        for r in self.fit_ranges:
            if not (self.start_frame <= r.start_frame <= r.end_frame <= self.end_frame):
                raise ValueError(
                    f"fit range [{r.start_frame}, {r.end_frame}] outside "
                    f"scene [{self.start_frame}, {self.end_frame}]")
            if r.start_frame <= prev_end:
                raise ValueError("fit_ranges must be ordered and non-overlapping")
            prev_end = r.end_frame
            for k in self.keyframes:
                if r.start_frame <= k.frame <= r.end_frame:
                    raise ValueError(
                        f"keyframe at frame {k.frame} inside fit range "
                        f"[{r.start_frame}, {r.end_frame}]")
        if self.fit_ranges and not self.fit:
            raise ValueError("fit_ranges requires fit=true")
        return self


class FramingPlanIdentity(BaseModel):
    """Immutable identity of the promoted stable sidecar + Task 12 plan."""

    model_config = ConfigDict(allow_inf_nan=False)

    plan_ref: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    schema_version: int = Field(ge=1)
    producer: str
    source_fingerprint: str
    config_digest: str
    tracker_digest: str = ""
    output_aspect: float = Field(gt=0)
    analysis_complete: bool
    timeline_digest: str

    @field_validator("size", "mtime_ns", "schema_version", "output_aspect",
                     mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)


class FramingSource(BaseModel):
    """Source slice the plan was authored against (plan header)."""

    model_config = ConfigDict(allow_inf_nan=False)

    fingerprint: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    slice_offset_frames: int = Field(ge=0)

    @field_validator("width", "height", "fps", "frame_count",
                     "slice_offset_frames", mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)


class FramingRender(BaseModel):
    """Render geometry the normalized trajectory is valid for."""

    model_config = ConfigDict(allow_inf_nan=False)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    aspect: float = Field(gt=0)

    @field_validator("width", "height", "fps", "aspect", mode="before")
    @classmethod
    def _no_bools(cls, v):
        return _reject_bool_numeric(v)


class ProjectFraming(BaseModel):
    """Task 13 bridge: the accepted automatic camera plan as project data.

    ``eligible=True`` means the normalized trajectory is complete and may
    be replayed/edited by a future trajectory tool. ``eligible=False``
    with ``ineligibility_reason`` is an audit reference only (e.g. an
    incomplete Task 12 plan): it must never be treated as a camera path.
    """

    model_config = ConfigDict(allow_inf_nan=False)

    version: Literal[1] = 1
    eligible: bool = True
    ineligibility_reason: Optional[str] = None
    plan: FramingPlanIdentity
    source: FramingSource
    render: FramingRender
    scenes: list[FramingSceneTimeline] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_eligibility(self):
        if self.eligible:
            if not self.scenes:
                raise ValueError(
                    "eligible framing requires at least one scene")
            for scene in self.scenes:
                if (scene.applies and not scene.keyframes
                        and not scene.fallback_reason and not scene.fit):
                    raise ValueError(
                        f"applicable scene {scene.scene_id} needs keyframes, "
                        "a fallback_reason, or fit")
        elif not self.ineligibility_reason:
            raise ValueError(
                "ineligible framing requires ineligibility_reason")
        return self


class HyperframesFinishing(BaseModel):
    """Task 18: optional Hyperframes finishing selection (additive, default OFF).

    ``renderer="hyperframes"`` is the explicit opt-in; the default
    ``"legacy"`` keeps the accepted renderer. ``template`` and the caption
    enums are closed sets validated by the finishing compiler. Old projects
    without this field validate, save/load, undo, preview and render through
    existing paths.
    """

    renderer: Literal["legacy", "hyperframes"] = "legacy"
    template: Optional[Literal["clean-captions", "emphasized-word",
                              "timed-opening"]] = None
    caption_size: Literal["small", "medium", "large"] = "medium"
    caption_position: Literal["lower-third", "middle"] = "lower-third"


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
    # Task 05: automatic framing preservation. When set on an auto project
    # whose source/timeline/render geometry is unchanged, the renderer
    # reuses the reframed base instead of materializing a static crop.
    # Additive optional field; old projects without it use existing routes.
    reframed_base: Optional[ReframedBase] = Field(default=None)
    # Task 13: stable framing artifact bridge. Optional; when present,
    # records the validated Task 12 turn-aware framing plan as a
    # normalized clip-relative trajectory plus immutable sidecar identity.
    # Additive optional Task 13 field; old projects without it validate,
    # save/load, undo, preview and render through existing paths.
    framing: Optional[ProjectFraming] = Field(default=None)

    # Task 15: grounded opening identity. Optional; when present,
    # records the validated selected option (or user-authored override,
    # or honest no-selection) plus packet identity and versions.
    # Additive optional Task 15 field; old projects without it validate,
    # save/load, undo, preview and render through existing paths.
    opening: Optional[OpeningSelection] = Field(default=None)

    # Task 18: optional Hyperframes finishing selection. Additive optional
    # field; default renderer is "legacy" (OFF). Old projects without it
    # validate, save/load, undo, preview and render through existing paths.
    finishing: Optional[HyperframesFinishing] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _migrate_word_ids(cls, data):
        # Stable word-ID migration: backfill + counter advance + dup check.
        return _backfill_word_ids(data)


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
                    # Stable word IDs from the persisted counter: w0, w1, ...
                    words.append(CaptionWord(
                        id=f"w{len(words)}", w=text, start=ws, end=we))

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
            next_word_id=len(words),
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
