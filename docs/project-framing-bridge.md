# Project framing bridge (Task 13)

How the accepted Task 12 turn-aware camera plan becomes a stable,
auditable part of the clip project — and what it does *not* do.

## Artifact lifecycle

```
render attempt (temp video)          pipeline (reframe.py / main.py /
  └─ temp sidecar                     orchestrator.py)
     <temp video stem>                  │
     .turn_framing_plan.json           │ render validation succeeds
                                       ▼
promote → STABLE sidecar             pipeline only
  <clip_final stem>.turn_framing_plan.json
  (job-relative, deterministic)

emit → project.framing                domain (project_framing.py)
  validated + normalized copy of the plan

render → verify_framing_at_render      domain, called in the
  (eligible-base branch)               eligible-base branch
```

1. **Temporary sidecar.** Each render attempt writes its Task 12 plan next
   to the temporary video. A failed render/QA/retry removes its temporary
   sidecar and never promotes; a retry promotes only its own plan.
2. **Promotion (coordinated pair).** Only after render validation succeeds,
   the pipeline publishes the video first and then coordinates the sidecar
   half via `publish_attempt_sidecar` (`project_framing.py`) — promotion,
   retire, or preserved-project skip — in the queued orchestrator, the
   direct CLI clip loop, and the reframe-only path alike. Promotion
   rebinds the plan's Task 12 `source_fingerprint` to the actual render
   input (`rebind_sidecar_source`; a byte-identity no-op in production,
   where reframe already stamps the true identity). A promotion failure
   is **never swallowed**: the just-published video is rolled back and
   the attempt fails, so an accepted clip can never pair a new video
   with a stale plan. A project must never reference `*.render.tmp*` /
   `*.reframe.tmp*` basenames — `resolve_sidecar` rejects them with
   `framing-sidecar-temp-name`.
3. **Retire rule.** A sidecar-less attempt (render produced no plan)
   retires the previous stable binding instead of inheriting it: the
   stale sidecar is deleted and the emitted project carries no `framing`
   field. A stable sidecar pinned by a preserved user project is left
   in place.
4. **Preserved projects pin their plan.** A stable sidecar referenced by
   a preserved user project (Task 01: `origin == "user"` or `version >
   1`, matching `plan_ref` basename + `sha256`) is immutable: a later
   attempt that produces a *different* plan does not overwrite it
   (promotion is skipped, the attempt temp file removed), so
   `verify_framing_at_render` keeps passing on the preserved project.
3. **Emission.** `build_project_framing` resolves the stable sidecar
   inside the job dir (real containment; traversal/symlink escape →
   `framing-sidecar-escape`), parses it through the Task 12 production
   parser, normalizes it, and stores the result as the additive optional
   `framing` project field. Any sidecar problem (missing, malformed,
   temp-named, escape, aspect mismatch, **cross-source**) yields **no field
   at all** — the pre-Task-13 behavior — never a lying field, and never an
   overwrite of an existing good project. When the caller supplies the
   actual current source slice's Task 12 identity
   (`expected_source_fingerprint`), a structurally valid plan from a
   different source is rejected (`framing-cross-source`).
4. **Render.** In the eligible-base branch the pipeline calls
   `verify_framing_at_render`, which re-resolves the recorded
   `plan_ref` and compares **sha256 AND source_fingerprint AND size**.
   Same-size/same-mtime content replacement is caught by the content
   hash (`framing-sidecar-mismatch`); a deleted sidecar fails with
   `framing-sidecar-missing`. Before the Task 05 base is reused, the
   verified plan's source identity is rechecked against the **actual
   current source slice** — a cross-source sidecar swap the stored
   record predates is rejected with a clear `ProjectRenderError`.
   A project that carries `framing` but **no** `reframed_base` lineage
   fails clearly before any static materialization
   (`framing-trajectory-edit`): dynamic replay is not implemented, so
   the renderer never silently publishes a center crop disconnected from
   the recorded camera plan. Genuinely old projects (no `framing` field)
   and explicit static edits on a project *with* a lineage base keep
   their accepted paths.

## Identity fields

`framing.plan` (FramingPlanIdentity) is the immutable identity:

| field | meaning |
|---|---|
| `plan_ref` | job-relative ref, `<stem>.turn_framing_plan.json` |
| `sha256` / `size` / `mtime_ns` | sidecar file identity (content hash catches same-size replacement) |
| `schema_version` / `producer` | `3` / `clippyme-turn-framing-plan` |
| `source_fingerprint` | Task 12 sampled source identity (size+mtime+head/tail bytes) |
| `config_digest` / `tracker_digest` | plan build inputs; a config or tracker change invalidates reuse |
| `output_aspect` | aspect the crops were built for |
| `analysis_complete` | `false` → audit record only (see below) |
| `timeline_digest` | sha256 of the canonical normalized scene timeline |

`framing.source` echoes the plan header (`fingerprint`, `width`,
`height`, `fps`, `frame_count`) plus `slice_offset_frames`, the single
rebase applied. `framing.render` records the render geometry the
trajectory is valid for (`width`, `height`, `fps`, `aspect`).

The **lineage baseline** (`framing_baseline_identity`) is the comparable
subset: `plan_ref`, `sha256`, `source_fingerprint`, `config_digest`,
`tracker_digest`, `output_aspect`, `analysis_complete`,
`timeline_digest`. `size`/`mtime_ns` are deliberately excluded —
re-promotion of byte-identical content must not read as a trajectory
change.

## Coordinate basis

- Task 12 plans store **absolute frame indices of the render input**,
  which in production is the already-cut source slice (frame 0 = slice
  start = selected clip start).
- `normalize_timeline` performs **exactly one rebase** with integer
  arithmetic: `clip_frame = plan_frame - slice_offset_frames`. There is
  no float frame math anywhere, so 23.976/24/29.97/30/60 fps plans cannot
  drift a boundary across a frame; `fps` is validated (finite, > 0,
  non-bool) and preserved as declared, never used to rescale frames.
- Scene `start_frame`/`end_frame` and keyframe `frame` stay **inclusive
  ints** — half-open `[start, end + 1)` in frame space. Only the origin
  moves.
- Every scene must satisfy `0 <= clip_start <= clip_end <
  clip_frame_count`; keyframes must be ordered, unique, inside their
  scene, with finite int coords `0 <= x1 < x2 <= frame_w` (same for y vs
  `frame_h`). Strategy strings are preserved verbatim.
- **Non-face scenes** (anything not `TRACK`/`WIDE`: `SPLIT`,
  `SCREENCAST`, `GENERAL`, `DISABLED`, …) keep `keyframes: []` and
  `applies: false`. Keyframes are never manufactured for them; a
  non-face scene carrying face content is rejected as malformed.
- Applicable face scenes carry anonymous `subject_ids` (`t-<int>`,
  first-appearance order from the plan's segments) and their keyframes;
  scenes with an accepted fallback carry `fallback_reason`/`fallback_detail`
  instead. No raw observations, pixels, mouth landmarks, embeddings,
  provider payloads, names, diarization labels, or temporary absolute
  paths are copied — the strict Task 12 parser already rejects unknown
  keys at every level.

## Eligible vs ineligible (audit) outcomes

- `eligible: true` — the normalized trajectory is complete. Every
  `applies: true` scene has ≥1 keyframe or a `fallback_reason`. This is
  the replayable/editable contract a future trajectory tool can build on.
- `eligible: false` + `ineligibility_reason: "analysis-incomplete"` —
  the Task 12 plan did not finish analysis (`analysis_complete: false`).
  The record is kept as an **audit reference only** (`scenes: []`): the
  exhausted fallback is never converted into a normal camera path, and
  the record must never be treated as a trajectory.

## Task 05 interaction

Task 05's eligible `reframed_base` remains the pixel authority for an
unchanged automatic project: caption/overlay/grade/music edits keep
exact framing because composition layers onto the reused base.

The new `framing` field joins the lineage geometry baseline:

- **unchanged** (baseline identity equal) → base reuse, as before;
- **changed** (any baseline field differs, field dropped, or identity
  malformed-but-present) → `framing_edit`: fail clearly with the stable
  reason `framing-trajectory-edit`. The old base is NOT reused and the
  renderer does NOT silently fall back to a static center crop — dynamic
  source replay of an edited trajectory is not implemented in this task;
- **malformed** project framing identity → `invalid_asset`;
- lineage without a framing baseline (pre-Task-13 emissions) keeps the
  old outcomes; `framing` is simply ignored.

Existing explicit static crop/timeline edits keep their accepted
source-render fallback.

## Old-project compatibility

`framing` is additive and optional. Projects without it validate,
save/load, undo, preview, and render through their existing paths
unchanged. The JSON Schema twin is regenerated from the production
Pydantic model (`project_json_schema()`), so the on-disk schema always
matches the model.

## Honest limits

This task stabilizes and exposes the accepted automatic camera plan to
the project contract. It does **not**:

- replay or edit keyframes dynamically (a `framing_edit` fails clear
  instead of pretending);
- add a browser/editor trajectory UI;
- validate active-speaker accuracy (the plan's evidence rules are Task
  12's; this bridge trusts the production parser, not the speaker);
- implement non-face action tracking;
- change hooks, captions, discovery, or Smart Cut.

Unchanged export still relies on the validated Task 05 reframed base.
