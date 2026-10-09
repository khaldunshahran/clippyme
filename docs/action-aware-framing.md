# Action-aware framing plans (Task 14)

Evidence-backed, action-aware camera framing for automatic GENERAL
(non-face) scenes in the quality two-pass route. The plan is built once
from validated visual observations in pass 1, persisted atomically as a
versioned v4 framing plan, and rendered verbatim in pass 2 — rendering
never recomputes decisions, never redetects.

This does NOT redesign candidate selection, identify people, add
provider calls or model dependencies, build editor controls, change
hook/caption styling, or claim real-footage/OpusClip parity. Injected
detections and synthetic landmarks validate the machinery, not
real-world sports/gameplay/action accuracy.

## Flag

`CLIPPYME_ACTION_AWARE_FRAMING` — enabled by default on this branch for
automatic GENERAL scenes. The explicit negatives `0`/`false`/`no`/`off`
restore the accepted Task 13 GENERAL behavior byte-for-byte (no action
plan is persisted; v3 sidecars are written). Unset/empty/anything else
means on.

TRACK, WIDE, SPLIT, SCREENCAST, DISABLED, explicit subject/object modes
and manually selected layouts keep their accepted routing and incur NO
action-analysis cost. Positive SPLIT/SCREENCAST decisions are
authoritative and never enter the action planner.

## Observations (`action_framing.ActionObservation`)

Strict pure record, validated on add (rejects — never coerces):

- absolute `frame` + `time_s` (monotonic per track; no regressing times)
- `scene_id`, anonymous `track_id` matching `^t-\d+$` (opaque, monotonic
  for the tracker's lifetime; never reused, so the cross-scene guard can
  never collide with a new subject)
- finite in-bounds `box` (x, y, w, h within the frame, w/h > 0)
- bounded detector `confidence` in [0, 1]
- normalized `category` family: `person` | `animal` | `vehicle` | `object`
  (anything else is rejected — no free-form label enters a plan)
- `motion` (per-track box motion 1−IoU), `saliency` (local support),
  `visibility` — each in [0, 1]
- `provenance` (`detector` | `tracker` | `injected-test` |
  `smoother-hold`), short machine-readable `reason`
- no booleans anywhere, no duplicate `(frame, track_id)`, no cross-scene
  identity carryover (the tracker resets at hard cuts; the log rejects a
  track id seen under a different scene id)
- stores no pixels, embeddings, names, OCR text, or provider payloads

## Detection seam (`reframe_detect.detect_action_candidates`)

Reuses the existing optional local YOLOv8 seam (no new dependency, no
live provider). Returns `[{"box", "confidence", "category"}]` with COCO
names mapped onto the normalized families. If detection is
unavailable/fails/exceeds budget, the scene keeps the accepted GENERAL
fallback with an explicit reason (`detector-unavailable` /
`detector-failed` / `analysis-budget-exhausted`).

Saliency or raw motion alone may bias a wide crop slightly
(`saliency_biased_wide_crop`, ≤5% of frame width) but can NEVER authorize
a tight subject crop.

## Segmentation (`action_framing.segment_actions`)

Ordered, non-overlapping, full-coverage scene-local segments from
**sustained** multi-frame evidence (`confidence × visibility` over
`evidence_duration_s`, default 0.8 s). Saliency and motion are support
signals only — they never count as evidence.

- One strong subject → `single` lock; two strong, close, separable
  subjects whose padded union fits → `group`; everything else → `wide`.
- **Size never selects**: a large weak box cannot steal the crop from
  sustained evidence (greedy confidence-ordered track assignment).
- Fast/uncertain motion (`fast_motion_widen`, `fast_travel_frac`,
  `uncertain_motion_range`), boundary pressure, large camera motion
  (`camera_motion_widen`), or evidence loss → widen. A fast crossing is
  held wide, never chased.
- Minimum dwell (`min_dwell_s` 1.0 s), switch margin (`switch_margin`
  1.25), cooldown (`cooldown_s` 1.5 s); short occlusions
  (≤ `occlusion_hold_s` 1.0 s) hold the framing; tiny oscillations merge
  (`merge_window_s` 0.5 s); a brief measurement blip adjacent to wide is
  absorbed by the wide.
- Hard cuts reset tracks and decisions; the log rejects cross-cut
  identity carryover.

## Keyframes (`action_framing.build_action_keyframes`)

One stable subject (or group) keeps ONE lock crop for the whole
segment, built from the subject's **temporal union** (its whole travel
within the segment — never cropped away) with configurable padding
(`padding_frac` 0.30), look-ahead lead room along the motion vector
(`look_ahead_frac` 0.15, clamped to preserve containment), and headroom
(`headroom_frac` 0.10, same convention as the Task 12 face path).

- Zoom targets `target_subject_occupancy` (0.5) of the crop height,
  clamped to [`min_zoom`, `max_zoom`] = [1.0, 2.0]; the padded union is
  fitted down to `min_zoom`, otherwise the segment widens.
- Boundary pressure (clamping eats > `boundary_pressure_frac` of the
  padding) widens instead of pinning the subject to a crop edge.
- Between accepted decisions, bounded `ease` keyframes interpolate
  linearly; a `lock` keyframe pins each segment's stable crop so the
  trajectory survives the project bridge.
- Crop-centre travel is capped at `max_travel_frac` (0.5) of the frame
  dimension **per second** at 24/29.97/30/60 FPS — the transition is
  extended over as many frames as the cap needs, so every consecutive
  rendered step including the final ease-to-lock step respects it. When
  the segment cannot fit the transition, the previous lock is held.
- Every crop preserves the output aspect, stays in bounds, and contains
  all supported participants.

## Schema v4 (`framing_plan`)

`SCHEMA_VERSION = 4` (`SUPPORTED_SCHEMA_VERSIONS = (3, 4)`). v4 adds an
explicit versioned `scene_kind` per scene: `face` | `action` | `none`.
The parser distinguishes the three contracts and validates relational
invariants (action scenes need GENERAL + applies; face scenes need a
face strategy + applies; `none` scenes must not claim applies or carry
content). v3 plans parse exactly as before (kind derived; Task 12/13
sidecars and projects remain valid).

With the action flag off, the builder still writes v3 (byte-identical to
Task 13). With it on, GENERAL scenes become `action` scenes and the plan
binds the combined turn+action config digest (`plan_config_digest`), so
any config or routing change invalidates reuse (`stale-config`).

`plan_matches` accepts both supported schemas; the config digest (not
the schema number) prevents cross-flag reuse.

## Budget (`AnalysisBudget`)

The single monotonic framing budget covers decode, detector work,
motion/saliency analysis, and planning — charged before work starts. On
exhaustion: `analysis_complete=false`, the first dropped frame/scene is
recorded, the partial plan is not reusable, and the scene falls back
safely. Detector/video/temp resources are released exactly once.

## Project bridge (`domain/project_framing.py`, `domain/clip_project.py`)

- `normalize_timeline` dispatches on the explicit scene kind; action
  scenes keep their keyframes (lock + ease) and subject ids.
- `FramingSceneTimeline` gains optional `scene_kind`
  (`face`|`action`|`none`); old projects without it validate unchanged.
  The JSON Schema twin is regenerated from the model.
- `build_project_framing` carries v4 action plans (eligible only when
  complete); incomplete plans are ineligible with an explicit reason;
  malformed/cross-source/aspect-mismatched plans return None (never a
  lying field).
- Unchanged projects reuse the validated pixel base via the Task 13
  lineage (`timeline_digest` over normalized scenes, stored-vs-stored
  comparison). A changed trajectory is `framing_edit` and fails
  clearly — dynamic replay is not implemented.

## Config

`CLIPPYME_ACTION_*` overrides (garbage → default, out-of-range →
clamped): `EVIDENCE_S`, `MIN_DWELL_S`, `COOLDOWN_S`, `SWITCH_MARGIN`,
`MERGE_S`, `OCCLUSION_S`, `EASE_FRAMES`, `PADDING`, `LOOK_AHEAD`,
`EVIDENCE_THRESHOLD`, `MAX_TRAVEL`, `SAMPLE_EVERY`.

## Limits (honest)

Injected detections and synthetic landmarks do not validate real
sports/gameplay/action accuracy, editorial taste, hook/caption quality,
editable trajectory UI, or OpusClip parity. Those claims require the
frozen real-footage benchmark.
