# Turn-aware framing plans (Task 12)

Evidence-backed, turn-aware camera framing for automatic TRACK/WIDE face
scenes in the quality two-pass route. The plan is built once from validated
face/mouth observations in pass 1, persisted atomically, and rendered
verbatim in pass 2 — rendering never recomputes decisions.

## Flag

`CLIPPYME_TURN_AWARE_FRAMING` — enabled by default on this branch for
automatic TRACK/WIDE scenes. The explicit negatives `0`/`false`/`no`/`off`
restore the accepted static comfort behavior (whole-scene median collapse
via `collapse_scene_targets`). Unset/empty/anything else means on.

The plan path additionally gates on static comfort: with
`REFRAME_STATIC_AUTO=0` (legacy moving-camera trajectory) the plan is not
built. Explicit `disabled`/`subject`/`split`/`screencast` modes never reach
the two-pass route, so they incur no framing-plan analysis cost. GENERAL,
SPLIT, SCREENCAST and DISABLED scene strategies keep their existing
renderers.

## Observations (`framing_plan.FaceObservation`)

Strict pure record, validated on construction (rejects — never coerces):

- absolute `frame` + `time_s` (monotonic per log; no regressing times)
- `scene_id`, anonymous `track_id` matching `^t-\d+$` (names, diarization
  labels or any other identity can never validate)
- finite in-bounds `box` (x, y, w, h within the frame, w/h > 0)
- `visibility` in [0, 1], `mouth_motion` finite in [0, 3]
- `provenance`, `reason` (short strings)
- no booleans anywhere, no duplicate `(frame, track_id)`, no cross-scene
  identity carryover (the tracker resets at hard cuts; the log rejects a
  track id seen under a different scene id)

## Decisions (`reframe_track.SpeakerTracker.decide`)

Additive alongside the preserved `get_target(...)` API (byte-identical
behavior, verified by parity tests). Returns `SubjectDecision` with chosen
subject id(s), box, per-track evidence components (size / mouth /
sustained), confidence, ambiguity and a machine-readable reason.

Turn rules:

- A switch requires the challenger's **sustained** mouth-motion evidence to
  beat the incumbent by `switch_margin` (default 1.25) after `cooldown_s`
  (1.5 s) and `min_dwell_s` (1.0 s). **Size alone can never prove the
  active speaker** — equal-evidence faces stay wide, and a weak single
  claim degrades to `weak-evidence-hold`/`weak-evidence-group`.
- Ambiguous dialogue (two strong, close evidence tracks) uses a two-subject
  crop when it fits 9:16, else holds wide — never crowns one speaker on
  ambiguity.

## Segments (`framing_plan.segment_turns`)

Ordered, non-overlapping, full-coverage scene-local segments:

- evidence window `evidence_duration_s` (0.8 s), tiny-oscillation merging
  (`merge_window_s` 0.5 s), hard-cut reset (new log per scene in practice)
- short occlusion (≤ `occlusion_hold_s` 1.0 s) holds the framing and counts
  `occluded_frames`; prolonged loss widens (`speaker-lost-widen`)
- one stable subject stays in one static segment for the whole scene

## Crops

Fixed crop within each stable segment (`crop_for_subjects`): headroom
(0.15), padding (0.35), clamped to the frame, zoom in [1.0, 1.6].
Between accepted turns only: a short bounded linear ease (`ease_frames`
6, travel capped at `max_travel_frac` 0.5 of frame width). Zoom holds
unless the framing widens; wide→single resets the zoom (deliberate
reframe). No continuous chase, breathing or jitter.

## Plan identity, persistence, reuse

`FramingPlan` carries: schema version, producer, source fingerprint
(SHA-256 of the file bytes), frame dims, FPS, frame count, scene intervals
(inclusive ends), config digest, per-scene strategies/tracks/segments/
keyframes, fallback crops and diagnostics. No pixels, embeddings, names,
provider responses or mouth landmarks.

- `save_plan_atomic` writes temp + `os.replace` (no `*.tmp-*` leftovers).
- Pass 2 re-loads the plan from disk and renders it verbatim.
- Reuse requires fingerprint + dims + FPS + frame count + scene plan +
  schema version + config digest to match, else rebuild with a logged
  reason (`stale-*`); missing/malformed plans rebuild (`malformed`).
- Analysis is bounded (`max_analysis_frames` 20000, observation cap);
  per-scene failures are isolated — invalid TRACK/WIDE scenes keep the
  accepted static/wide fallback crop with a reason.

## Limits (stated honestly)

Injected face/mouth observations and synthetic fixtures validate the
machinery (validation, segmentation, persistence, rendering, rollback),
not real active-speaker accuracy, identities, non-face action framing,
editable ClipProject trajectories, or OpusClip parity.
