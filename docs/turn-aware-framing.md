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
6) with travel limited to `max_travel_frac` (0.5) of the frame dimension
PER SECOND — the transition extends over as many frames as the per-frame
speed cap (`max_travel_frac * dim / fps`) needs, so every consecutive
rendered crop-centre step, including the final ease-to-lock step, respects
the cap. When the segment cannot fit the transition, the previous lock is
held (zero travel — the safe fallback); excess travel is never hidden in
the final step. Zoom holds unless the framing widens; wide→single resets
the zoom (deliberate reframe). No continuous chase, breathing or jitter.

## Plan identity, persistence, reuse

`FramingPlan` carries (schema version 2): schema version, producer, source
fingerprint (SHA-256 of the file bytes), frame dims, FPS, frame count,
scene intervals (inclusive ends), config digest, the ACTUAL render aspect
(`output_aspect` — the crops are built for it, never a silent 9:16 default),
the ordered per-scene strategies, the tracker decision digest
(`tracker_digest` — the `SpeakerTracker` decision config that produced the
observations), per-scene tracks/segments/keyframes, fallback crops and
diagnostics. No pixels, embeddings, names, provider responses or mouth
landmarks.

- `save_plan_atomic` writes temp + `os.replace`; the temp file is removed
  in `finally` after ANY write/flush/replace failure while an existing
  destination is preserved (no `*.tmp-*` leftovers).
- Pass 2 re-loads the plan from disk and renders it verbatim.
- Reuse requires fingerprint + dims + output aspect + FPS + frame count +
  scene plan + ordered strategies + schema version + config digest +
  tracker digest to match, else rebuild with a logged reason (`stale-*`);
  missing/malformed plans rebuild (`malformed`). A 9:16 sidecar is never
  reused for 1:1/16:9 output (`stale-aspect`), nor after a strategy or
  tracker-config change (`stale-strategies`, `stale-tracker-config`).
- `plan_from_dict` validates every persisted field recursively with strict
  non-boolean numerics: finite positive dims/FPS/frame count, ordered unique
  scene ids matching the header intervals, anonymous track ids with bounded
  frame/count fields, ordered non-overlapping full-coverage segments, unique
  ordered keyframes, allowed evidence keys with bounded scalar values only,
  bounded reason/detail tokens, finite in-bounds fallback crops. Unknown
  keys (pixels, landmarks, embeddings, names) are rejected at every level.
- Analysis is bounded by ONE monotonic budget (`AnalysisBudget`):
  `max_analysis_frames` (20000, observation cap) AND `analysis_deadline_s`
  (default 120 s, `CLIPPYME_TURN_ANALYSIS_DEADLINE_S`). The budget is checked
  before every expensive step (frame decode/retrieve, face detection,
  mouth analysis, planner build); once exhausted, detector/MAR call counts
  STOP increasing, the exhaustion is recorded (`analysis_budget_exhausted` /
  `analysis-frame-budget-exhausted`), and unobserved scenes keep the accepted
  static/wide fallback. The render continues safely and every resource is
  released exactly once.
- Per-scene failures are isolated — invalid TRACK/WIDE scenes keep the
  accepted static/wide fallback crop with a reason.

## Budget exhaustion and incomplete plans (review 02)

- Pass 1 records structured exhaustion on the observation log via
  `mark_exhaustion(reason, frame, scene_id)`: the exhaustion frame is the
  first frame NOT analyzed. `build_framing_plan` carries this state into
  planning: scenes ending strictly before the exhaustion frame were fully
  analyzed and may keep their plan; the affected partial scene and every
  later unobserved face scene skip turn segmentation/keyframe construction
  and get an explicit static/wide fallback with `analysis_budget_exhausted`
  or `analysis-frame-budget-exhausted`. Without frame context (a bare
  diagnostic note) every face scene falls back conservatively.
- The persisted plan records `analysis_complete` (schema v3). An
  exhausted/incomplete plan may be saved for audit and safe rendering,
  but `plan_matches()` returns `incomplete-analysis` for it, so the next
  invocation re-analyzes instead of reusing the partial result.
- `analysis_deadline_s` is part of `TurnFramingConfig.to_dict()`, so the
  deadline participates in `config_digest` and plan identity.

## Relational sidecar validation (review 02)

- Beyond scalar shapes, `plan_from_dict` enforces relationships: scene
  intervals strictly ordered and non-overlapping; unique track IDs whose
  frame ranges lie inside their scene; every segment subject ID exists in
  the scene's track table and its track range intersects the segment;
  two-subject IDs distinct; consecutive segments tile the scene
  contiguously with no gaps or overlaps; fallback scenes carry no active
  segments/keyframes; non-applicable scenes carry no planned face content.

## Limits (stated honestly)

Injected face/mouth observations and synthetic fixtures validate the
machinery (validation, segmentation, persistence, rendering, rollback),
not real active-speaker accuracy, identities, non-face action framing,
editable ClipProject trajectories, or OpusClip parity.
