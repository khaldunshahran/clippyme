# Auto Layout Policy (Task 02)

Safe scene layout selection for automatic SCREENCAST promotion.

## Problem

The geometry helper returned a face center or frame-center fallback for
every scene, and `reframe.py` promoted GENERAL scenes to SCREENCAST for
every entry. Ordinary or faceless footage could be rendered as a duplicated
screencast with no evidence of a screen.

## Policy

Semantic ELIGIBILITY is separated from presenter GEOMETRY. Automatic
promotion requires both:

1. **Scene-local semantic classification**: frames sampled within
   `[scene_start, scene_end)` are classified. Requires:
   - layout `"screencast"` (or `"inset"` for webcam-over-content)
   - confidence >= 0.85
   - `screen_evidence`: explicit boolean `true` (False/missing/invalid
     never becomes positive from the label alone)
   - nonempty `why` explanation
2. **Reliable presenter geometry**: `detect_presenter_geometry` returns a
   center only for scenes with valid detections:
   - box complete, finite, `[x, y, w, h]` with `w > 0`, `h > 0`
   - box wholly in bounds: `0 <= x`, `0 <= y`, `x + w <= fw`, `y + h <= fh`
   - frame dimensions finite and positive
   - coverage = valid detections / attempted samples >= 60%
   - at least 3 detections, spread <= 10% (static pane assumption)
   - Failed reads never inflate coverage; failures are per-scene local.

Outcomes:
- Screen + reliable presenter -> SCREENCAST (existing composition).
- Screen without presenter -> keep previous (no fabricated duplicated pane).
- Unknown, failed, low-confidence, provider-error, missing-key,
  unreadable-frame, or exhausted-budget -> keep previous strategy.

Eligible: GENERAL and TRACK (a visible webcam often yields TRACK).
SPLIT (two-person) and explicit strategies are never auto-promoted.
Negative/unknown TRACK scenes retain TRACK.

## Configuration

`AUTO_LAYOUT=1` (or `true`/`yes`) enables semantic classification.
Unset, empty, or false disables it: no provider calls, no promotion.
Checked at invocation time via `is_auto_layout_enabled()`.

The compatibility wrapper `pick_and_apply_layout` respects the
invocation-time flag: explicit `"auto"` with the flag disabled returns
`"auto"` without classifying. Explicit layout modes pass through unchanged.

Changing deployment flags waits for deployment authorization.

## Budgets

Per-scene:
- Max 8 samples, min 3 usable samples for a decision.
- Sample count clamped to `[1, MAX_SCENE_SAMPLES]`.

Invocation-level (`maybe_promote_screencast`):
- Max 32 scenes evaluated, max 16 provider requests, 120s deadline.
- Exhaustion stops evaluation; evaluated scenes keep results, rest keep
  previous strategy.
- Geometry (local CV) runs first; scenes without geometry skip paid
  classification entirely.

Provider (`_gemini_classify`):
- Explicit SDK timeout (30s per attempt, no unbounded defaults).
- Bounded retries (2 retries with backoff, no leaked workers).
- Missing key or disabled flag: no call, immediate None.

## Tests

- `test_layout_policy.py` (50): parser contract, provider injection,
  config flag, policy decisions (GENERAL/TRACK/SPLIT), budgets,
  compatibility wrapper, geometry validation.
- `test_reframe_layout_promotion.py` (5): synthetic 3-scene unit checks
  with colored landmarks; helper-level composition verification.
- `test_production_render.py` (3): actual `process_video_to_vertical`
  entrypoint (comfort + streaming) with 3-scene AV fixture; verifies
  dimensions, aspect, duration, retained audio, screencast composition
  in output frames; explicit modes bypass classifier.

Independent probes (13): all pass. Run from
`work/task02-review1/test_review_contract.py` (not committed).

## Visual-validation limits

Real-footage review across 10 genres NOT performed. OpusClip benchmark
unverified. Synthetic integration proves policy correctness; the benchmark
dependency is explicitly retained for follow-up.
