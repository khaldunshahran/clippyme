# Auto Layout Policy (Task 02)

Safe scene layout selection for automatic SCREENCAST promotion.

## Problem

The geometry helper (`detect_screencast_scenes`) returned a face center or
frame-center fallback for every scene, and `reframe.py` promoted GENERAL
scenes to SCREENCAST for every entry. Ordinary or faceless footage could be
rendered as a duplicated screencast (content on top, speaker below) with no
evidence of a screen.

## Policy

Semantic ELIGIBILITY is separated from presenter GEOMETRY. Automatic
promotion requires both:

1. **Scene-local semantic classification**: frames sampled within
   `[scene_start, scene_end)` are classified. Requires layout
   `"screencast"`, confidence >= 0.85, and positive screen-content evidence.
2. **Reliable presenter geometry**: `detect_presenter_geometry` returns a
   center only for scenes with stable real detections (>= 60% of samples,
   >= 3 detections, spread <= 10% of frame dimensions). No center fallback.

Outcomes:
- Screen + reliable presenter -> SCREENCAST (existing composition).
- Screen without presenter -> keep GENERAL (no fabricated duplicated pane).
- Unknown, failed, low-confidence, provider-error, missing-key,
  unreadable-frame, or exhausted-budget -> keep previous strategy.

TRACK and SPLIT scenes are never auto-promoted. Explicit modes
(`disabled`/`subject`/`object`/`screencast`/`split`) bypass the classifier
and keep their existing behavior.

## Configuration

`AUTO_LAYOUT=1` (or `true`/`yes`) enables semantic classification.
Unset, empty, or false disables it: no provider calls, no promotion.
Read at invocation time via `is_auto_layout_enabled()`.

Changing deployment flags waits for deployment authorization. The user's
quality preference permits a deliberate quality configuration after
validation.

## Budgets

- Max 8 samples per scene, min 3 usable samples for a decision.
- Provider timeout and retries bounded; failures retain previous strategy.
- No live API calls in tests; provider is injectable.

## Visual-validation limits

Real-footage review across genres (B-roll, action/sports, product demo,
voiceover, talking head/poster, two-person podcast, desktop tutorial/webcam,
presentation/presenter, gameplay/facecam, mixed screen/natural) has NOT been
performed. OpusClip benchmark comparison remains unverified. Policy
correctness is established via synthetic integration tests; the benchmark
dependency is explicitly retained for follow-up.
