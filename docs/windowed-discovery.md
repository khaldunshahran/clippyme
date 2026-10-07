# Windowed Candidate Discovery (Task 08)

Transcript-window discovery for long sources. This task does NOT include
editorial finishing, multimodal video analysis, framing, rendering, or
frontend work. Real-footage acceptance measurement remains a later task.

## Why windows

A single whole-transcript model call degrades on long sources: late
material gets less attention and one malformed response erases
everything. Bounded overlapping windows (default 240s target, 45s
overlap) give each region its own discovery call. Overlap exists so
moments near a boundary are seen with context on both sides.

## Pipeline

1. **Plan** (`plan_windows`): deterministic windows from word-level
   absolute timestamps. Every valid word lands in >= 1 window; first
   and last words are never dropped. Invalid words are isolated and
   reported.
2. **Discover** (per window): `build_discovery_prompt` asks for
   moments only — strict JSON with start/end, event type, transcript
   evidence, context, expected payoff, optional event key, confidence,
   recommendation score. Speaker IDs pass through exactly.
3. **Validate** (`validate_discovery_candidate`): each candidate parsed
   independently. Rejects booleans, non-finite numbers, reversed or
   out-of-window intervals, missing evidence/context/payoff, and
   wordless ranges. Stable IDs derive from evidence + interval hash —
   never from model output.
4. **Deduplicate** (`dedupe_candidates`): global, after aggregation.
   Time overlap alone NEVER removes a candidate. Collapse requires a
   shared explicit `event_key` with overlap, or evidence similarity
   >= 0.55 with overlapping/adjacent intervals.
5. **Select**: existing `CLIPPYME_MIN_VIRAL_SCORE` floor and
   `CLIPPYME_MAX_CLIPS` cap run only after aggregation/dedupe.
6. **Funnel** (`candidate_discovery` in job metadata): versioned counts
   — planned/attempted/succeeded/failed windows, raw/valid/invalid
   candidates, duplicate groups/removals, floor removals, final count,
   rejection reasons. Unknown stays separate from zero.

## Budgets

Discovery has its own budget, separate from the render/output target:
`CLIPPYME_WINDOW_MAX_WINDOWS` (32), `CLIPPYME_WINDOW_MAX_CANDIDATES`
(200), `CLIPPYME_WINDOW_MAX_ATTEMPTS` (24),
`CLIPPYME_WINDOW_DEADLINE_S` (600). Every provider attempt consumes
the shared budget; SDK timeouts never exceed remaining time. A failed
window is recorded, never silently re-covered.

## Rollout

`CLIPPYME_WINDOWED_DISCOVERY` (default ON): `0/false/no` selects the
unchanged legacy single-pass path. The client is closed exactly once
on every exit of both paths.

## What this does not do

- Silent visual/action moments (transcript-only; coverage is
  transcript coverage, not media coverage).
- Editorial continuity decisions (Task 09).
- Hook/caption/title copy (discovery finds moments; it does not
  polish them).
