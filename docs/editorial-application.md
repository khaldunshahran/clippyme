# Editorial plan application (Task 10)

Task 09 produces validated editorial plans, but the render loops still
consumed the original discovery `shorts` boundaries. This task closes
that gap for continuous clips: `run_windowed_discovery` now applies the
validated plans to the final `shorts` list exactly once, after Task 09
returns plans and before the result leaves discovery.

## What it does

`src/clippyme/pipeline/editorial_application.py` (pure, deterministic,
no provider calls):

- **accept** — retains the original short and interval.
- **adjust** — copies the short and replaces only `start`/`end`;
  titles, descriptions, hook, hashtags, score, reason, speaker metadata,
  and unknown additive fields are preserved.
- **reject** — removes the short with an auditable reason; rejected IDs
  never reach the source-slice call.
- **unrefined** — keeps the original short; the discovery interval is
  honestly retained (never editor-approved).
- **invalid/missing data** — keeps the original short, marks it
  `application_skipped` with the exact reason (duplicate IDs,
  missing plan/store entry, fingerprint mismatch, bool/NaN/Infinity,
  reversed/out-of-source/out-of-policy bounds, malformed state, unknown
  plan version, store-record identity mismatch
  (`candidate_store_id_mismatch`), malformed store/short intervals
  (`invalid_store_boundaries` / `invalid_short_boundaries`), short/store
  interval disagreement (`short_store_interval_mismatch`), unknown source
  duration (`invalid_source_duration` -- never authorizes a negative cut),
  plan interval not describing the current candidate
  (`plan_interval_mismatch` for accept/reject/unrefined,
  `plan_interval_disjoint_from_candidate` for adjust), and non-adjust
  group verdicts (`group_invalid_verdict`). Since review 02, an
  invalid source duration (missing, boolean, non-finite, zero, or
  negative) blocks every applicable Task 10 clip: each original
  short is retained with `invalid_source_duration` and no group
  may apply. Selected short and store intervals must also sit
  inside `[0, source_duration]`; escaping current evidence is
  skipped with `candidate_boundaries_out_of_source`. The store record must carry
  the exact `candidate_id`; store and short intervals must each be
  finite, non-boolean, and ordered, and must agree; accept/reject/
  unrefined plans retain the exact candidate interval while adjust plans
  must overlap it. Never silently approves or rejects; never erases
  unrelated clips.
- **continuity groups** — a `group:c1+c2` plan replaces its components
  only when every check passes (refined **adjust** verdict, unique
  non-empty components,
  all present, all survived as refined accept/adjust, finite/in-bounds/
  policy-compliant interval covering all applied component intervals,
  no identity conflict with a base ID). The group is inserted at the
  earliest component position, exposes `component_candidate_ids`, copies
  display fields from the earliest component (nothing invented), and
  keeps the editorial recommendation in `editorial_recommendation_score`
  — never relabeled as virality. Failed groups retain the individuals
  and record the rejection reason.
- Candidate-store items absent from the final shorts are never promoted;
  no quota padding.
- Inputs are never mutated (deep-copied); identical inputs always yield
  identical outputs and funnels.

## Funnel record

The returned `editorial_application` record carries: version/flag,
original selected count, plans supplied, accepts retained, adjustments
applied, rejects removed, unrefined retained, application skips by
reason, groups proposed/applied/rejected with reasons, final selected
count, exact reconciliation (`counts_reconcile`), and original +
applied intervals for every changed/removed/grouped ID. No provider
responses, no secrets.

## Rollback

`CLIPPYME_APPLY_EDITORIAL_PLANS` (default ON) gates the step; it only
runs when Task 09 ran on the quality path (`editorial_ran`). When off,
shorts are returned byte-for-byte unchanged with an explicit disabled
record. Legacy/non-windowed results without candidate IDs pass through
unchanged; empty stays empty.

## Wiring

- `candidate_discovery._finalize` runs the application exactly once and
  returns `editorial_application` beside the discovery/editorial
  records. Downstream clip loops consume the applied list, so adjusted
  timestamps reach the real source-slice call (`build_cut_command`)
  and rejected IDs never reach cutting.
- Direct CLI (`main.py`) and queued orchestrator (`orchestrator.py`)
  persist the record atomically (existing tmp+`os.replace` /
  `_atomic_json` paths) and preserve it through TextTiling/whole-video
  fallback metadata via `preserve_application_through_fallback` — the
  fallback record explicitly states plans were NOT applied to fallback
  shorts, while the earlier record survives for audit under
  `superseded_record`. Fallback shorts never inherit stale plans.

## Intentional empty results

When enabled editorial application reconciles a non-empty selection
down to zero shorts (every candidate rejected), that empty set is the
validated outcome -- not a failure. Both the direct CLI (`main.py`,
via `_resolve_empty_shorts_action`) and the queued orchestrator
(`orchestrator._load_or_analyze`, via
`is_intentional_editorial_empty`) persist zero shorts and render
nothing: no TextTiling and no whole-video fallback. Genuine
discovery/provider failures (empty discovery, errors, no reconciled
all-rejected record) keep the existing fallback behavior.

## Scope limits

This applies continuous editorial boundaries to newly rendered clips.
It does not execute discontinuous interval removals, discover silent
audiovisual moments, improve active-speaker/action framing, finish
hooks/captions, or establish quality on real paired footage.

## Strict intentional-empty detection (review 02)

Only a current APPLICATION_VERSION record with exact boolean
flags, non-boolean nonnegative integer counts, an actual empty
shorts list, and independently verified zero-output arithmetic
(every original selection actually rejected; nothing retained,
adjusted, or grouped) may suppress TextTiling/whole-video
fallback. Stale versions, type-coerced counts (booleans,
strings, floats), or inconsistent metadata never do.
