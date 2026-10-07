# Honest Performance Feedback (Task 03)

## What changed

Removed fabricated performance claims from automatic clip selection,
rescoring prompts, and the analytics insights panel.

**Before:** `performance_feedback.py` invented a 3.2x duration multiplier,
78% retention, whole-story dominance, and 2.8x comment velocity with no
measurements. `analyticsView.jsx` invented 84% completion and claimed the
learning engine was active.

**After:** Only validated descriptive observations are returned. No causal
rules are inferred.

## Backend

### `performance_feedback.py`

- `analyze_performance_patterns()` validates registry shapes and metric
  values (finite, non-negative counts, retention in [0,1]; rejects booleans,
  NaN/Infinity). Invalid entries are isolated; valid ones are kept.
- Returns `has_data`, `sample_size`, `status`, `reason`, plus null/empty
  legacy insight keys (`story_duration_multiplier`, `top_duration_tier`,
  `avg_completion_rate`, `top_hook_patterns`, `recommendations`).
- Useful observations identify metric, count, and denominator.
- `has_data` means "valid recorded measurements exist", not "a learned
  pattern exists".
- Never mutates the loaded registry.

### `get_learned_patterns_prompt()`

Returns `""` for current legacy records. No empirical duration, hook-theme,
completion, or comment-velocity directive enters default selection/rescoring.
Explicit `build_viral_prompt(learned_patterns=...)` overrides are preserved.

### `analytics_service.py`

Narrow additive `retention_recorded` presence marker: set True only when
`retention_rate` is explicitly provided and valid in `update_clip_metrics`.
Distinguishes a real recorded zero from legacy default unknown zero.
Preserved on republish. Existing files work without migration (absent = unknown).

## Frontend (`analyticsView.jsx`)

- Honest evidence state: "Performance Observations" / "Not Active in Selection".
- No invented percentages, multipliers, or "Active in Pipeline" claims.
- Retention input allows empty (unknown); only sends when explicitly set.
- Leaderboard distinguishes recorded 0% from Unknown.
- Save wording: "Performance metrics saved." (not "learning updated").

## Limitations

Comparable-cohort learning (creator, genre, post-age, experiment/version,
per-platform attribution) is not implemented. A future system would need
those cohorts before inferring causal rules. Real-footage and OpusClip
benchmarks remain unverified.

## Numeric validation and honest totals policy

- Only actual `int`/`float` values (excluding booleans) are accepted as
  numeric evidence. Float-convertible strings like `"100"` or `"0.8"` are
  rejected; they cannot crash valid neighboring records or contaminate
  observed averages.
- Integer inputs sum with exact integer arithmetic. `9007199254740993`
  stays `9007199254740993`, never rounded through float.
- Overflow is DISCLOSED per metric in the API output
  (`totals.integrity.<metric>: {overflow, inexact, status}`). An
  overflowed total is `None` (unavailable with a reason), never zero and
  never presented as an exact "Recorded Total". The dashboard must show
  this state honestly, not as 0 or an exact number.
- All API output stays strict-JSON serializable (`allow_nan=False`).
  Totals never become `Infinity`.

## Sample provenance

- Retention observations distinguish `explicit_count` (boolean marker
  `retention_recorded is True`) from `legacy_positive_count` (positive
  finite values without a marker).
- The combined average is labeled "observed", not "explicitly recorded".
- Averages are unweighted; the denominator (`N measured clips`) is
  exposed in both the feedback output and `get_analytics_summary`
  (`retention_sample_count`, `retention_averaging: "unweighted"`).
