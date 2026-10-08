# Editorial refinement (Task 09)

Transcript-grounded editorial planning pass that runs after windowed
candidate discovery (Task 08). For each post-dedupe discovery candidate
it builds a bounded transcript-context packet, requests ONE validated
edit decision per candidate (`accept` / `adjust` / `reject`), validates
the decision independently of the provider, and optionally merges
explicitly linked candidates into continuity groups. Nothing here
analyzes video or audio, renders clips, or claims measured audience
response.

## Purpose

Discovery finds moments worth clipping. Editorial refinement decides,
per candidate and on transcript evidence only, whether the interval is
already complete (`accept`), needs word-aligned start/end adjustment to
include the premise and payoff (`adjust`), or should be dropped
honestly (`reject`). Failures keep the discovery interval as explicitly
`unrefined` — never editor-approved.

## Packet schema (`build_context_packet`)

Deterministic dict, fixed key order:

- `candidate_id`, `moment_fingerprint`, `start_s`, `end_s` — ABSOLUTE
  timestamps, never rebased (`packet.start_s == candidate.start_s`
  exactly; validation rejects rebased packets).
- `context_start_s`, `context_end_s` — packet span, clipped to
  `[0, video_duration]`.
- `context_before_available`, `context_after_available` — booleans;
  `False` only when the *source edge* truncates that side (never
  silently omitted when the budget truncates).
- `context_before`, `candidate_words`, `context_after` — word dicts
  (`t`/`s`/`e`, plus `sp` when present; speaker `0` preserved).
  Bounded by `CLIPPYME_EDIT_CTX_WORDS` (default 60, 0–400) words per
  side and additionally by `CLIPPYME_EDIT_CTX_SECS` (default 90s, 0–600)
  per side.
- `evidence_text`, `evidence_context`, `expected_payoff`, `event_key`
  from discovery.
- `duration_min_s`, `duration_max_s` — resolved Task 04 policy bounds.
- `adjacent_before`, `adjacent_after` — up to 2 candidate IDs each
  side, nearest by time.
- `speaker_ids` (as discovered, `0` preserved),
  `evidence_disclosure: "transcript-only"`.

## Prompt contract (`build_editorial_prompt`)

- Transcript block with absolute `[t][spk:id]` lines and explicit
  `>>> CANDIDATE START xs` / `<<< CANDIDATE END xs` markers.
- Header: candidate ID, moment fingerprint, discovery
  evidence/context/payoff, duration bounds (hard limits), adjacent IDs,
  transcript-only disclosure.
- Strict JSON schema for ONE decision: `candidate_id`, `verdict`,
  `proposed_start_s`, `proposed_end_s`, `opening_evidence`,
  `premise_evidence`, `payoff_evidence`, `ending_evidence`,
  `evidence_basis`, `quality` (8 fields, 0–1 or null),
  `decision_reason`, `continuity_links[]`, `bridge_evidence`.
- Explicitly forbids requesting: hooks, titles, captions, hashtags,
  speaker names, virality probability, retention, visual/audio facts,
  platform copy.
- User preferences fenced as `<untrusted_user_preferences>` and
  sanitized **exactly** like Task 08 (module-level
  `_sanitize_pref_value`, shared with `build_discovery_prompt`;
  instructions capped at 2000 chars).

## Validation rules (`validate_editorial_decision`)

Raises `EditorialError` with exact reason strings; one malformed
decision never erases other candidates (the runner isolates it).

- Identity: decision must be an object; `candidate_id` must exist in
  the candidate set and match the packet; packet timestamps must equal
  the candidate interval; verdict ∈ {accept, adjust, reject}.
- Numerics: finite non-boolean numbers only (bools, NaN, Inf rejected).
- Geometry: proposed interval inside the packet context AND
  `[0, video_duration]`; must overlap the original candidate; must
  cover ≥1 real word; boundaries word-aligned within
  `WORD_ALIGN_TOL_S = 0.5`s — strictly inside a word beyond tolerance
  is "cut through a word".
- Snapping stores safe exact edges: a tolerated boundary snaps to the
  nearest word edge within tolerance that is **not strictly inside any
  valid word**. For overlapping speech an exact edge of one word may
  sit inside another speaker's word — such edges are unsafe and never
  stored. If no safe edge exists within tolerance, the decision is kept
  `unrefined` with conflict `boundary_conflict` (kept discovery
  interval) instead of storing a boundary that cuts content. All
  context/source/overlap/duration/neighbor/quote/payoff checks re-run
  against the snapped interval.
- Quotes: each of the four evidence fields non-empty and a contiguous
  normalized token-slice match inside the proposed interval
  (fabricated → `quote not grounded: <field>`); `evidence_basis` is
  **required** and must be `"transcript"` for every verdict, and
  `decision_reason` must be a non-empty bounded string — including for
  `reject` (no manufactured reasons); title-cased tokens absent
  verbatim from the interval words are `invented name`.
- `accept` requires the proposed interval to equal the candidate
  within 1e-6 (else it is an `adjust`).
- `adjust` content preservation: every quote span inside the proposed
  interval; never end before the grounded payoff span
  (`ends before grounded payoff`); never introduce NEW overlap with a
  neighbor guard interval (`violates neighbor guard`) — pre-existing
  overlap kept by discovery dedupe is not a violation.
- Duration conflicts: when the complete story (grounded quote span)
  cannot fit the resolved policy, or the packet context cannot reach
  the minimum, the discovery interval is kept as `unrefined` with
  conflict `duration_conflict` — never greedy truncation.
- `reject` needs no interval checks beyond identity.

## Quality vector and score

Eight bounded fields: `opening_strength`, `context_sufficiency`,
`payoff_delivery`, `useful_value`, `entertaining_value`,
`transcript_support`, `editability`, `confidence` — each in [0,1] or
`None`. Unknown stays unknown: missing, boolean, non-finite, or
out-of-range values become `None`, never fail the decision.

`compute_editorial_score` = mean of validated present fields × 100,
rounded to 1 decimal (`None` when none validated). It is an **editorial
recommendation score** for sorting — never labeled virality
probability or measured audience response.

Selection gates: `context_sufficiency` and `payoff_delivery` must
both be present and ≥ 0.5, else the decision degrades to `unrefined`
with reason `gate_failed: context/payoff incomplete`. The gates cover
`accept` **and** `adjust` — a strong opening cannot hide an incomplete
clip, and continuity groups are built only from refined plans that
passed them. No quota padding: sparse inputs stay sparse.

## Continuity rules (`plan_continuity_group`)

The quality path **never** calls `_extract_key_entities`,
`_has_narrative_continuity`, or `consolidate_narrative_continuations`.
A shared capitalized name, generated title, "then" phrase, proximity,
or overlap alone can never merge candidates.

A merge requires: ≥2 distinct valid IDs (no duplicates); source order;
gap ≤ `CONTINUITY_MAX_GAP_S = 120s`; bridge evidence grounded as a
contiguous token slice in the transcript *strictly between* the two
intervals (required whenever intervening words exist); compatible event
keys (all equal, non-None) **or** identical moment fingerprint; merged
`[min start, max end]` within the resolved duration policy and source;
all component words + bridge words covered; every component quote
re-grounded in the merged interval; quality recomputed as the per-field
mean (`None` if any component `None`); score recomputed.

Accepted merges append a `group:<id>+<id>` plan; rejected merges are
recorded with reasons. Candidates sharing a `moment_fingerprint` are
**variants** of one distinct moment (`group_variants`) and never
inflate `distinct_moments`.

Final metrics describe selected editorial outcomes only:
`distinct_moments` / `selected_variants` exclude `reject` verdicts,
retain explicit `unrefined` discovery candidates (they remain selected
per failure semantics), and collapse each accepted continuity group
exactly once. Candidates with a missing/empty fingerprint are never
collapsed together — each unknown identity stays its own distinct
moment.

## Budgets, flag, failure semantics

- `CLIPPYME_EDITORIAL_REFINEMENT` rollback flag (default ON;
  `0`/`false`/`no` disables). Disabled mode reproduces Task 08
  selection and metadata with zero provider calls.
- Separate `EditorialBudget`: `CLIPPYME_EDIT_MAX_ATTEMPTS` (24),
  `CLIPPYME_EDIT_DEADLINE_S` (600s), `CLIPPYME_EDIT_SDK_TIMEOUT_MS`
  (120s); per-attempt finite timeout; SDK retries disabled
  (`{"attempts": 1}`); model fallback loop mirrors discovery.
- Every owned client closed exactly once; a factory result identical
  to the caller's already-owned instance is borrowed, never
  double-closed.
- Per-candidate failure reasons: `budget_skipped`,
  `client_setup_failed`, `no_client_factory`, `parse_failed`,
  `generate_failed:<detail>`, `invalid_decision:<reason>`,
  `duration_conflict`, `gate_failed: ...`, `already_planned`,
  `refinement_disabled`. Failed/budget-skipped decisions keep the
  discovery candidate as explicitly `unrefined`.

## Funnel schema (`editorial_refinement`)

Versioned (`version: "1"`), reconciled record:

- `enabled`, `config` (model ids, context budgets, duration bounds),
- `packets_planned/attempted/succeeded/failed/skipped`,
- `accept_count/adjust_count/reject_count/unrefined_count`,
- `continuity_proposed/accepted/rejected` +
  `continuity_rejected_reasons`,
- `duration_conflicts`, `distinct_moments`, `selected_variants`,
- `budget`, `rejections`, `counts_reconcile` with equalities:
  `planned == succeeded+failed+skipped`;
  `accept+adjust+reject+unrefined == planned`;
  `continuity_proposed == accepted+rejected`.

No secrets or full provider responses in metadata.

## Resume semantics

`run_editorial_refinement(..., prior_plans={candidate_id: plan_dict})`
reuses a completed plan (counted as `skipped`, verdict still counted)
**only after revalidating it against the current transcript packet and
candidate** with the same material invariants as a fresh result:
overlap-safe exact word edges; `accept` interval equals the candidate
interval; `adjust` interval overlaps the candidate and stays inside the
packet bounds; source and resolved duration bounds; grounded quotes,
payoff preservation, and neighbor guards; status/verdict consistency;
non-empty reason; and the context/payoff selection gates for every
refined `accept`/`adjust`. Invalid prior data is never trusted as
completed work and never feeds continuity: it is retained as
`unrefined` with an exact `invalid_prior_plan:<detail>` reason
(recorded in `rejections`). The queued orchestrator resume path
returns saved metadata as-is — plans preserved, not repeated. Both
direct CLI and queued paths persist `editorial_refinement` +
`editorial_plans` through the atomic metadata writes (`_keep_funnel` /
`_preserve_funnel`).

## Honesty statement

This is transcript-grounded editorial **planning** only.
Audiovisual/silent-action discovery, execution of discontinuous cuts,
framing, creative finishing, and paired real-footage measurement are
later tasks. Quality scores are editorial estimates, not virality
predictions or audience measurements.
