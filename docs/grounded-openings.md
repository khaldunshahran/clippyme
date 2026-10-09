# Grounded openings for the final edit (Task 15)

## Problem

Discovery creates `viral_hook_text` before final editorial/cut
boundaries are known. `gemini_parser.backfill_hook_text()` may derive
the hook from the source title or fall back to a generic teaser, and
project emission places it at the top for the first three seconds.
Neither a source title nor discovery evidence establishes what the
final retained interval actually delivers.

Task 15 adds a bounded, auditable opening-choice stage grounded in the
FINAL retained clip. It produces a small set of supported text options
and one explicit, deterministic selection.

## What is wired

- `pipeline/opening_plan.py` (pure core):
  - `build_opening_packet()` — built only after the last accepted
    boundary repair, from the exact interval passed to the actual
    cutter. Packet identity: source fingerprint, candidate/moment
    identity, exact final interval, retained words (absolute source
    timestamps preserved, clip-relative timing derived once),
    validated AV evidence, language/preferences, prompt/schema/config
    versions. Words outside the final interval are rejected — never
    included from removed lead-in, removed payoff, or neighboring
    clips. Anonymous `speaker0` / `unknown` speaker distinctions are
    preserved verbatim.
  - Narrow injectable provider seam: `set_opening_provider()` /
    `get_opening_provider()` (+ fallback). No live providers; the
    module performs no network, downloads, or decoding. One request
    per eligible clip, at most 3 distinct options per request. Each
    option carries a stable local ID, actual language, short
    rationale, cited evidence, and an explicit display interval.
  - Strict validation per option: strict types (no boolean
    numerics/coercion), finite ordered timing, bounded text/evidence
    sizes, evidence basis, grounded citations (evidence quotes must
    match complete normalized token slices inside retained words),
    source bounds, actual text language, non-empty reasons. No
    invented names/quotes/events/numbers/guaranteed outcomes/rankings/
    measured-performance claims (mechanical guards + citation
    requirement; semantic invention remains a provider-side
    obligation). Malformed options are rejected with exact reasons;
    valid neighbors survive. Dedupe keeps the first, never invents.
  - Deterministic selection (`evidence_count_desc_text_len_asc_id_asc`)
    with a qualitative confidence label (`supported` /
    `single_evidence`) — never a virality probability. Explicit
    user-authored overrides (`clip["hook"]`) win, are marked
    user-authored, and are never rewritten by a provider.
  - Display timing is explicit, clip-relative, half-open, inside
    `[0, clip duration)`. A malformed interval rejects the option —
    never silently defaulted to legacy timing. Task 06 delayed
    animation and exclusive-end semantics are preserved (the overlay
    start/end flow through the existing `hook_start`/`hook_end`
    render path). Cut start/end are never changed to fit a hook.
  - `OpeningBudget`: one monotonic deadline + bounded request count
    across all clips and fallback attempts (env:
    `CLIPPYME_OPENING_DEADLINE_S`, `CLIPPYME_OPENING_MAX_REQUESTS`,
    `CLIPPYME_OPENING_REQUEST_TIMEOUT_S`). SDK-internal retries are
    disabled by contract, every attempt is charged, per-attempt
    timeout is capped by remaining time, and the runner stops before
    exhaustion. Setup/prompt/provider failures are isolated per clip;
    initialized clients close exactly once on all exits.
  - Versioned opening funnel (`requested`/`validated`/`rejected`/
    `selected`/`no_selection`/`user_authored` + per-clip reasons)
    with exact reconciliation. Resume reuses a plan only on full
    identity + material validation match; a changed final
    interval/words/source/config forces re-analysis. Fully valid
    resumed plans need no client. Intentional zero-clip results stay
    empty; fallback never resurrects candidates.
- `CLIPPYME_GROUNDED_OPENINGS` (default OFF). Disabled mode reproduces
  accepted Task 14 behavior with zero opening-provider calls.
- Orchestrator integration (`_maybe_plan_opening`, called from
  `_render_one_clip` right after `_clip_bounds`): the direct and
  queued routes share this single integration point, so final cutter
  bounds reach the packet on both. Plans persist on
  `clip["opening_plan"]` + `clips_data["opening_funnel"]` via the
  existing atomic metadata save.
- Project emission (`pipeline/project_emit.py`): with the flag ON, the
  hook overlay uses only the validated selected option (explicit
  Task 06 timing) or an explicit user-authored override; with no
  grounded choice, no automatic hook is emitted. The old automatic
  hook is kept only as `opening.legacy_unverified_hook` — clearly
  unverified legacy metadata, never promoted as grounded. Flag OFF
  keeps legacy emission byte-for-byte.
- `domain/clip_project.py`: additive optional `opening` field
  (`OpeningSelection`: status, option identity, evidence, display
  timing, packet identity, versions, selection record,
  no-selection reason, legacy unverified hook). Old projects without
  it validate, save/load, undo, preview, and render unchanged.
  The JSON Schema twin is regenerated from the model.
- Preview/export use the selected text + Task 06 timing through the
  existing overlay → `hook_start`/`hook_end` path. The Task 05
  reframed base and Task 14 framing are untouched.

## What remains unstyled / unmeasured

- No live opening provider is wired: with the flag ON and no
  injected provider, clips honestly record `no_provider_configured`
  and emit no automatic hook. Provider selection, prompt tuning, and
  any model evaluation are future work.
- Hook taste, viral performance, caption safety, and OpusClip parity
  are NOT established by the synthetic fixtures here; they need the
  frozen human-reviewed real-footage benchmark.
- Caption collision / layout coordination with the hook overlay is
  Task 16.
- The emitted project is still the auto "default edit", not a fully
  styled finished export.

## Honest limits

- The mechanical hard-claim guards catch numbers, quote marks, and a
  documented phrase list; subtler semantic invention (names, events)
  relies on the evidence-citation requirement and provider behavior.
- Paraphrase support is a token-overlap heuristic (documented in
  `opening_plan._check_text_support`), not a semantic entailment
  check.
- Translation is recorded via the actual-language field and rationale;
  the validator does not translate or verify translations itself.
