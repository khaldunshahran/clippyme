# Grounded openings for the final edit (Task 15, rev 2 — review-01 corrections)

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
    cutter. Packet identity (sha256 over canonical JSON): source
    fingerprint, candidate/moment identity, EXACT final interval
    (exact float repr — no ms rounding, so 20.0001 vs 20.0002 differ),
    retained words (absolute source timestamps preserved,
    clip-relative timing derived once), canonical AV evidence
    identity + exact timing (`av_sha256`), words, language/
    preferences, prompt/schema/config/provider versions. Words
    outside the final interval are rejected — never included from
    removed lead-in, removed payoff, or neighboring clips.
  - Narrow injectable provider seam: `set_opening_provider()` /
    `get_opening_provider()` (+ fallback) plus replaceable factory
    registries (`set/get_opening_provider_factory`,
    `set/get_opening_fallback_factory`). No live providers in tests;
    the module performs no network, downloads, or decoding. One
    request per eligible clip, at most 3 distinct options per
    request. Each option carries a stable local ID, actual language,
    short rationale, cited evidence, and an explicit display
    interval.
  - `GeminiOpeningProvider` — lazy production adapter reusing the
    existing supported Gemini config (`GEMINI_API_KEY` + the
    auxiliary model config Task 15's prompt already targets), built
    lazily only when the flag is on, exactly one SDK call per
    `request_options` (no adapter retry loop; transient failures fall
    back within the budget), per-attempt timeout from the budget's
    remaining deadline via the established call config
    (`{"http_options": {"timeout": ..., "retry_options": {"attempts": 1}}}`),
    per-attempt client closed exactly once. Its `describe()` feeds
    packet identity.
  - Strict validation per option: strict types (no boolean
    numerics/coercion), finite ordered timing, bounded text/evidence
    sizes, evidence basis, grounded citations (evidence quotes must
    match complete normalized token slices inside retained words),
    source bounds, actual text language, non-empty reasons. No
    invented names/quotes/events/numbers/guaranteed outcomes/rankings/
    measured-performance claims. Malformed options are rejected with
    exact reasons; valid neighbors survive. Dedupe keeps the first,
    never invents.
  - **Material claim support** (speech evidence, English): the option
    text may only add low-information glue (`_GLUE_TOKENS`) to the
    evidenced wording — invented people/events/objects/numbers are
    rejected (`text_unsupported`). Supported paraphrases remain
    accepted; non-English speech falls back to the token-overlap
    rule. The option language must be a recognized language code/name
    (`bad_language`) and must match the packet's language or an
    explicit translation preference (`language_mismatch`); actual
    language and translation preferences are passed into the
    production prompt AND validation.
  - **Sensory citations** (AV evidence): a citation must bind to a
    specific accepted event's identity AND its interval — same-label
    invented intervals are rejected. Production AV evidence is read
    from the accepted Task 11 candidate evidence
    (`clips_data["candidate_store"][candidate_id]`, intersected with
    final bounds); `clip.opening_av_evidence` is kept only as a
    legacy/test alias. The prompt labels packet AV intervals as
    ABSOLUTE source times (display clocks are clip-relative, derived
    once).
  - **Speaker attribution** follows the Task 04 contract exactly:
    word-level speaker wins, integer 0 is valid and preserved, a word
    without a label stays unknown (no `speaker` key), segment
    summaries are never copied into word labels. Legacy verbatim
    string preservation is kept in `_window_words`.
  - Deterministic selection (`evidence_count_desc_text_len_asc_id_asc`)
    with a qualitative confidence label (`supported` /
    `single_evidence`) — never a virality probability. Explicit
    user-authored overrides (`clip["hook"]`) are handled FIRST —
    before bounds validation, fingerprinting, and packet
    construction — so no packet failure can ever block the authorized
    override; they are marked user-authored and never rewritten by a
    provider, without depending on provider readiness.
  - Display timing is explicit, clip-relative, half-open, inside
    `[0, clip duration)`. A malformed interval rejects the option —
    never silently defaulted to legacy timing. Task 06 delayed
    animation and exclusive-end semantics are preserved (the overlay
    start/end flow through the existing `hook_start`/`hook_end`
    render path). Cut start/end are never changed to fit a hook.
  - `OpeningBudget`: one monotonic deadline + bounded request count
    across all clips and fallback attempts (env:
    `CLIPPYME_OPENING_DEADLINE_S`, `CLIPPYME_OPENING_MAX_REQUESTS`,
    `CLIPPYME_OPENING_REQUEST_TIMEOUT_S`). `reset_run_budget()` is
    called at real job entries (`orchestrator.run()`,
    `main.__main__`) so jobs never inherit each other's budget.
    Ownership is explicit: caller-passed providers are NEVER closed
    by the stage (one job's shared provider survives all its clips);
    factory-built providers close exactly once; failed factory builds
    are isolated as setup failures. `try_charge()` never rounds a
    positive remainder to zero — sub-ms remainders produce positive
    timeouts. Every attempt (including fallback) is charged and
    counted; SDK-internal retries are disabled IN THE ADAPTER (it
    never loops on transient failures beyond the budgeted fallback).
    Setup/prompt/provider failures are isolated per clip.
  - Versioned opening funnel (`requested`/`validated`/`rejected`/
    `selected`/`no_selection`/`user_authored` + per-clip reasons)
    with exact reconciliation. Resume reuses a plan only on full
    identity + material validation match (a saved selection must be a
    member of the validated options); a changed final
    interval/words/source/config forces re-analysis. Transient
    no-provider/setup/budget failure with empty options is never
    resumable as a success. Fully valid resumed plans need no
    client. Intentional zero-clip results stay empty; fallback never
    resurrects candidates.
- `CLIPPYME_GROUNDED_OPENINGS` (default OFF). Disabled mode reproduces
  accepted Task 14 behavior with zero opening-provider calls.
- Entry points (both real routes): the stage is wired AFTER the final
  boundary repair and BEFORE the actual cutter in the direct CLI loop
  (`_plan_opening_for_direct_clip` in `main.py`, flag-gated, disabled
  mode is a no-op with zero provider calls) and in the queued path
  (`orchestrator._maybe_plan_opening`, called from `_render_one_clip`
  after `_clip_bounds`, passing `route="queued"` and the
  replaceable/default provider factory). With the flag ON and no
  provider configured, clips honestly record `no_provider_configured`
  and emit no automatic hook.
- Project emission (`pipeline/project_emit.py::_grounded_hook_overlay`)
  fails closed: the selected option must be a member of
  `validated_options`, the plan's `packet_id` must match the packet
  id recomputed from the CURRENT clip
  (`current_opening_packet_or_none`), and the selection's evidence
  must re-validate. A fabricated selected dict with
  `validated_options=[]` emits nothing. Stale automatic plans are
  cleared on packet-construction failure (funnel records
  `packet_build_failed`); user-authored overrides are preserved.
  With no grounded choice, no automatic hook is emitted; the old
  automatic hook is kept only as `opening.legacy_unverified_hook` —
  clearly unverified legacy metadata, never promoted as grounded.
  Flag OFF keeps legacy emission byte-for-byte. Task 06 strict
  half-open timing and user-edited working/archived projects are
  preserved.
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

- No live provider is configured by default: with the flag ON and no
  `GEMINI_API_KEY`, clips honestly record `no_provider_configured`
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
  documented phrase list; the material claim-support gate additionally
  rejects invented people/events/objects for English speech options
  (glue-vocabulary only). Subtler semantic invention in other
  languages still relies on the evidence-citation requirement and
  provider behavior.
- Paraphrase support is a token-overlap heuristic (documented in
  `opening_plan._check_text_support`), not a semantic entailment
  check. Translation is recorded via the actual-language field and
  explicit preference; the validator checks the language is a known
  code/name and matches the packet language or the explicit
  translation target — it does not translate or verify translations
  itself.
