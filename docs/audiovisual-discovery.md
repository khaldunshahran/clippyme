# Audiovisual discovery (Task 11)

Bounded audiovisual evidence for speech-light and visual moments:
reactions, reveals, action, demonstrations, applause, music/performance
beats. Implemented in `src/clippyme/pipeline/av_discovery.py` with narrow
additions to `gemini_request.py` (prompt contract) and `gemini_parser.py`
(strict validation). Pure, deterministic, no provider calls in the core
module; tests inject fakes and never decode real media.

## Honest limits

This implements **bounded sparse audiovisual discovery with
injected-provider verification**. It does **not** prove:

- recall or editorial quality on real footage (no real-footage quality
  claim; the benchmark and later framing/creative milestones own that);
- continuous action understanding (frames are sparse observations at
  listed timestamps, never proof of what happened between samples);
- active-speaker framing or camera trajectories;
- OpusClip parity.

Referencing a frame proves only that the model observed that sampled
frame. Model-produced visual descriptions are labeled as observations,
never verified fact. Sound events are never inferred from loudness.

## Module API — `av_discovery.py`

Public functions (see `__all__` in the module):

**Config**
- `av_discovery_enabled() -> bool` — master flag (`CLIPPYME_AV_DISCOVERY`,
  default on). The path additionally requires a readable source file.
- `source_media_usable(source_path) -> bool` — readable regular file check.
- `AVDiscoveryConfig.from_env() -> AVDiscoveryConfig` — validated, bounded
  config. See "Config / env vars" below.
- `FrameLimits.from_config(config) -> FrameLimits` — sampler caps.

**Source identity**
- `source_identity_details(source_path, probe_bytes=65536) -> dict` —
  `{"fingerprint", "size_bytes", "mtime_ns", "probe_bytes"}`. Fingerprint
  format: `"src-" + sha256(first probe_bytes || size_bytes || mtime_ns)
  .hexdigest()[:16]`. Any head/size/mtime change invalidates it.
- `fingerprint_source(source_path, probe_bytes=65536) -> str` — the
  fingerprint string. Raises `SourceIdentityError` when unreadable.
  Used to gate checkpoint/resume reuse: a resumed record is reusable only
  when fingerprint, audiovisual config/version, window plan, and evidence
  manifest identities still match.

**Media-window planning**
- `plan_media_windows(source_duration, transcript_windows, config=None)
  -> dict` — deterministic tiling of `[0, source_duration)` with absolute
  timestamps. Transcript windows (dicts with `start_s`/`end_s`, `(start,
  end)` tuples, or objects with `start_s`/`end_s`) falling inside a media
  window are coalesced (`kind="transcript"`, indices recorded); the rest
  are `kind="media"`. Media windows are their own planned entities with
  explicit intervals. Returns `{"windows": [MediaWindow...],
  "unattempted_windows": [{"window_index","start_s","end_s","reason"}...],
  "coverage": {...}, "transcript_windows_seen": int,
  "transcript_windows_ignored": int}`.
  - Caps leave trailing windows unattempted with exact reasons
    `media_window_cap` / `frame_budget_cap`; intervals reported explicitly.
  - Lifecycle distinction: **planned interval** (here) vs
    **provider-attempted** (orchestrator per-window status) vs **exact
    sampled-frame timestamps** (window sample result). Never conflated.
- `window_frame_timestamps(window, n_frames) -> [float]` — deterministic
  centered absolute sample timestamps, millisecond-rounded, strictly
  inside the window. Never rebased.

**Frame-sampling seam**
- `FrameSampler` — injectable callable protocol:
  `sample_frames(source_path, timestamps, limits) -> [FrameSample]`
  where `FrameSample = (requested_ts, decoded_ts, frame_bytes)` with
  absolute source seconds. Tests inject fakes; production uses
  `ffmpeg_frame_sampler`.
- `ffmpeg_frame_sampler(source_path, timestamps, limits=None,
  *, ffmpeg_bin="ffmpeg") -> [FrameSample]` — fast-seek (`-ss` before
  `-i`), one frame per timestamp, downscale filter enforcing
  `max_frame_width` and `max_frame_pixels`, MJPEG encode with a quality
  ladder (`jpeg_quality_steps`, default `(5, 12, 20)`) until under
  `max_frame_bytes`. The decoded PTS comes from `showinfo` stderr and is
  authoritative. Source opened read-only, never modified; no temp files
  (bytes captured from stdout). Raises `SamplingError` on decode failure,
  missing decoded timestamps, or cap violations; `FileNotFoundError` when
  ffmpeg is missing.
- `sample_media_window(source_path, window, sampler, limits=None,
  n_frames=None) -> WindowSample` — plans timestamps, calls the seam,
  assigns opaque IDs (`vf_<window_index>_<seq>` via `make_frame_id`),
  records requested + decoded timestamps, enforces the per-window byte
  cap. Failures (sampler error, contract violation, decoded timestamp
  outside the window, byte cap) yield `status="failed"` with an exact
  reason — the caller falls back to transcript-only evidence for that
  window and reports it. `WindowSample.metadata()` strips frame bytes
  (timestamps/counts only) for job metadata.
- `SamplingBudget(config, time_fn=...)` — local frames/bytes/deadline
  budget (`charge()`, `exhausted()`), mirroring the Task 08 shape.

**Audio events**
- `normalize_audio_events(raw_events, source_duration, id_prefix="ae",
  max_label_chars=64) -> (events, issues)` — keeps an event only when the
  label is a bounded nonempty string and start/end are finite,
  non-boolean, ordered, within source bounds. Assigns local IDs
  (`ae_<n>` via `make_audio_event_id`; pass `id_prefix="ae_w3"` for
  window scoping). Invalid events become issue records with exact
  reasons (`event_not_object`, `invalid_event_label`,
  `boolean_event_time`, `invalid_event_time`, `event_time_reversed`,
  `event_out_of_source_bounds`). No loudness inference exists.

**Local identity (never provider IDs)**
- `make_av_candidate_id(source_fingerprint, start_s, end_s,
  evidence_basis, evidence_refs, window_index=0, evidence_text="")`
  → `"avcand-" + hex[:16]`. Distinct per window occurrence.
- `make_av_moment_fingerprint(source_fingerprint, start_s, end_s,
  evidence_basis, evidence_refs, evidence_text="")`
  → `"avmoment-" + hex[:16]`. Stable across windows; the dedupe key
  (time overlap alone never dedupes).

**Strict validation**
- `validate_av_evidence(item, *, window_start_s, window_end_s,
  window_index=0, source_duration=None, frames_by_id=None,
  audio_events_by_id=None, window_words=(), min_duration=None,
  max_duration=None, source_fingerprint="src",
  max_visual_description_chars=500) -> ValidatedAVCandidate` —
  raises `EvidenceValidationError` (with exact `.reason`) on any
  violation. `min_duration`/`max_duration` are the resolved Task 04
  policy bounds — applied here, not re-resolved (resolve upstream with
  `duration_policy.resolve_duration_policy`).

## Evidence ID scheme

- Frames: `vf_<window_index>_<seq>` (e.g. `vf_2_0`) — assigned by
  `sample_media_window`, scoped to the window that sampled them.
- Audio events: `ae_<n>` (e.g. `ae_0`; window-scoped `ae_w3_0` via
  `id_prefix`) — assigned by `normalize_audio_events`.
- Candidates: `avcand-<hex16>`; moment fingerprints: `avmoment-<hex16>`.
- Source: `src-<hex16>` (see fingerprint format above).
- All IDs are opaque and local. Provider-supplied `id`/`candidate_id`
  fields are ignored. Frame bytes are transient request material and
  never persisted in job metadata or checkpoints.

## Validation rules and exact rejection reasons

`evidence_basis` must be one of `transcript | visual | audio_event |
multimodal`, and must match the references:

| basis | requires | forbids |
|---|---|---|
| `transcript` | grounded verbatim quote (contiguous token slice of the candidate's actual words) | `visual_frame_ids`, `audio_event_ids` |
| `visual` | ≥ 1 `visual_frame_ids` naming supplied frames from the same window | `audio_event_ids` |
| `audio_event` | ≥ 1 `audio_event_ids` naming validated events overlapping the interval | `visual_frame_ids` |
| `multimodal` | grounded quote **and** ≥ 1 visual or audio reference | — |

A wordless range is eligible **only** with ≥ 1 valid visual or
audio-event reference. Transcript grounding is the strict contiguous
token-slice rule (no subsequence fallback): character substrings never
count.

Exact rejection reasons (`EvidenceValidationError.reason`):

- `candidate_not_object`, `missing_start_end`, `boolean_time`,
  `non_finite_time`, `interval_reversed`
- `interval_out_of_window`, `interval_out_of_source`
- `duration_below_minimum`, `duration_above_maximum` (resolved Task 04
  bounds)
- `empty_context`, `empty_expected_payoff`
- `unsupported_evidence_basis`
- `invalid_visual_frame_ids_type` / `invalid_audio_event_ids_type`,
  `invalid_visual_frame_ids_entry` / `invalid_audio_event_ids_entry`
- `unknown_frame_id:<id>` — missing or foreign (another window's) frame
- `unknown_audio_event_id:<id>`,
  `audio_event_not_overlapping:<id>`
- `wordless_without_evidence`
- `missing_evidence_text`, `evidence_not_grounded`
- `unexpected_visual_reference`, `unexpected_audio_reference`
- `missing_visual_reference`, `missing_audio_reference`
- `multimodal_missing_transcript_evidence`,
  `multimodal_missing_media_reference`
- `empty_visual_description`, `visual_description_too_long`,
  `invalid_visual_description_type` (non-string description)
- `non_finite_confidence`, `non_finite_score`,
  `confidence_out_of_range` ([0,1]), `score_out_of_range` ([1,100])

## Checkpoint identity and resume

`av_checkpoint_identity` persists `av_prompt_schema_version`
explicitly (the current `AV_PROMPT_SCHEMA_VERSION` from
`gemini_request.py`, alongside the provider-contract digest).
Resume validation compares it field-for-field: a checkpoint whose
explicit version differs from the running code forces re-analysis
instead of reuse, even when every digest still matches.

## Config / env vars (`CLIPPYME_AV_DISCOVERY`)

| Variable | Default | Meaning |
|---|---|---|
| `CLIPPYME_AV_DISCOVERY` | `1` (on) | master flag; `0`/`false`/`no` disables |
| `CLIPPYME_AV_WINDOW_TARGET_S` | `240.0` | media window target length (s) |
| `CLIPPYME_AV_WINDOW_OVERLAP_S` | `45.0` | overlap between media windows (s) |
| `CLIPPYME_AV_MAX_WINDOWS` | `16` | max media windows planned |
| `CLIPPYME_AV_MAX_TOTAL_FRAMES` | `64` | max sampled frames per invocation |
| `CLIPPYME_AV_FRAMES_PER_WINDOW` | `3` | frames sampled per window |
| `CLIPPYME_AV_FRAME_MAX_WIDTH` | `320` | downscale width cap (px) |
| `CLIPPYME_AV_FRAME_MAX_PIXELS` | `102400` | downscale pixel cap |
| `CLIPPYME_AV_MAX_FRAME_BYTES` | `65536` | per-frame byte cap |
| `CLIPPYME_AV_MAX_WINDOW_BYTES` | `262144` | per-window byte cap |
| `CLIPPYME_AV_MAX_INVOCATION_BYTES` | `4194304` | invocation byte cap |
| `CLIPPYME_AV_SAMPLING_TIMEOUT_S` | `30.0` | per-operation monotonic
| | | sampling deadline shared by every frame decode and JPEG-quality |
| | | subprocess; remaining time is recomputed before each attempt |
| `CLIPPYME_AV_FINGERPRINT_PROBE_BYTES` | `65536` | head bytes fingerprinted |
| `CLIPPYME_AV_MAX_EVENT_LABEL_CHARS` | `64` | audio-event label cap |
| `CLIPPYME_AV_MAX_VISUAL_DESC_CHARS` | `500` | model description cap |
| `CLIPPYME_AV_MAX_MANIFEST_FRAMES` | `12` | frames per provider packet |
| `CLIPPYME_AV_MAX_MANIFEST_EVENTS` | `20` | events per provider packet |

## Byte / pixel limits

Per frame: ≤ `max_frame_bytes` (default 64 KiB), width ≤ 320 px,
pixels ≤ 102400 (enforced by the ffmpeg scale filter chain, then
verified in Python with a quality ladder). Per window: ≤
`max_window_bytes` (default 256 KiB). Per invocation: ≤
`max_invocation_bytes` (default 4 MiB) and ≤ `max_total_sampled_frames`
frames. Manifests carry IDs and timestamps only — never frame bytes.

## The sampler seam and how tests inject it

The seam is the `FrameSampler` protocol:
`sample_frames(source_path, timestamps, limits) -> [FrameSample]`.
Production wires `ffmpeg_frame_sampler`; tests pass fakes (see
`tests/pipeline/test_av_discovery.py::fake_sampler_factory`), so unit
tests never decode real media and never touch the network. The
production sampler is exercised only against a synthetic local fixture
(solid-color segments built with ffmpeg, all-intra for exact seeks):
in-range absolute timestamps, byte caps, distinguishable decoded colors,
and source-hash invariance are asserted there. A custom sampler that
writes temp files must clean them up itself (documented seam contract);
the built-in sampler and `sample_media_window` never write temp files.

## Request contract (`gemini_request.py`)

- `build_av_evidence_manifest(window, frames, audio_events=None, *,
  max_frames=12, max_events=20) -> dict` — bounded manifest with frame
  IDs + requested/decoded timestamps, event IDs + labels + intervals,
  and the sparse-observation disclaimer. Raises `ValueError` when a
  frame's decoded timestamp falls outside the window (rebased
  timestamps are a contract violation).
- `build_av_discovery_prompt(window, window_words, evidence_manifest,
  instructions=None, clip_type=None, duration_mode=None,
  duration_min_s=None, duration_max_s=None, audience_intel=None)
  -> str` — discovery-only prompt requiring `evidence_basis` and local
  refs; never asks for hooks, titles, captions, hashtags, speaker
  names, virality, retention, or unavailable facts. Preferences stay
  fenced/untrusted with existing caps.
- `build_discovery_prompt(..., evidence_packet=None)` — optional AV
  block appended to the Task 08 prompt; `None` keeps the transcript-only
  prompt byte-identical (backward compatible).
- `build_av_image_parts(sampled_frames, *, max_parts=12)` →
  `[{"frame_id","mime_type","data"}]` (SDK-free plain data);
  `av_image_parts_to_sdk(image_parts)` converts lazily via
  `google.genai.types` at the call site that owns the client.

## Parser contract (`gemini_parser.py`)

- `parse_av_discovery_payload(text, *, evidence_context) -> dict` —
  parses `{"candidates": [...]}` (malformed → `ValueError`, mirroring
  `parse_discovery_payload`), then validates each item independently
  via `av_discovery.validate_av_evidence`. Returns `{"candidates":
  [...], "rejections": [{"index","reason","detail"}...], "manifest":
  {...counts...}}`. Frames whose decoded timestamps fall outside the
  window are excluded from the manifest scope and reported in
  `manifest_frame_issues`, so references to them fail closed with
  `unknown_frame_id`. Visual descriptions return as
  `{"text": ..., "model_produced": True}`.

## Task 11 acceptance tests (`tests/pipeline/test_av_acceptance.py`)

39 tests exercising PRODUCTION entry points with injected deterministic
providers only (no live SDK calls, no network):

- `candidate_discovery.run_windowed_discovery` with `source_path`,
  `frame_sampler`, and `av_generate_fn`/`av_parse_fn` seams;
- `main.get_viral_clips` (the real direct entry point; heavy CV/ML
  imports are stubbed at test-module load so the real `main.py` code
  runs, with a deterministic fake `genai.Client`);
- `orchestrator._load_or_analyze` (the real queued analysis function,
  including the whole-video fallback path);
- `candidate_discovery.av_resume_reusable` and
  `orchestrator._analysis_checkpoint_reusable` (checkpoint/resume gates).

Coverage per acceptance bullet:

1. Wordless source -> bounded media windows covering `[0, duration)`,
   one visual candidate per window grounded in the actually sampled
   frame IDs; frame metadata carries IDs/timestamps/sizes only.
2. Speech-light gap planned with media-only windows while first/last
   source intervals and all Task 08 words stay covered.
3. `ffmpeg_frame_sampler` on a synthetic 9s tri-color fixture (all-intra):
   absolute in-range decoded timestamps, per-frame byte cap, clearly
   distinguishable decoded colors, source hash unchanged.
4. Transcript evidence keeps exact token-slice grounding and speaker ID
   `0`; a multimodal candidate references both a real quote and
   supplied frames (description labeled `model_produced`).
5. Foreign frame/event IDs, invalid audio-event types/times, invented
   phrases, wordless candidates without media evidence, out-of-window
   intervals, and duration violations isolated with exact reasons;
   invalid candidates never reach the store or shorts; funnels reconcile.
6. Two different visual moments sharing time survive selection; a
   genuine repeat (matching moment fingerprint) dedupes with basis
   `av_moment_fingerprint`; overlap alone never dedupes.
7. Score floor and `max_clips` apply only after global aggregation,
   including visual candidates; removals reconcile globally and in the
   AV sub-funnel.
8. `CLIPPYME_AV_DISCOVERY=0/false/no` or omitted source runs the
   accepted Task 10 behavior: zero sampler calls, no AV prompt parts,
   and byte-equivalent metadata outside the new disabled record.
9. A valid speech-free candidate reaches the direct (`get_viral_clips`)
   and queued (`_load_or_analyze`) render-selection loops (`shorts`)
   with an `unrefined` plan and the exact reason
   `speech_free_av_no_transcript_evidence`, retained by Task 10; a
   top-scored invalid item (unknown frame ID) never renders. The queued
   fallback preserves the AV discovery record and candidate store.
10. Direct and queued checkpoint/resume persist matching media identity
    (source fingerprint, AV config/version, window plan, evidence
    manifest digests); stale fingerprints/config digests and
    enablement changes force re-run, never silent reuse.
11. Provider/sampling budget exhaustion, decode failure, client
    setup/replacement failure, and parse failure isolate per window,
    preserve transcript candidates, clean temporary sampler assets,
    and close the owned client exactly once.

Mutation-sensitive gates (`TestMutationGates`): removing media-only
windows, rebasing frame timestamps, accepting an unknown frame ID,
treating overlap as duplication, dropping a wordless candidate in
Task 09 refinement, bypassing final caps, reusing a stale source
fingerprint, and losing AV metadata in fallback each fail the suite.
Where the production seam allows it, the gate additionally executes
the mutation (monkeypatched; `src/` untouched) and shows the
acceptance assertion failing, proving sensitivity.

Honest limits (unchanged): bounded sparse discovery with
injected-provider verification only. No recall or editorial-quality
claim on real footage, no continuous action understanding, no
active-speaker framing, no OpusClip parity.
