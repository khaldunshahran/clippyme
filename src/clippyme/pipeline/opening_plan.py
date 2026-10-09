"""Grounded opening (hook) choices for the final edit — Task 15.

Pure core: builds a bounded, auditable opening packet from the FINAL
retained clip interval, validates provider-produced opening options
against retained evidence, and deterministically selects one.

Design rules (see docs/grounded-openings.md for the full contract):

* The packet is built ONLY after the last accepted boundary repair, from
  the exact interval passed to the actual cutter. Absolute source
  timestamps are preserved; clip-relative display timing is derived once.
* Never include words/claims from the removed lead-in, the removed
  payoff, or a neighboring clip: every retained word must lie inside
  the final interval, and the packet builder rejects anything else.
* Providers are INJECTED, never live: this module performs no network,
  no downloads, no decoding. One request per eligible clip, at most
  three distinct opening text options per request. The lazy production
  adapter (GeminiOpeningProvider) reuses the supported Gemini client
  configuration (GEMINI_API_KEY + auxiliary model) via a replaceable
  factory, builds its SDK client lazily on first request, makes exactly
  one SDK call per request (no internal retries — retries are disabled
  structurally in the adapter, not by a dict flag), and caps the call
  by the budgeted per-attempt timeout.
* Evidence quotes must match complete normalized token slices inside the
  retained words. Material claim support: an option may only add
  low-information glue words to its evidence (English); invented people,
  events, objects, numbers, endorsements, guaranteed outcomes, rankings,
  or measured-performance claims are rejected. If semantic support is
  not established the option is rejected (unknown/unverified), never
  selected or promoted as grounded. Genuinely supported paraphrases
  (reordering + glue) and non-English options are allowed; English is
  never forced.
* A speech-free option may use bounded visual/audio-event evidence inside
  the final interval, labeled explicitly, never described as speech or
  verified OCR. Sensory citations bind to a specific accepted event's
  identity AND interval.
* Strict validation per option and per clip: strict types (no boolean
  numerics/coercion), finite ordered timing, bounded text/evidence
  sizes, grounded citations, complete token matching, source bounds,
  recognized language codes, language/preference consistency, non-empty
  reasons. Malformed options are rejected with exact reasons; valid
  neighbors survive.
* Selection is deterministic from the validated options plus user
  preference, with a documented heuristic and a qualitative confidence
  label — never a virality probability.
* Explicit user-authored hook overrides win and are marked user-authored;
  no provider rewrites them. Override handling runs BEFORE packet
  construction so a packet failure can never block the authorized
  override.
* Display timing is explicit, clip-relative, half-open, inside
  [0, clip duration). A malformed interval rejects the option; it is
  never silently defaulted to legacy timing.
* A separate configurable opening invocation budget: one monotonic
  deadline and a bounded request count across all clips and fallback
  attempts. SDK-internal retries are disabled by contract, every attempt
  is charged, per-attempt timeout is capped by remaining time (sub-
  millisecond remainders still produce positive timeouts), and the
  runner stops before exhaustion. Ownership: a provider passed directly
  by the caller is caller-owned and is NEVER closed by the stage (one
  job's shared provider survives all its clips); a provider the stage
  builds itself via a factory is closed exactly once. Jobs never
  consume each other's budget: the job entry resets the run budget.
* The versioned opening funnel (requested/validated/rejected/selected/
  no-selection counts plus per-clip reasons) must reconcile exactly.
* Packet identity binds the EXACT final bounds (no ms rounding — exact
  float repr in the hash), canonical AV identity/timing, words,
  language/preferences, and the relevant model/fallback/prompt/schema/
  config. Resume reuses a plan only when the full current identity and
  material validation match; a changed final interval/words/source/
  evidence/config forces re-analysis. Transient no-provider/setup/
  budget failures with empty options are never resumable as successful
  plans. Fully valid resumed plans need no client.
* Word speaker evidence follows the accepted Task 04 contract: the
  word-level label wins, integer 0 is valid and preserved, a missing
  word label stays unknown (no key), and segment summaries are never
  copied into word labels.
* AV evidence comes from the accepted Task 11 candidate evidence
  (clips_data["candidate_store"][candidate_id]); clip.opening_av_evidence
  is kept only as a legacy/test alias.
* Emission fails closed: the selected option must be a member of the
  validated options, the plan's packet id must match the packet id
  recomputed from the current clip, and the selection's evidence must
  re-validate. Stale automatic plans are cleared on failed re-analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Versions, flag, and bounds
# ---------------------------------------------------------------------------

OPENING_PLAN_SCHEMA_VERSION = 1
"""Version of the opening packet/plan record shape."""

OPENING_PROMPT_VERSION = "opening-prompt/1"
"""Version of the deterministic prompt construction."""

OPENING_CONFIG_VERSION = "opening-config/1"
"""Version of the budget/config interpretation."""

MAX_OPENING_OPTIONS = 3
"""At most three distinct opening text options per request."""

MAX_TEXT_CHARS = 140
MAX_RATIONALE_CHARS = 280
MAX_EVIDENCE_ITEMS = 6
MAX_EVIDENCE_QUOTE_CHARS = 240
MAX_EVIDENCE_LABEL_CHARS = 120
MAX_OPTION_ID_CHARS = 64
MAX_LANGUAGE_CHARS = 16
MAX_PREFERENCES_CHARS = 2000
MAX_PROMPT_CHARS = 20000
MAX_AV_EVIDENCE_ITEMS = 12

_EVIDENCE_DELIMITER = "\n--- evidence ---\n"


def grounded_openings_enabled() -> bool:
    """CLIPPYME_GROUNDED_OPENINGS, default OFF.

    Disabled mode reproduces accepted Task 14 behavior with zero
    opening-provider calls.
    """
    return os.environ.get("CLIPPYME_GROUNDED_OPENINGS", "").strip().lower() in (
        "1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Token normalization (grounding primitive)
# ---------------------------------------------------------------------------

_TOKEN_STRIP_RE = re.compile(r"^[^\w']+|[^\w']+$", re.UNICODE)


def normalize_token(token: str) -> str:
    """Lowercase, strip surrounding punctuation. Empty string when nothing remains."""
    if not isinstance(token, str):
        return ""
    return _TOKEN_STRIP_RE.sub("", token.lower())


def normalize_tokens(text: str) -> list[str]:
    """Ordered normalized tokens; empties dropped."""
    if not isinstance(text, str):
        return []
    return [t for t in (normalize_token(w) for w in text.split()) if t]


# ---------------------------------------------------------------------------
# Word speaker evidence — the accepted Task 04 contract
# ---------------------------------------------------------------------------

def _canonical_word_speaker(value: Any) -> Optional[int]:
    """Task 04 word-attribution contract for speaker evidence.

    Implements the same contract as the authoritative Task 04
    normalizer (``gemini_request._canonical_speaker``): the word-level
    label wins, integer 0 is valid and preserved (falsy but valid),
    a missing label stays unknown (None — never mapped to 0), and
    bool/negative/non-integer/non-numeric labels are rejected.
    Segment summaries are never consulted here.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    # Floats that are exact integers (e.g. 1.0 from JSON) are accepted.
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


# ---------------------------------------------------------------------------
# Preferences (sanitized, capped, delimiter-safe)
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_preferences(preferences: Any) -> dict:
    """Sanitize user preferences into a bounded dict.

    Accepts a dict or a JSON/plain string. Caps total serialized size,
    strips control characters, and removes any occurrence of the
    evidence delimiter so preferences cannot break prompt fencing.
    Never raises: unparseable input becomes {}.
    """
    prefs: dict[str, Any] = {}
    try:
        if isinstance(preferences, dict):
            prefs = dict(preferences)
        elif isinstance(preferences, str):
            text = preferences.strip()
            if text.startswith("{"):
                loaded = json.loads(text)
                if isinstance(loaded, dict):
                    prefs = loaded
            elif text:
                prefs = {"style_note": text}
    except (ValueError, TypeError):
        prefs = {}
    cleaned: dict[str, Any] = {}
    for key, value in prefs.items():
        if not isinstance(key, str):
            continue
        key = _CONTROL_RE.sub("", key)[:64]
        if isinstance(value, str):
            value = _CONTROL_RE.sub("", value)
            value = value.replace(_EVIDENCE_DELIMITER.strip(), " ")
        elif isinstance(value, list):
            value = [
                _CONTROL_RE.sub("", str(v))[:160]
                for v in value[:12]
                if isinstance(v, (str, int, float)) and not isinstance(v, bool)
            ]
        elif isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        cleaned[key] = value
        serialized = json.dumps(cleaned, ensure_ascii=False)
        if len(serialized) > MAX_PREFERENCES_CHARS:
            # Drop the key that pushed us over; keep the rest.
            del cleaned[key]
            break
    return cleaned


def user_preference_from_env() -> dict:
    """Read CLIPPYME_OPENING_PREFERENCES (JSON or plain text), sanitized."""
    return sanitize_preferences(os.environ.get("CLIPPYME_OPENING_PREFERENCES", ""))


# ---------------------------------------------------------------------------
# Language codes (recognized languages; English never forced)
# ---------------------------------------------------------------------------

# ISO 639-1 codes. An option's language must be a recognized code (or a
# known language name below) — a non-empty arbitrary string is not a
# language.
_LANGUAGE_CODES = frozenset((
    "aa", "ab", "ae", "af", "ak", "am", "an", "ar", "as", "av", "ay", "az",
    "ba", "be", "bg", "bh", "bi", "bm", "bn", "bo", "br", "bs",
    "ca", "ce", "ch", "co", "cr", "cs", "cu", "cv", "cy",
    "da", "de", "dv", "dz",
    "ee", "el", "en", "eo", "es", "et", "eu",
    "fa", "ff", "fi", "fj", "fo", "fr", "fy",
    "ga", "gd", "gl", "gn", "gu", "gv",
    "ha", "he", "hi", "ho", "hr", "ht", "hu", "hy", "hz",
    "ia", "id", "ie", "ig", "ii", "ik", "io", "is", "it", "iu",
    "ja", "jv",
    "ka", "kg", "ki", "kj", "kk", "kl", "km", "kn", "ko", "kr", "ks",
    "ku", "kv", "kw", "ky",
    "la", "lb", "lg", "li", "ln", "lo", "lt", "lu", "lv",
    "mg", "mh", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my",
    "na", "nb", "nd", "ne", "ng", "nl", "nn", "no", "nr", "nv", "ny",
    "oc", "oj", "om", "or", "os",
    "pa", "pi", "pl", "ps", "pt",
    "qu",
    "rm", "rn", "ro", "ru", "rw",
    "sa", "sc", "sd", "se", "sg", "si", "sk", "sl", "sm", "sn", "so",
    "sq", "sr", "ss", "st", "su", "sv", "sw",
    "ta", "te", "tg", "th", "ti", "tk", "tl", "tn", "to", "tr", "ts",
    "tt", "tw", "ty",
    "ug", "uk", "ur", "uz",
    "ve", "vi", "vo",
    "wa", "wo",
    "xh",
    "yi", "yo",
    "za", "zh", "zu",
))

# Common English language names mapped to their code.
_LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "finnish": "fi", "french": "fr",
    "german": "de", "swedish": "sv", "italian": "it", "portuguese": "pt",
    "dutch": "nl", "norwegian": "nb", "danish": "da", "polish": "pl",
    "russian": "ru", "ukrainian": "uk", "japanese": "ja", "chinese": "zh",
    "korean": "ko", "arabic": "ar", "hindi": "hi", "turkish": "tr",
    "greek": "el", "czech": "cs", "hungarian": "hu", "romanian": "ro",
    "thai": "th", "vietnamese": "vi", "indonesian": "id", "malay": "ms",
    "swahili": "sw", "hebrew": "he", "bengali": "bn", "tamil": "ta",
    "telugu": "te", "urdu": "ur", "persian": "fa", "farsi": "fa",
}


def _canonical_language(value: Any) -> Optional[str]:
    """Recognized language -> ISO 639-1 code; None when not recognized."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in _LANGUAGE_CODES:
        return text
    return _LANGUAGE_NAMES.get(text)


def _expected_option_language(packet: dict[str, Any]) -> Optional[str]:
    """The language the option text must be written in, when known.

    An explicit translation preference wins; otherwise the packet
    (clip) language. None when neither is a recognized language.
    """
    prefs = packet.get("preferences") or {}
    if isinstance(prefs, dict):
        for key in ("translate_to", "target_language", "language"):
            canon = _canonical_language(prefs.get(key))
            if canon is not None:
                return canon
    return _canonical_language(packet.get("language"))


# ---------------------------------------------------------------------------
# Packet construction
# ---------------------------------------------------------------------------

class PacketBuildError(ValueError):
    """The final interval/words cannot form a grounded packet."""


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _window_words(
    words: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    """Words overlapping [start, end): the retained evidence window.

    Same window rule as project_emit._clip_relative_words: absolute
    source timestamps preserved (EXACT — no ms rounding, so tiny
    boundary changes stay distinguishable in the packet identity),
    clip-relative timing derived once.
    Words outside the final interval are dropped here; the builder
    additionally rejects a packet whose caller-supplied words are not
    already windowed (defense in depth against lead-in/payoff leakage).
    """
    out: list[dict[str, Any]] = []
    for w in words or []:
        if not isinstance(w, dict):
            continue
        ws, we = w.get("start"), w.get("end")
        if not _is_finite_number(ws) or not _is_finite_number(we):
            continue
        if we <= start or ws >= end:
            continue
        text = (w.get("word") or "").strip()
        if not text:
            continue
        rs = max(0.0, ws - start)
        re_ = max(rs, min(we, end) - start)
        rs, re_ = round(rs, 3), round(re_, 3)
        if re_ <= rs:
            continue
        entry: dict[str, Any] = {
            "w": text,
            "abs_start": float(ws),
            "abs_end": float(we),
            "start": rs,
            "end": re_,
        }
        # Task 04 contract: word-level speaker wins; integer 0 is valid
        # and preserved; a missing word label stays unknown (no key).
        # Legacy verbatim string labels are preserved as-is for
        # backward compatibility with already-windowed evidence.
        speaker = _canonical_word_speaker(w.get("speaker"))
        if speaker is not None:
            entry["speaker"] = speaker
        elif isinstance(w.get("speaker"), str) and w.get("speaker").strip():
            entry["speaker"] = w.get("speaker").strip()[:64]
        out.append(entry)
    out.sort(key=lambda e: (e["abs_start"], e["abs_end"]))
    return out


def _validate_av_evidence(
    av_evidence: Any, start: float, end: float
) -> list[dict[str, Any]]:
    """Bounded visual/audio-event evidence inside the final interval.

    Absolute source timestamps are preserved EXACTLY (no ms rounding)
    so shifted evidence changes the packet identity. Each item keeps
    its event identity (event_id when supplied) so same-label
    different-event evidence stays distinguishable.
    """
    if av_evidence is None:
        return []
    if not isinstance(av_evidence, list):
        raise PacketBuildError("av_evidence must be a list")
    if len(av_evidence) > MAX_AV_EVIDENCE_ITEMS:
        raise PacketBuildError(
            f"av_evidence exceeds {MAX_AV_EVIDENCE_ITEMS} items")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(av_evidence):
        if not isinstance(item, dict):
            raise PacketBuildError("av_evidence item must be an object")
        kind = item.get("kind")
        if kind not in ("visual", "audio"):
            raise PacketBuildError(
                f"av_evidence kind must be visual|audio, got {kind!r}")
        label = item.get("label")
        if not isinstance(label, str) or not label.strip():
            raise PacketBuildError("av_evidence label must be non-empty text")
        if len(label) > MAX_EVIDENCE_LABEL_CHARS:
            raise PacketBuildError("av_evidence label too long")
        es, ee = item.get("start"), item.get("end")
        if not _is_finite_number(es) or not _is_finite_number(ee):
            raise PacketBuildError("av_evidence interval must be finite numeric")
        if not (start - 1e-6 <= es <= ee <= end + 1e-6):
            raise PacketBuildError(
                "av_evidence interval outside the final clip interval")
        entry: dict[str, Any] = {
            "kind": kind,
            "label": " ".join(label.split()),
            "start": float(es),
            "end": float(ee),
        }
        event_id = item.get("event_id")
        if isinstance(event_id, str) and event_id.strip():
            entry["event_id"] = event_id.strip()[:128]
        else:
            entry["event_id"] = f"av-{idx}"
        provenance = item.get("source_fingerprint")
        if isinstance(provenance, str) and provenance.strip():
            entry["source_fingerprint"] = provenance.strip()[:256]
        out.append(entry)
    return out


def _words_hash(windowed: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for e in windowed:
        h.update(e["w"].encode("utf-8", errors="replace"))
        h.update(b"\x00")
        # Exact float repr: no ms rounding, so tiny boundary changes
        # change the identity.
        h.update(repr(e["abs_start"]).encode("ascii"))
        h.update(b"\x00")
        h.update(repr(e["abs_end"]).encode("ascii"))
        h.update(b"\x00")
        if "speaker" in e:
            h.update(str(e["speaker"]).encode("utf-8", errors="replace"))
            h.update(b"\x00")
    return h.hexdigest()


def _av_identity_hash(evidence: list[dict[str, Any]]) -> str:
    """Canonical AV identity: event identity + kind + label + exact timing."""
    h = hashlib.sha256()
    for ev in evidence:
        h.update(str(ev.get("event_id", "")).encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(str(ev.get("kind", "")).encode("ascii"))
        h.update(b"\x00")
        h.update(str(ev.get("label", "")).encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(repr(ev.get("start")).encode("ascii"))
        h.update(b"\x00")
        h.update(repr(ev.get("end")).encode("ascii"))
        h.update(b"\x00")
        h.update(str(ev.get("source_fingerprint", "")).encode(
            "utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()


def build_opening_packet(
    *,
    source_fingerprint: str,
    final_start: float,
    final_end: float,
    words: list[dict[str, Any]],
    candidate_id: Optional[str] = None,
    moment_id: Optional[str] = None,
    av_evidence: Optional[list[dict[str, Any]]] = None,
    language: Optional[str] = None,
    preferences: Optional[dict[str, Any]] = None,
    provider_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build the bounded opening packet from the FINAL cutter interval.

    ``words`` must already be windowed to [final_start, final_end) —
    the builder rejects any word outside the final interval so removed
    lead-in/payoff or neighboring-clip words can never leak in.

    The packet identity binds the EXACT final bounds (no ms rounding),
    the canonical AV identity/timing, the words, the language/
    preferences, and the relevant provider model/fallback/prompt/
    schema/config — so any material change forces re-analysis.
    """
    if not _is_finite_number(final_start) or not _is_finite_number(final_end):
        raise PacketBuildError("final interval must be finite numeric")
    if not (final_end > final_start >= 0):
        raise PacketBuildError(
            f"invalid final interval [{final_start}, {final_end})")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise PacketBuildError("source_fingerprint must be a non-empty string")

    # Defense in depth: the caller windows words, but any straggler
    # outside the final interval fails the packet loudly.
    for w in words or []:
        if not isinstance(w, dict):
            raise PacketBuildError("word entries must be objects")
        ws, we = w.get("start"), w.get("end")
        if not _is_finite_number(ws) or not _is_finite_number(we):
            raise PacketBuildError("word timing must be finite numeric")
        if ws < final_start - 1e-6 or we > final_end + 1e-6:
            raise PacketBuildError(
                "word outside the final interval: "
                f"[{ws}, {we}) not in [{final_start}, {final_end})")

    windowed = _window_words(words, float(final_start), float(final_end))
    evidence = _validate_av_evidence(av_evidence, float(final_start), float(final_end))
    prefs = sanitize_preferences(preferences or {})
    prefs_hash = hashlib.sha256(
        json.dumps(prefs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if provider_config is not None and not isinstance(provider_config, dict):
        raise PacketBuildError("provider_config must be a dict or None")

    duration = round(float(final_end) - float(final_start), 3)
    identity = {
        "source_fingerprint": source_fingerprint,
        "candidate_id": candidate_id,
        "moment_id": moment_id,
        # EXACT bounds: no ms rounding (20.0001 and 20.0002 differ).
        "final_start": float(final_start),
        "final_end": float(final_end),
        "words_sha256": _words_hash(windowed),
        "av_sha256": _av_identity_hash(evidence),
        "language": language,
        "preferences_sha256": prefs_hash,
        "provider_config": provider_config,
        "prompt_version": OPENING_PROMPT_VERSION,
        "schema_version": OPENING_PLAN_SCHEMA_VERSION,
        "config_version": OPENING_CONFIG_VERSION,
    }
    packet_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    return {
        "packet_id": packet_id,
        "identity": identity,
        "clip_duration": duration,
        "words": windowed,
        "av_evidence": evidence,
        "language": language,
        "preferences": prefs,
    }


def _translation_target(prefs: dict[str, Any]) -> Optional[str]:
    for key in ("translate_to", "target_language", "language"):
        canon = _canonical_language(prefs.get(key))
        if canon is not None:
            return canon
    return None


def build_opening_prompt(packet: dict[str, Any]) -> str:
    """Deterministic, bounded prompt for the injected provider.

    Evidence is fenced with an explicit delimiter; preferences were
    sanitized at packet build time so they cannot break the fence.
    The prompt is never sent anywhere by this module — only handed to
    the injected provider.

    Packet AV intervals are ABSOLUTE source seconds; the prompt labels
    them as such and derives the clip-relative display clocks once.
    The actual option language (packet language or an explicit
    translation preference) is passed to the provider here.
    """
    words = packet.get("words") or []
    word_lines = []
    for e in words:
        speaker = f"[{e['speaker']}] " if e.get("speaker") is not None else ""
        word_lines.append(
            f"{e['start']:.2f}-{e['end']:.2f} {speaker}{e['w']}")
    identity = packet.get("identity") or {}
    final_start = identity.get("final_start", 0.0)
    try:
        final_start_f = float(final_start)
    except (TypeError, ValueError):
        final_start_f = 0.0
    evidence_lines = []
    for ev in packet.get("av_evidence") or []:
        rel_start = ev["start"] - final_start_f
        rel_end = ev["end"] - final_start_f
        evidence_lines.append(
            f"[{ev['kind']}] id={ev.get('event_id', '?')} "
            f"abs {ev['start']:.2f}-{ev['end']:.2f} "
            f"(clip {rel_start:.2f}-{rel_end:.2f}): {ev['label']}")
    prefs = packet.get("preferences") or {}
    pref_lines = [f"{k}: {v}" for k, v in sorted(prefs.items())]

    target = _translation_target(prefs)
    packet_lang = packet.get("language")
    if target is not None:
        lang_line = (
            f"- Write each option in '{target}' (explicit user translation "
            f"request). The language field must carry the ACTUAL language "
            f"of the hook text.")
    elif isinstance(packet_lang, str) and packet_lang.strip():
        lang_line = (
            f"- Write each option in '{packet_lang.strip()}' (the clip's "
            f"language). The language field must carry the ACTUAL language "
            f"of the hook text.")
    else:
        lang_line = (
            "- Write each option in the clip's spoken language. The "
            "language field must carry the ACTUAL language of the hook "
            "text.")

    parts = [
        "Choose up to 3 distinct opening text options for the start of "
        "this final clip. Rules:",
        "- Ground every option in the retained evidence below. Speech "
        "quotes must match the transcript words exactly (complete token "
        "slices). A paraphrase may only add low-information glue words — "
        "never invent people, events, objects, numbers, endorsements, "
        "guaranteed outcomes, rankings, or measured-performance claims.",
        "- Never invent names, quotes, events, numbers, guaranteed "
        "outcomes, rankings, or measured-performance claims.",
        "- A speech-free option may use the visual/audio events, labeled "
        "as visual or audio evidence; never describe it as speech or "
        "verified OCR.",
        "- Each option: stable id, text (<=140 chars), the ACTUAL language "
        "of the hook text as a language code, short rationale, cited "
        "evidence, explicit clip-relative display interval [start, end) "
        "inside the clip duration.",
        lang_line,
        "- Respond with ONLY a JSON array of option objects: "
        '[{"id": ..., "text": ..., "language": ..., "rationale": ..., '
        '"evidence_basis": "speech|visual|audio", "evidence": '
        '[{"basis": ..., "quote"|"label": ..., "start": ..., "end": ...}], '
        '"display": {"start": ..., "end": ...}}]. No other text.',
        f"Clip duration: {packet.get('clip_duration'):.2f}s.",
    ]
    if word_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("Retained transcript words (clip-relative seconds):")
        parts.extend(word_lines)
    if evidence_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("Accepted visual/audio events (ABSOLUTE source "
                     "seconds; clip-relative display in parentheses):")
        parts.extend(evidence_lines)
    if pref_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("User preferences (style requests only, never proof):")
        parts.extend(pref_lines)
    prompt = "\n".join(parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        # Truncate the word list first (keep head: openings favor early
        # material), never the rules.
        head = "\n".join(parts[:8])
        budget = MAX_PROMPT_CHARS - len(head) - 64
        tail = "\n".join(parts[8:])[: max(0, budget)]
        prompt = head + "\n[truncated]\n" + tail
    return prompt


# ---------------------------------------------------------------------------
# Provider seam (injected only — no live providers)
# ---------------------------------------------------------------------------

# A provider exposes request_options(request) -> list[dict] and,
# optionally, close() and describe(). The request carries the prompt,
# the packet, the per-attempt timeout, and the retries-disabled
# contract. Ownership: a provider passed directly to run_opening_stage
# is caller-owned and is NEVER closed by the stage; a provider the
# stage builds via a factory is closed exactly once by the stage.
Provider = Any

_OPENING_PROVIDER: Provider = None
_OPENING_FALLBACK_PROVIDER: Provider = None
_OPENING_PROVIDER_FACTORY: Optional[Callable[[], Provider]] = None


def set_opening_provider(provider: Provider) -> None:
    """Inject the opening provider (tests / wiring). None clears it."""
    global _OPENING_PROVIDER
    _OPENING_PROVIDER = provider


def get_opening_provider() -> Provider:
    return _OPENING_PROVIDER


def set_opening_fallback_provider(provider: Provider) -> None:
    global _OPENING_FALLBACK_PROVIDER
    _OPENING_FALLBACK_PROVIDER = provider


def get_opening_fallback_provider() -> Provider:
    return _OPENING_FALLBACK_PROVIDER


def set_opening_provider_factory(factory: Optional[Callable[[], Provider]]) -> None:
    """Inject the opening provider factory (tests / wiring). None clears it.

    When no factory is injected, the wiring falls back to the lazy
    production factory (Gemini). The factory is only invoked when the
    flag is on and a clip actually needs a provider.
    """
    global _OPENING_PROVIDER_FACTORY
    _OPENING_PROVIDER_FACTORY = factory


def get_opening_provider_factory() -> Optional[Callable[[], Provider]]:
    return _OPENING_PROVIDER_FACTORY


# ---------------------------------------------------------------------------
# Lazy production provider adapter (Gemini)
# ---------------------------------------------------------------------------

def _import_google_genai():
    """Import the google-genai SDK lazily (never at module import)."""
    try:
        from google import genai as _genai
        return _genai
    except Exception as exc:
        raise RuntimeError(
            "google-genai SDK is not available; inject a provider/factory "
            "for tests") from exc


def _opening_model_name() -> str:
    """Supported model configuration for opening choices.

    Reuses the existing supported config path (auxiliary/lightweight
    model selection); falls back to env/default when the config
    stack is unavailable. Lazy — never at module import.
    """
    try:
        from clippyme.pipeline.gemini_service import (
            get_auxiliary_gemini_model,
        )
        return get_auxiliary_gemini_model()
    except Exception:
        return (
            os.environ.get("GEMINI_LITE_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )


def _opening_fallback_models() -> list[str]:
    return [
        m.strip()
        for m in os.environ.get("GEMINI_FALLBACK_MODELS", "").split(",")
        if m.strip()
    ]


def _parse_opening_response(response: Any) -> list[dict[str, Any]]:
    """Parse the production JSON-array response into option dicts.

    Raises on empty/non-JSON/non-array payloads (the stage isolates
    these as provider failures). Option-level validation happens
    downstream in validate_opening_option — never here.
    """
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("opening provider returned an empty response")
    cleaned = text.strip()
    # Strip a single markdown fence if the model wrapped the JSON.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"opening provider response is not JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("opening provider response is not a JSON array")
    options = [item for item in data if isinstance(item, dict)]
    if len(options) != len(data):
        raise ValueError(
            "opening provider response array must contain only objects")
    return options


class OpeningBudgetExhausted(RuntimeError):
    """The opening budget was exhausted before an SDK call could start.

    Raised instead of granting more time than remains: a positive
    remainder that cannot be represented, or setup work that consumed
    the whole remainder, stops the attempt with this explicit budget
    reason. The stage records it as budget exhaustion, not a provider
    failure.
    """


class GeminiOpeningProvider:
    """Lazy production opening provider using the supported Gemini config.

    The SDK client is built lazily on first request (never at
    construction); the SDK module itself is injected for tests or
    imported lazily in production. Each request_options call makes
    EXACTLY ONE SDK call — no retry loop in the adapter (retries are
    disabled structurally here, plus the established
    ``retry_options: {attempts: 1}`` call config). The per-attempt
    timeout comes from the request's budgeted ``timeout_s``. The
    client is closed exactly once per attempt.
    """

    def __init__(
        self,
        *,
        client_factory: Callable[[], Any],
        model: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
    ):
        if not callable(client_factory):
            raise ValueError("client_factory must be callable")
        self._client_factory = client_factory
        self._model = model or _opening_model_name()
        self._fallback_models = list(fallback_models or [])
        self.attempts = 0

    def describe(self) -> dict[str, Any]:
        """Provider config for the packet identity."""
        return {
            "kind": "gemini-opening",
            "model": self._model,
            "fallback_models": list(self._fallback_models),
            "prompt_version": OPENING_PROMPT_VERSION,
        }

    def request_options(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        timeout_s = request.get("timeout_s")
        if not _is_finite_number(timeout_s) or timeout_s <= 0:
            raise ValueError("request timeout_s must be positive finite")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("request prompt must be non-empty text")
        # A fallback attempt may override the model for this call only.
        model = request.get("model") or self._model
        if not isinstance(model, str) or not model.strip():
            raise ValueError("request model must be non-empty text")
        client = self._client_factory()
        try:
            # Recompute the remaining budget AFTER client/setup work:
            # setup may have consumed part of the remainder, and the
            # adapter must never exceed what is actually left.
            remaining = request.get("remaining_s")
            if callable(remaining):
                try:
                    fresh = float(remaining())
                except Exception:
                    # The remaining-budget callback failed: the actual
                    # remainder cannot be verified. Never fall back to
                    # the stale request timeout; fail closed.
                    raise OpeningBudgetExhausted(
                        "remaining-budget callback failed before "
                        "the SDK call")
                effective = min(float(timeout_s), fresh)
            else:
                effective = float(timeout_s)
            if not _is_finite_number(effective) or effective <= 0:
                raise OpeningBudgetExhausted(
                    "opening budget exhausted before the SDK call")
            # Floor, never round up: the SDK timeout must not exceed
            # the remaining budget. If the remainder cannot represent
            # a strictly positive SDK timeout, stop BEFORE the SDK
            # call — never send timeout=0.
            timeout_ms = int(effective * 1000)
            if timeout_ms < 1:
                raise OpeningBudgetExhausted(
                    "remaining budget below the minimum representable "
                    "SDK timeout")
            # Single SDK attempt: no loop on transient failures. The
            # budgeted provider->fallback switch is the stage's job.
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "http_options": {
                        "timeout": timeout_ms,
                        "retry_options": {"attempts": 1},
                    },
                },
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self.attempts += 1
        return _parse_opening_response(response)

    def close(self) -> None:
        """No cached client: per-attempt clients close in request_options."""
        return None


class GeminiOpeningProviderFactory:
    """Default lazy production provider factory (replaceable).

    Called per clip by the stage; builds one GeminiOpeningProvider.
    describe() exposes the model/fallback config for the packet
    identity without building any client. The SDK client is built
    lazily inside the provider on first request — never here, and
    only when the flag is on (the wiring checks the flag first).
    """

    def __init__(self, *, sdk: Any = None):
        self._sdk = sdk

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "gemini-opening",
            "model": _opening_model_name(),
            "fallback_models": _opening_fallback_models(),
            "prompt_version": OPENING_PROMPT_VERSION,
        }

    def __call__(self) -> GeminiOpeningProvider:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        config = self.describe()
        sdk = self._sdk

        def _build_client():
            if sdk is not None:
                # Injected SDK (tests): no live credentials needed.
                return sdk.Client(api_key=api_key or "injected-test-key")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is not configured")
            return _import_google_genai().Client(api_key=api_key)

        return GeminiOpeningProvider(
            client_factory=_build_client,
            model=config["model"],
            fallback_models=config["fallback_models"],
        )


_DEFAULT_GEMINI_FACTORY = GeminiOpeningProviderFactory()


def default_opening_provider_factory() -> GeminiOpeningProviderFactory:
    """The default lazy production factory (Gemini). Replaceable via
    set_opening_provider_factory in tests/wiring."""
    return _DEFAULT_GEMINI_FACTORY


# Backwards-compatible alias used by the wiring.
def default_gemini_opening_provider_factory() -> GeminiOpeningProviderFactory:
    return default_opening_provider_factory()


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------

_EVIDENCE_BASES = ("speech", "visual", "audio")

# Mechanical hard-claim guards. The material claim-support contract
# below additionally rejects invented people/events/objects for
# English options; semantic invention in other languages falls back to
# the evidence-overlap rule (unknown/unverified, never promoted).
_HARD_CLAIM_PATTERNS = [
    (re.compile(r"\bguarantee[sd]?\b", re.I), "guaranteed outcome claim"),
    (re.compile(r"#\s*1\b"), "ranking claim"),
    (re.compile(r"\bnumber\s+one\b", re.I), "ranking claim"),
    (re.compile(r"\bworld\s+record\b", re.I), "measured-performance claim"),
    (re.compile(r"\bbest\s+ever\b", re.I), "ranking claim"),
    (re.compile(r"\bdoctors?\s+hate\b", re.I), "invented endorsement claim"),
    (re.compile(r"\bscientists?\s+(prove|say|confirm)\b", re.I),
     "invented authority claim"),
]

_MEASURE_WORDS = frozenset(
    {"million", "millions", "billion", "billions", "thousand", "thousands"}
)

_SPEECH_VERBS = frozenset({
    "said", "says", "say", "saying", "told", "tells", "telling",
    "speaks", "speaking", "spoke", "spoken", "announced", "announces",
    "claim", "claims", "claimed", "stated", "states", "according",
})

# Low-information glue words a paraphrase may add to its evidence.
# Polarity/negation words are NEVER glue: a novel negation flips the
# meaning of the evidence ("coffee never works" vs "coffee works") and
# must appear in the cited evidence itself. Deterministic and
# reviewable; not a semantic model.
_GLUE_TOKENS = frozenset((
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours",
    "he", "him", "his", "she", "her", "hers", "it", "its",
    "we", "us", "our", "ours", "they", "them", "their", "theirs",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    "and", "or", "but", "yet", "so", "for",
    "of", "in", "on", "at", "to", "from", "with", "by",
    "about", "as", "into", "over", "under", "after", "before",
    "between", "through", "during", "within",
    "against", "among",
    "yes",
    "very", "really", "just", "quite", "rather", "too", "also",
    "even", "still", "already",
    "here", "there", "now", "then", "today", "tonight",
    "always", "sometimes", "often",
    "oh", "wow", "hey", "hmm", "well", "okay", "ok",
    "watch", "see", "look", "listen", "hear",
))

# Negation/polarity words. A novel negation flips the meaning of the
# evidence ("coffee never works" against "coffee works") and is never
# low-information glue: it must appear in the cited evidence itself to
# be supported. Checked after the evidence/glue membership test so a
# negation that IS in the evidence stays supported.
_NEGATION_TOKENS = frozenset((
    "not", "no", "never", "n't", "cannot",
    "don't", "doesn't", "didn't", "won't", "can't",
    "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "couldn't", "shouldn't", "wouldn't", "mustn't",
    "without", "none", "nobody", "nothing", "neither", "nor",
))

# Event/outcome words. Naming a competition, prize, or outcome the
# evidence does not show ("championship", "won", "medal") invents an
# event. Descriptive paraphrase verbs/adjectives for the depicted
# action ("erupts" for "cheers") are deliberately NOT in this list.
_EVENT_OUTCOME_WORDS = frozenset((
    "championship", "championships", "campeonato", "campeonatos",
    "champion", "champions", "campeon", "campeones",
    "medal", "medals", "medalla", "medallas",
    "trophy", "trophies", "trofeo", "trofeos",
    "title", "titles", "titulo", "titulos",
    "victory", "victories", "victoria", "victorias",
    "win", "wins", "won", "winner", "winners",
    "lose", "loses", "lost", "loser", "losers",
    "defeat", "defeats", "defeated",
    "beat", "beats", "beaten",
    "score", "scores", "scored",
    "goal", "goals", "gol", "goles",
    "prize", "prizes", "premio", "premios",
    "award", "awards",
    "crown", "crowned",
    "final", "finals", "semifinal", "semifinals",
))

# Contraction negations: "doesn't" carries the same polarity as "not".
# A contraction is supported when its canonical negation appears in the
# cited evidence; a genuinely novel contraction is still rejected.
_CONTRACTION_NEGATIONS = {
    "don't": "not", "doesn't": "not", "didn't": "not",
    "won't": "not", "can't": "not", "isn't": "not",
    "aren't": "not", "wasn't": "not", "weren't": "not",
    "hasn't": "not", "haven't": "not", "hadn't": "not",
    "couldn't": "not", "shouldn't": "not", "wouldn't": "not",
    "mustn't": "not", "cannot": "not", "n't": "not",
}

# Words that look verb-like by suffix but are not verbs. Used by the
# sensory action-verb gate so descriptive prepositions/adjectives are
# not mistaken for action claims.
_NON_VERB_WORDS = frozenset({
    "across", "towards", "toward", "besides",
    "red",
})

# Bounded manner-paraphrase verbs for the sensory path. A novel verb is
# NOT a supported paraphrase by default: it is allowed only when it
# describes the manner/intensity of the evidenced action (never a new
# action or outcome). Maps a normalized evidence action verb to the
# allowed paraphrase verbs. Curated, reviewable, default-deny; not a
# semantic model.
_SENSORY_MANNER_VERBS = {
    "cheer": {"erupt", "erupts"},
    "cheers": {"erupt", "erupts"},
}


def _looks_like_verb(token: str) -> bool:
    """Heuristic: does a normalized token look like an inflected verb?

    Bounded approximation for the sensory action-verb gate. English
    verbs inflect with -ing/-ed/-s. This is a heuristic, not a tagger;
    it errs toward flagging (fail closed) and is documented as such.
    """
    if not token or token in _GLUE_TOKENS or token in _NON_VERB_WORDS:
        return False
    return (token.endswith("ing") or token.endswith("ed")
            or token.endswith("s"))


# Verb-position markers: a closed class of modals/auxiliaries plus the
# infinitive marker "to". A novel token directly after one of these
# sits in a verb slot ("can explode", "to detonate") even when the
# base form carries no inflectional suffix for _looks_like_verb to
# catch. This is a structural grammatical signal, not a content-word
# blacklist: it generalizes to any uninflected action without naming
# words.
_VERB_POSITION_MARKERS = frozenset({
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "to",
})


def _is_sensory_action_claim(token: str, prev_token: str) -> bool:
    """Is this novel token an action claim rather than a descriptive modifier?

    Two bounded structural signals, no content blacklist:
    * the inflection heuristic (_looks_like_verb) catches inflected
      forms ("explodes", "erupts");
    * verb position (directly after a modal/auxiliary or "to") catches
      uninflected base forms ("explode" in "can explode").
    An action claim is grounded only through the explicit bounded
    manner-paraphrase authorization; descriptive modifiers (nouns,
    adjectives in non-verb positions, e.g. "championship" used
    descriptively) stay allowed.
    """
    if _looks_like_verb(token):
        return True
    return bool(prev_token) and prev_token in _VERB_POSITION_MARKERS


def _is_manner_paraphrase(token: str, evidence_union: set[str]) -> bool:
    """Is this novel verb a bounded manner paraphrase of an evidenced action?"""
    for ev_token in evidence_union:
        allowed = _SENSORY_MANNER_VERBS.get(ev_token)
        if allowed is not None and token in allowed:
            return True
    return False


def _negation_count(tokens: list[str]) -> int:
    # A token reclassified as glue (e.g. by a weakening mutation) is
    # not a negation marker: the polarity check stays consistent with
    # the token classification so the glue-vocabulary mutation stays
    # load-bearing.
    return sum(1 for t in tokens
               if t in _NEGATION_TOKENS and t not in _GLUE_TOKENS)


def _content_tokens(tokens: list[str]) -> set[str]:
    """Non-glue, non-negation tokens: the claim-bearing words."""
    return {t for t in tokens
            if t and t not in _GLUE_TOKENS and t not in _NEGATION_TOKENS}


# Clause boundaries for per-claim polarity validation: coordinate and
# adversative conjunctions that join independent claims ("but", "and",
# "or", ...). Polarity is validated per claim/subject, not per quote,
# so a negation cannot be transferred from one claim to another.
_CLAUSE_BOUNDARY_TOKENS = frozenset({
    "but", "and", "or", "yet", "so", "nor",
    "while", "whereas", "although", "though", "however",
})

_CLAUSE_SPLIT_PUNCT_RE = re.compile(r"[,;:—–]")


def _split_clauses(text: str) -> list[list[str]]:
    """Split text into clauses (ordered normalized token lists).

    Splits on clause punctuation and boundary conjunctions. Empty
    fragments are dropped. Deterministic; not a parser.
    """
    clauses: list[list[str]] = []
    for part in _CLAUSE_SPLIT_PUNCT_RE.split(text or ""):
        current: list[str] = []
        for token in normalize_tokens(part):
            if token in _CLAUSE_BOUNDARY_TOKENS:
                if current:
                    clauses.append(current)
                current = []
            else:
                current.append(token)
        if current:
            clauses.append(current)
    return clauses


def _check_polarity_preserved(
    text: str, evidence: list[dict[str, Any]], *, text_key: str,
) -> Optional[str]:
    """Polarity preservation per claim/subject.

    Dropping (or adding) a negation reverses meaning even when every
    remaining token is supported ("coffee does work" from "coffee does
    not work"). Counting markers over the whole quote is not enough:
    "coffee does work but tea does not work" keeps one negation while
    reversing both claims of "coffee does not work but tea does work".
    Each option clause is therefore paired with the evidence clause it
    draws content from (best content-token overlap); the paired
    clauses must carry the same number of negation markers. A tie for
    best overlap with differing polarity fails closed (unverifiable
    relation); a clause with no evidence overlap is left to the
    material-claims gate.
    """
    opt_clauses = _split_clauses(text)
    ev_clauses: list[list[str]] = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        ev_clauses.extend(_split_clauses(item.get(text_key) or ""))
    for opt_tokens in opt_clauses:
        opt_content = _content_tokens(opt_tokens)
        if not opt_content:
            continue
        best = -1
        best_negs: set[int] = set()
        for ev_tokens in ev_clauses:
            overlap = len(opt_content & _content_tokens(ev_tokens))
            if overlap == 0:
                continue
            neg = _negation_count(ev_tokens)
            if overlap > best:
                best = overlap
                best_negs = {neg}
            elif overlap == best:
                best_negs.add(neg)
        if best < 0:
            # No evidence overlap: the material-claims gate decides.
            continue
        opt_neg = _negation_count(opt_tokens)
        if len(best_negs) != 1 or opt_neg not in best_negs:
            return (
                "option changes the polarity of the cited evidence "
                "(no evidence clause with matching negation scope; "
                f"option clause carries {opt_neg} negation marker(s))"
            )
    return None

_PROPER_STRIP_CHARS = " \t\n\"\u201c\u201d\u2018\u2019()[]{}:;,.!?\u00ab\u00bb\u00a1\u00bf"



def _reject(option_id: Any, code: str, reason: str) -> dict[str, Any]:
    return {
        "option_id": option_id if isinstance(option_id, str) else None,
        "code": code,
        "reason": reason,
    }


def _strict_text(value: Any, *, field: str, max_chars: int,
                 allow_empty: bool = False) -> tuple[Optional[str], Optional[str]]:
    """Strict text: must be str (no coercion), bounded, non-empty unless allowed."""
    if not isinstance(value, str):
        return None, f"{field} must be text, got {type(value).__name__}"
    text = value.strip()
    if not text and not allow_empty:
        return None, f"{field} must be non-empty"
    if len(text) > max_chars:
        return None, f"{field} exceeds {max_chars} chars"
    if _CONTROL_RE.search(text):
        return None, f"{field} contains control characters"
    return text, None


def _strict_interval(value: Any, *, duration: float) -> tuple[
        Optional[tuple[float, float]], Optional[str]]:
    """Explicit clip-relative half-open [start, end) inside [0, duration)."""
    if not isinstance(value, dict):
        return None, "display interval must be an object with start/end"
    start, end = value.get("start"), value.get("end")
    for name, bound in (("start", start), ("end", end)):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            return None, f"display {name} must be numeric, got {bound!r}"
        if not math.isfinite(bound):
            return None, f"display {name} must be finite, got {bound!r}"
    s, e = float(start), float(end)
    if s < 0 or e < 0:
        return None, "display interval must be >= 0"
    if e <= s:
        return None, "display interval end must be after start"
    if s >= duration:
        return None, "display interval starts at/after clip end"
    if e > duration + 1e-6:
        return None, "display interval extends past clip duration"
    return (round(s, 3), round(min(e, duration), 3)), None


def _word_token_stream(words: list[dict[str, Any]]) -> list[str]:
    return [t for e in words for t in normalize_tokens(e.get("w", ""))]


def _quote_is_token_slice(quote_tokens: list[str],
                          stream: list[str]) -> bool:
    """Complete normalized token slice: contiguous, in order, in stream."""
    if not quote_tokens:
        return False
    n = len(quote_tokens)
    return any(
        stream[i: i + n] == quote_tokens for i in range(len(stream) - n + 1)
    )


def _validate_speech_evidence(
    items: list[dict[str, Any]], stream: list[str]
) -> Optional[str]:
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return f"evidence[{idx}] must be an object"
        if item.get("basis") != "speech":
            return (f"evidence[{idx}] basis {item.get('basis')!r} does not "
                    f"match option evidence_basis 'speech'")
        quote, err = _strict_text(item.get("quote"), field=f"evidence[{idx}].quote",
                                  max_chars=MAX_EVIDENCE_QUOTE_CHARS)
        if err:
            return err
        tokens = normalize_tokens(quote)
        if not _quote_is_token_slice(tokens, stream):
            return (f"evidence[{idx}].quote is not a complete normalized "
                    f"token slice of the retained words")
    return None


def _validate_sensory_evidence(
    items: list[dict[str, Any]], basis: str,
    av_evidence: list[dict[str, Any]],
    source_fingerprint: Optional[str] = None,
) -> Optional[str]:
    """Bind each sensory citation to a specific accepted event's
    identity AND interval AND source.

    The cited ``event_id``, label, basis and interval must all belong
    to ONE accepted event: a forged event_id with a matching
    label/interval is rejected, and same-label events stay distinct
    (a citation binds the event whose id — or, without an id, whose
    label AND interval — it cites; a valid later event is never
    mistaken for the first label match). An event recorded from a
    different source is foreign evidence and is rejected. Fail closed.
    """
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return f"evidence[{idx}] must be an object"
        if item.get("basis") != basis:
            return (f"evidence[{idx}] basis {item.get('basis')!r} does not "
                    f"match option evidence_basis {basis!r}")
        label, err = _strict_text(item.get("label"),
                                  field=f"evidence[{idx}].label",
                                  max_chars=MAX_EVIDENCE_LABEL_CHARS)
        if err:
            return err
        es, ee = item.get("start"), item.get("end")
        # The citation interval is required and must be finite/ordered.
        if not _is_finite_number(es) or not _is_finite_number(ee):
            return (f"evidence[{idx}] must cite an explicit finite "
                    f"interval inside the final clip")
        if ee <= es:
            return f"evidence[{idx}] interval end must be after start"
        cited_id = item.get("event_id")
        cited_id = (cited_id.strip()[:128]
                    if isinstance(cited_id, str) and cited_id.strip()
                    else None)
        match = None
        if cited_id is not None:
            # Identity-first binding: the event_id selects the event;
            # label and interval must then agree with THAT event.
            for ev in av_evidence or []:
                if ev.get("kind") != basis:
                    continue
                if ev.get("event_id") == cited_id:
                    match = ev
                    break
            if match is None:
                return (f"evidence[{idx}].event_id does not match any "
                        f"accepted {basis} event inside the final interval")
            if _normalize_label(match.get("label", "")) != _normalize_label(label):
                return (f"evidence[{idx}].label does not match the cited "
                        f"event's label")
        else:
            # No event_id: bind label AND interval together so a valid
            # later same-label event is not mistaken for the first.
            for ev in av_evidence or []:
                if ev.get("kind") != basis:
                    continue
                if _normalize_label(ev.get("label", "")) != _normalize_label(label):
                    continue
                if (abs(float(es) - float(ev["start"])) <= 1e-6
                        and abs(float(ee) - float(ev["end"])) <= 1e-6):
                    match = ev
                    break
            if match is None:
                return (f"evidence[{idx}] does not cite an accepted "
                        f"{basis} event (label and interval must both "
                        f"match inside the final interval)")
        # The cited interval must be the bound event's interval — a
        # fabricated interval for a real event is rejected.
        if abs(float(es) - float(match["start"])) > 1e-6 or abs(
                float(ee) - float(match["end"])) > 1e-6:
            return (f"evidence[{idx}] interval [{es}, {ee}] does not match "
                    f"the cited {basis} event's interval "
                    f"[{match['start']}, {match['end']}]")
        # Provenance binding: foreign-source evidence is rejected.
        ev_source = match.get("source_fingerprint")
        if (ev_source and source_fingerprint
                and ev_source != source_fingerprint):
            return (f"evidence[{idx}] cites {basis} event from a "
                    f"different source")
    return None


def _normalize_label(label: str) -> str:
    return " ".join(normalize_tokens(label))


def _check_hard_claims(text: str, quote_tokens_union: set[str]) -> Optional[str]:
    if '"' in text or '"' in text or "'" in text and text.count("'") >= 2:
        # Quote marks in option text would present a quote not present
        # in evidence as verbatim.
        if '"' in text or '"' in text:
            return "option text contains quote marks (quotes must come from evidence)"
    for token in normalize_tokens(text):
        if re.fullmatch(r"\d+(\.\d+)?", token) and token not in quote_tokens_union:
            return (f"option text contains number {token!r} not grounded "
                    f"in cited evidence")
        if token in _MEASURE_WORDS and token not in quote_tokens_union:
            return (f"option text contains measured-performance word {token!r} "
                    f"not grounded in cited evidence")
    for pattern, label in _HARD_CLAIM_PATTERNS:
        if pattern.search(text):
            return f"option text contains {label}"
    return None


def _raw_text_tokens(text: str) -> list[str]:
    """Raw whitespace tokens with case preserved (for entity detection)."""
    if not isinstance(text, str):
        return []
    return [w for w in text.split() if w]


def _looks_like_proper_noun(raw_token: str, first_token: bool) -> bool:
    """A capitalized token names a person/entity (Bob, Alice, Alicia).

    Sentence-initial capitalization is conventional, so the first
    token is exempt — an invented sentence-initial person is still
    caught by the strict speech rule or the event/outcome list.
    Deterministic and reviewable; not a semantic model.
    """
    if first_token:
        return False
    stripped = raw_token.strip(_PROPER_STRIP_CHARS)
    return any(ch.isupper() for ch in stripped)


def _check_material_claims(
    text: str, evidence_union: set[str], *, basis: str,
) -> Optional[str]:
    """Material claim-support contract, applied consistently across
    languages and modalities.

    Citation validity or bag-of-words overlap cannot establish
    meaning, so every token of the option text is checked:
    * in the cited evidence -> supported;
    * in the low-information glue vocabulary -> allowed;
    * otherwise the novel token is classified deterministically:
      a capitalized token names an invented person/entity ("Bob",
      "Alice"); a novel negation flips polarity ("coffee never
      works"); a novel event/outcome word invents an event
      ("championship", "won", "medal"); any other novel token in a
      speech option is unsupported (the speech gate stays strict, and
      the review-01 glue-vocabulary mutation stays load-bearing);
      any other novel token in a speech-free option is an action claim
      only when it sits in a verb slot (inflected form, or base form
      after a modal/auxiliary/"to"): action claims are grounded solely
      through the explicit bounded manner-paraphrase authorization
      ("erupts" for "cheers"); anything else novel and unverifiable is
      rejected, while descriptive modifiers (nouns/adjectives in
      non-verb positions) stay allowed.
    Short names and action words are claims too: no minimum token
    length. At least one non-glue token must be evidence-grounded, so
    an all-glue text is never marked grounded. Genuinely supported
    paraphrases and non-English options pass; English is never
    forced and copied strings are never required.
    """
    # Module-global on purpose: the review-01 glue-vocabulary
    # fail-then-restore mutation stays load-bearing.
    glue = _GLUE_TOKENS
    grounded_content = False
    raw_tokens = _raw_text_tokens(text)
    norm_tokens = [normalize_token(r) for r in raw_tokens]
    for pos, raw in enumerate(raw_tokens):
        token = norm_tokens[pos]
        if not token:
            continue
        if token in evidence_union:
            if token not in glue:
                grounded_content = True
            continue
        if token in glue:
            continue
        if _looks_like_proper_noun(raw, pos == 0):
            return (f"option introduces a person/entity {raw.strip()!r} "
                    f"not supported by cited evidence")
        if token in _NEGATION_TOKENS:
            # A contraction ("doesn't") is supported when its canonical
            # negation ("not") is in the cited evidence; a genuinely
            # novel negation flips polarity and is never glue.
            canonical = _CONTRACTION_NEGATIONS.get(token, token)
            if (token not in evidence_union
                    and canonical not in evidence_union):
                return (f"option introduces negation/polarity {token!r} "
                        f"not supported by cited evidence")
            continue
        if token in _EVENT_OUTCOME_WORDS:
            return (f"option introduces an event/outcome {token!r} "
                    f"not supported by cited evidence")
        if basis == "speech":
            return ("option introduces material claims not supported by "
                    f"cited evidence: {token!r}")
        # Speech-free path: a novel action claim is not a descriptive
        # paraphrase by default. It is grounded only as a bounded
        # manner paraphrase of an evidenced action ("erupts" for
        # "cheers"); an unlisted novel action ("explodes"/"explode"
        # for "rolls") is unverifiable, never grounded. Action claims
        # are detected structurally: inflected forms via the suffix
        # heuristic, uninflected base forms via verb position (after a
        # modal/auxiliary or "to").
        prev_token = norm_tokens[pos - 1] if pos > 0 else ""
        if _is_sensory_action_claim(token, prev_token) and not _is_manner_paraphrase(
                token, evidence_union):
            return (f"option introduces an unverifiable sensory action "
                    f"{token!r} not supported by cited evidence")
        # Descriptive modifiers for the depicted event stay allowed
        # (never people/events/polarity).
    if not grounded_content:
        return ("option text has no evidence-grounded content token "
                "(unknown/unverified)")
    return None


def validate_opening_option(
    option: dict[str, Any], *, packet: dict[str, Any]
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Validate one provider option against the packet.

    Returns (normalized_option, None) or (None, rejection).
    """
    opt_id = option.get("id") if isinstance(option, dict) else None
    if not isinstance(option, dict):
        return None, _reject(opt_id, "not_an_object",
                             "option must be an object")
    raw_id = option.get("id")
    if not isinstance(raw_id, str) or not raw_id.strip():
        return None, _reject(opt_id, "bad_id", "option id must be non-empty text")
    if len(raw_id.strip()) > MAX_OPTION_ID_CHARS:
        return None, _reject(opt_id, "bad_id",
                             f"option id exceeds {MAX_OPTION_ID_CHARS} chars")
    option_id = raw_id.strip()

    text, err = _strict_text(option.get("text"), field="text",
                             max_chars=MAX_TEXT_CHARS)
    if err:
        return None, _reject(option_id, "bad_text", err)
    language, err = _strict_text(option.get("language"), field="language",
                                 max_chars=MAX_LANGUAGE_CHARS)
    if err:
        return None, _reject(option_id, "bad_language", err)
    # A non-empty arbitrary string is not a language: the option must
    # name a recognized language (code or known name).
    canon_lang = _canonical_language(language)
    if canon_lang is None:
        return None, _reject(option_id, "bad_language",
                             f"language {language!r} is not a recognized language")
    # The option text must be written in the packet language (or an
    # explicit translation preference) — actual language, verified.
    expected = _expected_option_language(packet)
    if expected is not None and canon_lang != expected:
        return None, _reject(
            option_id, "language_mismatch",
            f"option language {canon_lang!r} does not match the required "
            f"{expected!r} (packet language / explicit translation preference)")
    rationale, err = _strict_text(option.get("rationale"), field="rationale",
                                  max_chars=MAX_RATIONALE_CHARS)
    if err:
        return None, _reject(option_id, "bad_rationale", err)

    basis = option.get("evidence_basis")
    if basis not in _EVIDENCE_BASES:
        return None, _reject(
            option_id, "unsupported_evidence_basis",
            f"evidence_basis must be one of {list(_EVIDENCE_BASES)}, "
            f"got {basis!r}")

    evidence = option.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None, _reject(option_id, "bad_evidence",
                             "evidence must be a non-empty list")
    if len(evidence) > MAX_EVIDENCE_ITEMS:
        return None, _reject(option_id, "bad_evidence",
                             f"evidence exceeds {MAX_EVIDENCE_ITEMS} items")

    words = packet.get("words") or []
    stream = _word_token_stream(words)
    quote_union: set[str] = set()
    label_union: set[str] = set()
    if basis == "speech":
        if not stream:
            return None, _reject(option_id, "quote_not_grounded",
                                 "no retained speech to ground quotes in")
        err = _validate_speech_evidence(evidence, stream)
        if err:
            return None, _reject(option_id, "quote_not_grounded", err)
        for item in evidence:
            quote_union.update(normalize_tokens(item["quote"]))
    else:
        err = _validate_sensory_evidence(
            evidence, basis, packet.get("av_evidence") or [],
            source_fingerprint=(packet.get("identity") or {}).get(
                "source_fingerprint"))
        if err:
            return None, _reject(option_id, "evidence_not_accepted", err)
        for item in evidence:
            label_union.update(normalize_tokens(item["label"]))

    err = _check_hard_claims(text, quote_union)
    if err:
        return None, _reject(option_id, "hard_claim", err)
    # Polarity preservation: dropping (or adding) a negation reverses
    # meaning even when every remaining token is supported.
    err = _check_polarity_preserved(
        text, evidence,
        text_key="quote" if basis == "speech" else "label")
    if err:
        return None, _reject(option_id, "text_unsupported", err)
    if basis == "speech":
        err = _check_material_claims(text, quote_union, basis="speech")
    else:
        if set(normalize_tokens(text)) & _SPEECH_VERBS:
            err = ("speech-free option describes speech "
                   "without speech evidence")
        else:
            err = _check_material_claims(text, label_union, basis=basis)
    if err:
        return None, _reject(option_id, "text_unsupported", err)

    duration = packet.get("clip_duration")
    if not _is_finite_number(duration) or duration <= 0:
        return None, _reject(option_id, "bad_packet",
                             "packet clip_duration invalid")
    interval, err = _strict_interval(option.get("display"), duration=duration)
    if err:
        # Never silently default a malformed interval to legacy timing.
        code = ("display_interval_reversed" if "after start" in err
                else "display_interval_out_of_bounds"
                if "past clip" in err or "at/after clip end" in err
                else "bad_display_interval")
        return None, _reject(option_id, code, err)

    normalized_evidence = []
    for item in evidence:
        entry: dict[str, Any] = {"basis": item["basis"]}
        if basis == "speech":
            entry["quote"] = item["quote"].strip()
        else:
            entry["label"] = " ".join(item["label"].split())
            entry["start"] = float(item["start"])
            entry["end"] = float(item["end"])
            eid = item.get("event_id")
            if isinstance(eid, str) and eid.strip():
                entry["event_id"] = eid.strip()[:128]
        normalized_evidence.append(entry)

    return ({
        "id": option_id,
        "text": text,
        "language": language,
        "rationale": rationale,
        "evidence_basis": basis,
        "evidence": normalized_evidence,
        "display": {"start": interval[0], "end": interval[1]},
        "grounded": True,
    }, None)


def dedupe_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate validated options by normalized text; keep first.

    Never invents replacements.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for opt in options:
        key = " ".join(normalize_tokens(opt.get("text", "")))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(opt)
    return out


# ---------------------------------------------------------------------------
# Deterministic selection
# ---------------------------------------------------------------------------

SELECTION_HEURISTIC = (
    "evidence_count_desc_text_len_asc_id_asc"
)
"""Documented selection heuristic (deterministic, not a virality model)."""


def select_opening_option(
    options: list[dict[str, Any]], *, user_preference: Optional[dict] = None
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    """Deterministically select one validated option.

    Heuristic: most grounded evidence citations first, then punchier
    (shorter) text, then stable local id. An explicit user language
    preference breaks ties before length. Confidence is a qualitative
    label about evidence support — never a virality probability.
    """
    prefs = user_preference or {}
    pref_lang = prefs.get("language") if isinstance(prefs, dict) else None

    def _key(opt: dict[str, Any]) -> tuple:
        lang_match = (
            0 if (isinstance(pref_lang, str) and pref_lang and
                  opt.get("language") == pref_lang) else 1
        )
        return (-len(opt.get("evidence") or []), lang_match,
                len(opt.get("text") or ""), opt.get("id") or "")

    ranked = sorted(options, key=_key)
    if not ranked:
        return None, {
            "heuristic": SELECTION_HEURISTIC,
            "confidence": "none",
            "reason": "no validated options to select from",
        }
    chosen = ranked[0]
    evidence_count = len(chosen.get("evidence") or [])
    return chosen, {
        "heuristic": SELECTION_HEURISTIC,
        "confidence": "supported" if evidence_count >= 2 else "single_evidence",
        "reason": (
            f"selected '{chosen['id']}' with {evidence_count} grounded "
            f"evidence citation(s)"),
    }


# ---------------------------------------------------------------------------
# Invocation budget (one monotonic deadline, bounded request count)
# ---------------------------------------------------------------------------

class OpeningBudget:
    """Separate configurable opening invocation budget.

    One monotonic deadline and one bounded request count across all
    clips and fallback attempts. Every attempt is charged before it
    starts; per-attempt timeout is capped by the remaining time; the
    runner stops before exhaustion. Sub-millisecond remainders still
    produce positive timeouts (never rounded to zero).
    """

    def __init__(self, *, deadline_s: float, max_requests: int,
                 request_timeout_s: float,
                 clock: Callable[[], float] = time.monotonic):
        if not _is_finite_number(deadline_s) or deadline_s <= 0:
            raise ValueError("deadline_s must be positive finite")
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) \
                or max_requests <= 0:
            raise ValueError("max_requests must be a positive int")
        if not _is_finite_number(request_timeout_s) or request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive finite")
        self.deadline_s = float(deadline_s)
        self.max_requests = max_requests
        self.request_timeout_s = float(request_timeout_s)
        self._clock = clock
        self._deadline = clock() + float(deadline_s)
        self.requests_used = 0

    def remaining(self) -> float:
        """Seconds left on the monotonic deadline (never negative)."""
        return max(0.0, self._deadline - self._clock())

    def try_charge(self) -> tuple[bool, float]:
        """Charge one attempt. Returns (ok, timeout_s).

        Fails when the request count is exhausted or the deadline has
        passed (stops before exhaustion). The per-attempt timeout is
        capped by the remaining time; a positive remainder always
        yields a positive timeout (never rounded down to zero).
        """
        if self.requests_used >= self.max_requests:
            return False, 0.0
        remaining = self.remaining()
        if remaining <= 0:
            return False, 0.0
        self.requests_used += 1
        timeout = min(self.request_timeout_s, remaining)
        rounded = round(timeout, 3)
        # A positive remainder must never become a zero timeout.
        return True, rounded if rounded > 0 else timeout

    def snapshot(self) -> dict[str, Any]:
        return {
            "deadline_s": self.deadline_s,
            "max_requests": self.max_requests,
            "request_timeout_s": self.request_timeout_s,
            "requests_used": self.requests_used,
            "remaining_s": round(self.remaining(), 3),
        }


def opening_budget_from_env(
    clock: Callable[[], float] = time.monotonic,
) -> OpeningBudget:
    """Build the budget from environment (config version pinned)."""
    def _float(name: str, default: float) -> float:
        try:
            value = float(os.environ.get(name, "") or default)
        except (TypeError, ValueError):
            return default
        return value if _is_finite_number(value) and value > 0 else default

    def _int(name: str, default: int) -> int:
        try:
            value = int(os.environ.get(name, "") or default)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    return OpeningBudget(
        deadline_s=_float("CLIPPYME_OPENING_DEADLINE_S", 60.0),
        max_requests=_int("CLIPPYME_OPENING_MAX_REQUESTS", 8),
        request_timeout_s=_float("CLIPPYME_OPENING_REQUEST_TIMEOUT_S", 20.0),
        clock=clock,
    )


_RUN_BUDGET: Optional[OpeningBudget] = None


def get_run_budget() -> OpeningBudget:
    """Process-wide run budget: one monotonic deadline across all clips.

    The budget is (re)created lazily; every job entry MUST call
    reset_run_budget() first so a new job never inherits an exhausted
    budget from a previous job in this process.
    """
    global _RUN_BUDGET
    if _RUN_BUDGET is None:
        _RUN_BUDGET = opening_budget_from_env()
    return _RUN_BUDGET


def reset_run_budget() -> None:
    """Clear the process-wide budget so the next job starts fresh.

    Called at every real job entry (direct CLI and queued
    orchestrator); tests use it to simulate job boundaries.
    """
    global _RUN_BUDGET
    _RUN_BUDGET = None


# ---------------------------------------------------------------------------
# Stage runner (one request per clip, isolated failures, exactly-once close)
# ---------------------------------------------------------------------------

def _close_provider(provider: Provider, closed: set[int]) -> None:
    if provider is None or id(provider) in closed:
        return
    closed.add(id(provider))
    close = getattr(provider, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def run_opening_stage(
    *,
    packet: dict[str, Any],
    provider: Provider = None,
    provider_factory: Optional[Callable[[], Provider]] = None,
    fallback_provider: Provider = None,
    fallback_factory: Optional[Callable[[], Provider]] = None,
    budget: Optional[OpeningBudget] = None,
    user_preference: Optional[dict] = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one opening-choice stage for a single clip.

    Ownership contract: a provider passed directly is caller-owned —
    the stage NEVER closes it, so one job's shared provider survives
    all its clips. A provider the stage builds itself via a factory is
    closed exactly once on all exits; a failed factory build is
    isolated as a setup failure. Provider/setup/prompt failures are
    isolated to this clip (recorded, never raised).

    Returns a run record: packet_id, requested count, provider/
    fallback call counts, per-attempt records (every attempt charged),
    validated options, rejections with exact reasons, selected option
    (or None), no-selection reason, and per-attempt failures.
    """
    budget = budget or get_run_budget()
    closed: set[int] = set()
    own_primary: Provider = None
    own_fallback: Provider = None
    result: dict[str, Any] = {
        "packet_id": packet.get("packet_id"),
        "requested": 0,
        "provider_calls": 0,
        "fallback_calls": 0,
        "attempts": [],
        "validated": [],
        "rejected": [],
        "failures": [],
        "selected": None,
        "selection": None,
        "no_selection_reason": None,
        "notes": [],
    }

    def _resolve(name: str, direct: Provider,
                 factory: Optional[Callable[[], Provider]]) -> Optional[Provider]:
        """Resolve the effective provider for one slot.

        A directly-passed provider is caller-owned (never closed here).
        A factory-built provider is recorded in own_primary/own_fallback
        and closed exactly once by the stage. A failed factory build is
        isolated as a setup failure.
        """
        nonlocal own_primary, own_fallback
        if direct is not None:
            return direct
        if factory is None:
            return None
        try:
            built = factory()
        except Exception as exc:
            result["failures"].append(
                {"stage": f"{name}_setup",
                 "error": f"provider factory failed: {str(exc)[:160]}"})
            return None
        if name == "primary":
            own_primary = built
        else:
            own_fallback = built
        return built

    try:
        primary = _resolve("primary", provider, provider_factory)
        fallback = _resolve("fallback", fallback_provider, fallback_factory)
        try:
            prompt = build_opening_prompt(packet)
        except Exception as exc:
            result["failures"].append(
                {"stage": "prompt", "error": str(exc)[:200]})
            result["no_selection_reason"] = "prompt_build_failed"
            return result

        def _configured_fallback_models(prov: Provider) -> list[str]:
            """Fallback models a provider exposes for the stage to
            drive. Only the production adapter honors a per-call model
            override, so only it contributes a fallback chain here —
            generic providers are never blind-called with a model they
            would ignore."""
            if not isinstance(prov, GeminiOpeningProvider):
                return []
            try:
                config = prov.describe()
            except Exception:
                return []
            fbs = (config or {}).get("fallback_models") or []
            seen: list[str] = []
            for fb in fbs:
                if isinstance(fb, str) and fb.strip() and fb not in seen:
                    seen.append(fb.strip())
            return seen

        # Attempt plan: primary, then the primary's configured fallback
        # models, then the fallback provider and its fallback models.
        # Every attempt is charged against the single shared budget;
        # the adapter itself never retries (no hidden retries).
        plan: list[tuple[str, Provider, Optional[str]]] = []
        if primary is not None:
            plan.append(("primary", primary, None))
            for fb_model in _configured_fallback_models(primary):
                plan.append((f"fallback:{fb_model}", primary, fb_model))
        if fallback is not None:
            plan.append(("fallback", fallback, None))
            for fb_model in _configured_fallback_models(fallback):
                plan.append((f"fallback:{fb_model}", fallback, fb_model))
        if not plan:
            result["no_selection_reason"] = "no_provider_configured"
            return result

        raw_options: list[dict[str, Any]] = []
        budget_exhausted = False
        for name, prov, model in plan:
            ok, timeout_s = budget.try_charge()
            if not ok:
                budget_exhausted = True
                result["failures"].append(
                    {"stage": name, "error": "budget_exhausted"})
                result["attempts"].append(
                    {"stage": name, "timeout_s": 0.0, "ok": False,
                     "error": "budget_exhausted"})
                break
            request = {
                "prompt": prompt,
                "packet": packet,
                "timeout_s": timeout_s,
                "max_options": MAX_OPENING_OPTIONS,
                "retries_disabled": True,
                # The adapter recomputes the remaining budget after
                # client/setup work and never exceeds it.
                "remaining_s": budget.remaining,
            }
            if model is not None:
                request["model"] = model
            try:
                # No SDK-internal retries by contract: a single call per
                # attempt, every attempt charged above. The adapter
                # itself performs no retry loop.
                options = prov.request_options(request)
                call_ok = True
                call_error = None
            except OpeningBudgetExhausted as exc:
                # The adapter stopped before the SDK call: explicit
                # budget exhaustion, not a provider failure.
                call_ok = False
                call_error = f"budget_exhausted: {str(exc)[:120]}"
                budget_exhausted = True
                options = None
            except Exception as exc:
                call_ok = False
                call_error = str(exc)[:200]
                options = None
            if name == "primary":
                result["provider_calls"] += 1
            else:
                result["fallback_calls"] += 1
            result["attempts"].append(
                {"stage": name, "timeout_s": timeout_s, "ok": call_ok,
                 **({"model": model} if model else {}),
                 **({"error": call_error} if call_error else {})})
            if not call_ok:
                result["failures"].append(
                    {"stage": name, "error": call_error})
                continue
            result["requested"] += 1
            if not isinstance(options, list):
                result["failures"].append(
                    {"stage": name, "error": "provider did not return a list"})
                continue
            if len(options) > MAX_OPENING_OPTIONS:
                result["notes"].append(
                    f"{name}: truncated {len(options)} options to "
                    f"{MAX_OPENING_OPTIONS}")
                options = options[:MAX_OPENING_OPTIONS]
            raw_options.extend(options)
            break  # one request per clip; fallback only on failure

        for option in raw_options:
            normalized, rejection = validate_opening_option(
                option, packet=packet)
            if rejection is not None:
                result["rejected"].append(rejection)
            else:
                result["validated"].append(normalized)

        unique = dedupe_options(result["validated"])
        if len(unique) != len(result["validated"]):
            result["notes"].append(
                f"deduplicated {len(result['validated'])} to {len(unique)}")
        result["validated"] = unique

        selected, selection = select_opening_option(
            unique, user_preference=user_preference)
        result["selected"] = selected
        result["selection"] = selection
        if selected is None:
            if budget_exhausted:
                result["no_selection_reason"] = "budget_exhausted"
            elif result["rejected"]:
                result["no_selection_reason"] = "no_valid_options"
            elif result["failures"]:
                result["no_selection_reason"] = "provider_failed"
            else:
                result["no_selection_reason"] = "no_options_returned"
        return result
    finally:
        # Close ONLY stage-built providers, exactly once. Caller-owned
        # providers are never closed here.
        if own_primary is not None:
            _close_provider(own_primary, closed)
        if own_fallback is not None:
            _close_provider(own_fallback, closed)


# ---------------------------------------------------------------------------
# Versioned opening funnel (exact reconciliation)
# ---------------------------------------------------------------------------

def new_opening_funnel() -> dict[str, Any]:
    return {
        "schema_version": OPENING_PLAN_SCHEMA_VERSION,
        "clips": {},
        "totals": {
            "clips": 0,
            "requested": 0,
            "validated": 0,
            "rejected": 0,
            "selected": 0,
            "no_selection": 0,
            "user_authored": 0,
        },
    }


def record_opening_result(
    funnel: dict[str, Any], clip_key: str, run_result: dict[str, Any],
    *, user_authored: bool = False,
) -> None:
    """Record one clip's run into the funnel (idempotent per clip_key)."""
    clips = funnel.setdefault("clips", {})
    totals = funnel.setdefault("totals", {})
    if clip_key in clips:
        # Remove the old record's contribution before re-recording.
        old = clips[clip_key]
        for key in ("requested", "validated"):
            totals[key] = totals.get(key, 0) - old.get(key, 0)
        totals["rejected"] = totals.get("rejected", 0) - len(
            old.get("rejected") or [])
        totals["clips"] = totals.get("clips", 0) - 1
        totals["selected"] = totals.get("selected", 0) - (
            1 if old.get("selected_id") else 0)
        totals["no_selection"] = totals.get("no_selection", 0) - (
            0 if old.get("selected_id") or old.get("user_authored") else 1)
        totals["user_authored"] = totals.get("user_authored", 0) - (
            1 if old.get("user_authored") else 0)
    selected = run_result.get("selected")
    record = {
        "packet_id": run_result.get("packet_id"),
        "requested": int(run_result.get("requested") or 0),
        "validated": len(run_result.get("validated") or []),
        "rejected": [
            {"option_id": r.get("option_id"), "code": r.get("code"),
             "reason": r.get("reason")}
            for r in run_result.get("rejected") or []
        ],
        "selected_id": selected.get("id") if selected else None,
        "selection": run_result.get("selection"),
        "no_selection_reason": run_result.get("no_selection_reason"),
        "user_authored": bool(user_authored),
        "failures": run_result.get("failures") or [],
        "notes": run_result.get("notes") or [],
    }
    clips[clip_key] = record
    totals["clips"] = totals.get("clips", 0) + 1
    totals["requested"] = totals.get("requested", 0) + record["requested"]
    totals["validated"] = totals.get("validated", 0) + record["validated"]
    totals["rejected"] = totals.get("rejected", 0) + len(record["rejected"])
    if user_authored:
        totals["user_authored"] = totals.get("user_authored", 0) + 1
    elif record["selected_id"]:
        totals["selected"] = totals.get("selected", 0) + 1
    else:
        totals["no_selection"] = totals.get("no_selection", 0) + 1


def funnel_reconciles(funnel: dict[str, Any]) -> tuple[bool, str]:
    """Check the funnel counts reconcile exactly."""
    clips = funnel.get("clips") or {}
    totals = funnel.get("totals") or {}
    expected = {
        "clips": len(clips),
        "requested": 0,
        "validated": 0,
        "rejected": 0,
        "selected": 0,
        "no_selection": 0,
        "user_authored": 0,
    }
    for record in clips.values():
        expected["requested"] += record.get("requested", 0)
        expected["validated"] += record.get("validated", 0)
        expected["rejected"] += len(record.get("rejected") or [])
        if record.get("user_authored"):
            expected["user_authored"] += 1
        elif record.get("selected_id"):
            expected["selected"] += 1
        else:
            expected["no_selection"] += 1
    for key, want in expected.items():
        got = totals.get(key, 0)
        if got != want:
            return False, f"funnel total {key}: recorded {got}, actual {want}"
    accounted = (expected["selected"] + expected["no_selection"]
                 + expected["user_authored"])
    if accounted != expected["clips"]:
        return False, (
            f"funnel clips {expected['clips']} != selected "
            f"{expected['selected']} + no_selection {expected['no_selection']} "
            f"+ user_authored {expected['user_authored']}")
    return True, "funnel reconciles"


# ---------------------------------------------------------------------------
# Resume (full identity + material validation)
# ---------------------------------------------------------------------------

# No-selection reasons that are transient: a saved plan with empty
# validated options and one of these reasons must be re-analyzed, never
# resumed as a successful plan.
_TRANSIENT_NO_SELECTION_REASONS = frozenset({
    "no_provider_configured",
    "provider_failed",
    "budget_exhausted",
    "prompt_build_failed",
    "packet_build_failed",
})


def _validate_saved_options(
    saved: list[dict[str, Any]], packet: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-validate saved options against the current packet."""
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for option in saved:
        normalized, rejection = validate_opening_option(option, packet=packet)
        if rejection is not None:
            rejected.append(rejection)
        else:
            valid.append(normalized)
    return valid, rejected


def opening_plan_resumable(
    saved_plan: dict[str, Any], packet: dict[str, Any]
) -> tuple[bool, str]:
    """True only when the full current identity matches AND the saved
    validated options still pass material validation against the
    current packet. A changed final interval/words/source/evidence/
    config forces re-analysis. A transient no-provider/setup/budget
    failure with empty options is never resumable as a successful
    plan. The saved selection must be a member of the saved validated
    options. Fully valid resumed plans need no client.
    """
    if not isinstance(saved_plan, dict):
        return False, "no saved plan"
    saved_identity = saved_plan.get("packet_identity")
    current_identity = (packet.get("identity") or {})
    if not isinstance(saved_identity, dict):
        return False, "saved plan has no packet identity"
    for key, value in current_identity.items():
        if saved_identity.get(key) != value:
            return False, f"identity changed: {key}"
    if saved_plan.get("schema_version") != OPENING_PLAN_SCHEMA_VERSION:
        return False, "schema version changed"
    saved_validated = saved_plan.get("validated_options") or []
    # Transient failures with empty options are never resumable as a
    # successful plan: the next attempt must re-analyze.
    saved_selected = saved_plan.get("selected")
    if (not saved_validated and not saved_plan.get("user_authored")
            and not (isinstance(saved_selected, dict)
                    and saved_selected.get("id"))):
        reason = saved_plan.get("no_selection_reason")
        if reason in _TRANSIENT_NO_SELECTION_REASONS:
            return False, (
                f"transient no-selection not resumable: {reason}")
    # The saved selection must be a member of the saved validated
    # options — a forged/obsolete selection never resumes.
    if isinstance(saved_selected, dict) and saved_selected.get("id"):
        valid_ids = {v.get("id") for v in saved_validated
                     if isinstance(v, dict)}
        if saved_selected["id"] not in valid_ids:
            return False, "saved selection not a member of validated options"
    valid, rejected = _validate_saved_options(saved_validated, packet)
    if rejected:
        return False, (
            f"{len(rejected)} saved option(s) fail material validation")
    if len(valid) != len(saved_validated):
        return False, "saved options changed under re-validation"
    return True, "full identity and material validation match"


# ---------------------------------------------------------------------------
# Clip integration (called after the last accepted boundary repair)
# ---------------------------------------------------------------------------

def _window_transcript_words(
    transcript: dict[str, Any], start: float, end: float
) -> list[dict[str, Any]]:
    """Window transcript words to [start, end), absolute times preserved.

    Task 04 contract: the word-level speaker label wins (integer 0 is
    valid and preserved); a word without a label stays unknown (no
    'speaker' key); segment summaries are NEVER copied into word
    labels — they stay separate.
    """
    out: list[dict[str, Any]] = []
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words") or []:
            if not isinstance(w, dict):
                continue
            ws, we = w.get("start"), w.get("end")
            if not _is_finite_number(ws) or not _is_finite_number(we):
                continue
            if we <= start or ws >= end:
                continue
            text = (w.get("word") or "").strip()
            if not text:
                continue
            entry: dict[str, Any] = {
                "word": text, "start": float(ws), "end": float(we)}
            speaker = _canonical_word_speaker(w.get("speaker"))
            if speaker is not None:
                entry["speaker"] = speaker
            # Unknown words stay unknown: no 'speaker' key, and the
            # segment summary is never inherited.
            out.append(entry)
    out.sort(key=lambda e: (e["start"], e["end"]))
    return out


_FRAME_AT_RE = re.compile(
    r"^(?:frame|event|vf)-at-([0-9]+(?:\.[0-9]+)?)$")


def _resolve_reference_time(
    ref_id: str, frame_times: dict[str, Any],
) -> Optional[float]:
    """Resolve a frame/event reference to an absolute source time.

    Prefers an explicit ``clips_data["av_frame_times"]`` binding
    (production provenance); falls back to the ``frame-at-<t>`` /
    ``event-at-<t>`` synthetic convention. Returns None when the
    reference cannot be time-resolved — its scope is then unknown.
    """
    if isinstance(frame_times, dict):
        t = frame_times.get(ref_id)
        if _is_finite_number(t):
            return float(t)
    m = _FRAME_AT_RE.match(str(ref_id or "").strip())
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return None
    return None


def _reference_scope(
    ref_ids: list[str], frame_times: dict[str, Any],
    start: float, end: float,
) -> str:
    """Temporal scope of referenced frames/events vs the final clip.

    Returns "inside" when at least one referenced frame/event is
    inside [start, end); "outside" when at least one reference
    resolves and none is inside; "unknown" when no references are
    listed or none resolve (scope cannot be established — the
    established candidate-interval behavior is kept).
    """
    resolved = [t for t in
                (_resolve_reference_time(rid, frame_times) for rid in ref_ids)
                if t is not None]
    if not ref_ids or not resolved:
        return "unknown"
    if any(start <= t < end for t in resolved):
        return "inside"
    return "outside"


def _opening_av_evidence_for_clip(
    clip: dict[str, Any], clips_data: dict[str, Any],
    start: float, end: float,
) -> list[dict[str, Any]]:
    """Production AV evidence for one clip: the accepted Task 11
    candidate evidence.

    Reads ``clips_data["candidate_store"][clip["candidate_id"]]`` (the
    accepted Task 11 candidate record) and converts its visual/audio
    evidence to packet items {kind, label, start, end, event_id,
    source_fingerprint} in absolute source time, intersected with the
    final clip interval.

    Scope rule: when the candidate lists frame/event references
    (``visual_frame_ids`` / ``audio_event_ids``) whose times resolve,
    the description is usable only if at least one referenced
    frame/event is inside the final clip. If every resolved reference
    is outside, the cut excludes the relevant evidence — the bridge
    declines ([]) rather than retiming a whole-candidate summary onto
    a cut that removed its payoff. Unresolvable references keep the
    established candidate-interval behavior. Falls back to the legacy
    ``clip["opening_av_evidence"]`` test alias only when the Task 11
    path yields nothing and did not explicitly decline. Returns []
    when neither has usable evidence.
    """
    items: list[dict[str, Any]] = []
    declined = False
    store = (clips_data or {}).get("candidate_store")
    cid = clip.get("candidate_id")
    rec = (store.get(cid) if isinstance(store, dict)
           and isinstance(cid, str) else None)
    frame_times = (clips_data or {}).get("av_frame_times") or {}
    if isinstance(rec, dict):
        basis = str(rec.get("evidence_basis") or "").strip().lower()
        rs, re_ = rec.get("start_s"), rec.get("end_s")
        provenance = rec.get("source_fingerprint")
        provenance = (provenance.strip()[:256]
                      if isinstance(provenance, str) and provenance.strip()
                      else None)
        if _is_finite_number(rs) and _is_finite_number(re_) and re_ > rs:
            cs, ce = max(float(rs), start), min(float(re_), end)
            if ce > cs:
                if basis in ("visual", "multimodal"):
                    scope = _reference_scope(
                        [i for i in (rec.get("visual_frame_ids") or [])
                         if isinstance(i, str)],
                        frame_times, start, end)
                    if scope == "outside":
                        declined = True
                    else:
                        label = rec.get("visual_description")
                        if isinstance(label, str) and label.strip():
                            entry: dict[str, Any] = {
                                "kind": "visual", "label": label.strip(),
                                "start": cs, "end": ce, "event_id": cid,
                            }
                            if provenance:
                                entry["source_fingerprint"] = provenance
                            items.append(entry)
                if basis in ("audio_event", "multimodal"):
                    scope = _reference_scope(
                        [i for i in (rec.get("audio_event_ids") or [])
                         if isinstance(i, str)],
                        frame_times, start, end)
                    if scope == "outside":
                        declined = True
                    else:
                        label = rec.get("evidence_text")
                        if isinstance(label, str) and label.strip():
                            entry = {
                                "kind": "audio", "label": label.strip(),
                                "start": cs, "end": ce, "event_id": cid,
                            }
                            if provenance:
                                entry["source_fingerprint"] = provenance
                            items.append(entry)
    if declined:
        return []
    if not items:
        legacy = clip.get("opening_av_evidence")
        if isinstance(legacy, list):
            items = legacy
    return items


def _provider_describe(provider: Provider) -> Optional[dict[str, Any]]:
    describe = getattr(provider, "describe", None)
    if not callable(describe):
        return None
    try:
        config = describe()
    except Exception:
        return None
    return config if isinstance(config, dict) else None


def _packet_provider_config(
    provider: Provider,
    provider_factory: Optional[Callable[[], Provider]],
) -> Optional[dict[str, Any]]:
    """Provider model/fallback config for the packet identity."""
    config = _provider_describe(provider)
    if config is None and provider_factory is not None:
        config = _provider_describe(provider_factory)
    return config


def current_opening_packet_or_none(
    clip: dict[str, Any], clips_data: dict[str, Any],
    start: float, end: float,
    current_fingerprint: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Recompute the current opening packet for emission checks.

    Uses the CURRENT clip bounds/words/AV evidence/language/
    preferences/config. The source fingerprint is rechecked against
    the authoritative current input (``current_fingerprint``, recomputed
    by the caller from the actual source file) rather than treating
    the saved fingerprint as current by definition: a mismatch — or
    an unverifiable source — fails reconstruction, and the emission
    boundary then emits no automatic overlay. Returns None (never
    raises) when the packet cannot be rebuilt.
    """
    try:
        plan = clip.get("opening_plan") or {}
        # Fail closed: without an authoritative current fingerprint the
        # current source identity cannot be verified. The saved
        # fingerprint is never trusted on its own — a missing or failed
        # current identity suppresses automatic emission.
        if (not isinstance(current_fingerprint, str)
                or not current_fingerprint):
            return None
        saved = clip.get("opening_source_fingerprint")
        # Authoritative recheck: the saved fingerprint must match the
        # fingerprint recomputed from the current source file.
        if (not isinstance(saved, str) or not saved
                or saved != current_fingerprint):
            return None
        fingerprint = current_fingerprint
        if not _is_finite_number(start) or not _is_finite_number(end):
            return None
        if not (end > start >= 0):
            return None
        transcript = clips_data.get("transcript") or {}
        words = _window_transcript_words(transcript, float(start), float(end))
        avidence = _opening_av_evidence_for_clip(
            clip, clips_data, float(start), float(end))
        return build_opening_packet(
            source_fingerprint=fingerprint,
            final_start=float(start),
            final_end=float(end),
            words=words,
            candidate_id=clip.get("candidate_id"),
            moment_id=clip.get("moment_id"),
            av_evidence=avidence or None,
            language=clip.get("language") or clips_data.get("language"),
            preferences=user_preference_from_env(),
            provider_config=plan.get("provider_config"),
        )
    except Exception:
        return None


def user_authored_override(clip: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Explicit user-authored hook override (clip['hook']).

    Wins over any provider output and is marked user-authored; no
    provider rewrites it.
    """
    text = (clip.get("hook") or "")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > MAX_TEXT_CHARS:
        return None
    if _CONTROL_RE.search(text):
        return None
    return {"text": text, "author": "user"}


def plan_clip_opening(
    *,
    clip: dict[str, Any],
    clips_data: dict[str, Any],
    input_video: str,
    index: int,
    budget: Optional[OpeningBudget] = None,
    provider: Provider = None,
    provider_factory: Optional[Callable[[], Provider]] = None,
    fallback_provider: Provider = None,
    fallback_factory: Optional[Callable[[], Provider]] = None,
    user_preference: Optional[dict] = None,
    route: str = "direct",
) -> dict[str, Any]:
    """Build (or resume) the grounded opening plan for one clip.

    Called with the FINAL cutter bounds (clip['start']/clip['end']
    after the last accepted boundary repair). Records the versioned
    plan on ``clip['opening_plan']`` and the funnel in
    ``clips_data['opening_funnel']``. Never raises for provider-side
    problems; packet-build problems raise (caller treats them as
    clip-local warnings) AFTER clearing a stale automatic plan so no
    obsolete hook can reach emission.

    The explicit user-authored override is handled FIRST — before
    fingerprinting and packet construction — so a packet failure can
    never block the authorized override.
    """
    clip_key = f"clip-{index}"
    funnel = clips_data.get("opening_funnel")
    if not isinstance(funnel, dict):
        funnel = new_opening_funnel()
        clips_data["opening_funnel"] = funnel

    # Explicit user-authored override: wins, marked, no provider call,
    # no packet needed — handled FIRST, before bounds validation,
    # fingerprinting, and packet construction, so no packet failure can
    # ever block the authorized override, and without depending on
    # provider readiness.
    override = user_authored_override(clip)
    if override is not None:
        plan = {
            "schema_version": OPENING_PLAN_SCHEMA_VERSION,
            "user_authored": True,
            "text": override["text"],
            "route": route,
        }
        clip["opening_plan"] = plan
        record_opening_result(
            funnel, clip_key,
            {"packet_id": None, "requested": 0,
             "validated": [], "rejected": [], "selected": None,
             "selection": None,
             "no_selection_reason": "user_authored_override",
             "failures": [], "notes": []},
            user_authored=True,
        )
        return plan

    try:
        start = clip.get("start")
        end = clip.get("end")
        if not _is_finite_number(start) or not _is_finite_number(end):
            raise PacketBuildError("clip start/end must be finite numeric")
        if not (end > start >= 0):
            raise PacketBuildError(
                f"invalid clip interval [{start}, {end})")

        from clippyme.pipeline.framing_plan import (
            source_fingerprint as _source_fingerprint,
        )
        try:
            fingerprint = _source_fingerprint(input_video)
        except Exception as exc:
            raise PacketBuildError(f"cannot fingerprint source: {exc}")

        # Additive: the emission boundary recomputes the current packet
        # from this stored fingerprint plus the clip's current content.
        # The authoritative ORIGINAL source path is threaded through so
        # emission re-fingerprints the original asset (never the cut
        # slice) and fails closed when it is missing or unreadable.
        clip["opening_source_fingerprint"] = fingerprint
        clip["opening_source_path"] = input_video

        provider_config = _packet_provider_config(provider, provider_factory)

        transcript = clips_data.get("transcript") or {}
        words = _window_transcript_words(transcript, float(start), float(end))
        packet = build_opening_packet(
            source_fingerprint=fingerprint,
            final_start=float(start),
            final_end=float(end),
            words=words,
            candidate_id=clip.get("candidate_id"),
            moment_id=clip.get("moment_id"),
            av_evidence=_opening_av_evidence_for_clip(
                clip, clips_data, float(start), float(end)) or None,
            language=clip.get("language") or clips_data.get("language"),
            preferences=user_preference or {},
            provider_config=provider_config,
        )
    except PacketBuildError as exc:
        # Failed re-analysis invalidates a stale automatic plan: clear
        # it so no obsolete hook can reach emission. User-authored
        # overrides were handled above and are never cleared here.
        prior = clip.get("opening_plan")
        cleared = (isinstance(prior, dict) and not prior.get("user_authored")
                   and "opening_plan" in clip)
        if cleared:
            del clip["opening_plan"]
        record_opening_result(
            funnel, clip_key,
            {"packet_id": None, "requested": 0,
             "validated": [], "rejected": [], "selected": None,
             "selection": None,
             "no_selection_reason": "packet_build_failed",
             "failures": [{"stage": "packet", "error": str(exc)[:200]}],
             "notes": (["stale automatic plan cleared"]
                       if cleared else [])},
        )
        raise

    # Resume: reuse only on full identity + material validation match.
    # Fully valid resumed plans need no client.
    saved = clip.get("opening_plan")
    resumable, resume_reason = opening_plan_resumable(saved or {}, packet)
    if resumable:
        saved_validated = saved.get("validated_options") or []
        valid, _ = _validate_saved_options(saved_validated, packet)
        unique = dedupe_options(valid)
        selected, selection = select_opening_option(
            unique, user_preference=user_preference)
        plan = {
            "schema_version": OPENING_PLAN_SCHEMA_VERSION,
            "packet_id": packet.get("packet_id"),
            "packet_identity": packet.get("identity"),
            "provider_config": provider_config,
            "resumed": True,
            "resume_reason": resume_reason,
            "validated_options": valid,
            "selected": selected,
            "selection": selection,
            "no_selection_reason": (
                None if selected else saved.get("no_selection_reason")),
            "route": route,
        }
        clip["opening_plan"] = plan
        record_opening_result(
            funnel, clip_key,
            {"packet_id": packet.get("packet_id"), "requested": 0,
             "validated": valid, "rejected": [],
             "selected": selected, "selection": selection,
             "no_selection_reason": plan["no_selection_reason"],
             "failures": [], "notes": ["resumed without provider"]},
        )
        return plan

    run = run_opening_stage(
        packet=packet,
        provider=provider,
        provider_factory=provider_factory,
        fallback_provider=fallback_provider,
        fallback_factory=fallback_factory,
        budget=budget or get_run_budget(),
        user_preference=user_preference,
    )
    plan = {
        "schema_version": OPENING_PLAN_SCHEMA_VERSION,
        "packet_id": packet.get("packet_id"),
        "packet_identity": packet.get("identity"),
        "provider_config": provider_config,
        "resumed": False,
        "resume_reason": resume_reason,
        "validated_options": run["validated"],
        "rejected_options": run["rejected"],
        "selected": run["selected"],
        "selection": run["selection"],
        "no_selection_reason": run["no_selection_reason"],
        "provider_calls": run["provider_calls"],
        "fallback_calls": run["fallback_calls"],
        "failures": run["failures"],
        "notes": run["notes"],
        "route": route,
    }
    clip["opening_plan"] = plan
    record_opening_result(funnel, clip_key, run)
    return plan
