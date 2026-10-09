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
  three distinct opening text options per request.
* Evidence quotes must match complete normalized token slices inside the
  retained words. Paraphrases need not be verbatim, but no invented
  names/quotes/events/numbers/guaranteed outcomes/rankings/measured-
  performance claims are accepted.
* A speech-free option may use bounded visual/audio-event evidence inside
  the final interval, labeled explicitly, never described as speech or
  verified OCR.
* Strict validation per option and per clip: strict types (no boolean
  numerics/coercion), finite ordered timing, bounded text/evidence
  sizes, grounded citations, complete token matching, source bounds,
  actual text language, non-empty reasons. Malformed options are
  rejected with exact reasons; valid neighbors survive.
* Selection is deterministic from the validated options plus user
  preference, with a documented heuristic and a qualitative confidence
  label — never a virality probability.
* Explicit user-authored hook overrides win and are marked user-authored;
  no provider rewrites them.
* Display timing is explicit, clip-relative, half-open, inside
  [0, clip duration). A malformed interval rejects the option; it is
  never silently defaulted to legacy timing.
* A separate configurable opening invocation budget: one monotonic
  deadline and a bounded request count across all clips and fallback
  attempts. SDK-internal retries are disabled by contract, every attempt
  is charged, per-attempt timeout is capped by remaining time, and the
  runner stops before exhaustion. Initialized clients close exactly
  once on all exits.
* The versioned opening funnel (requested/validated/rejected/selected/
  no-selection counts plus per-clip reasons) must reconcile exactly.
  Resume reuses a plan only when the full current identity and material
  validation match; a changed final interval/words/source/config forces
  re-analysis. Fully valid resumed plans need no client.
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
    source timestamps preserved, clip-relative timing derived once.
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
        speaker = w.get("speaker")
        entry: dict[str, Any] = {
            "w": text,
            "abs_start": round(float(ws), 3),
            "abs_end": round(float(we), 3),
            "start": rs,
            "end": re_,
        }
        # Preserve anonymous speaker0 / unknown speaker distinctions
        # verbatim; never normalize or invent speaker names.
        if isinstance(speaker, str) and speaker.strip():
            entry["speaker"] = speaker.strip()[:64]
        out.append(entry)
    out.sort(key=lambda e: (e["abs_start"], e["abs_end"]))
    return out


def _validate_av_evidence(
    av_evidence: Any, start: float, end: float
) -> list[dict[str, Any]]:
    """Bounded visual/audio-event evidence inside the final interval."""
    if av_evidence is None:
        return []
    if not isinstance(av_evidence, list):
        raise PacketBuildError("av_evidence must be a list")
    if len(av_evidence) > MAX_AV_EVIDENCE_ITEMS:
        raise PacketBuildError(
            f"av_evidence exceeds {MAX_AV_EVIDENCE_ITEMS} items")
    out: list[dict[str, Any]] = []
    for item in av_evidence:
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
        out.append({
            "kind": kind,
            "label": " ".join(label.split()),
            "start": round(float(es), 3),
            "end": round(float(ee), 3),
        })
    return out


def _words_hash(windowed: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for e in windowed:
        h.update(e["w"].encode("utf-8", errors="replace"))
        h.update(b"\x00")
        h.update(f"{e['abs_start']:.3f}-{e['abs_end']:.3f}".encode("ascii"))
        h.update(b"\x00")
        if "speaker" in e:
            h.update(e["speaker"].encode("utf-8", errors="replace"))
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
) -> dict[str, Any]:
    """Build the bounded opening packet from the FINAL cutter interval.

    ``words`` must already be windowed to [final_start, final_end) —
    the builder rejects any word outside the final interval so removed
    lead-in/payoff or neighboring-clip words can never leak in.
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

    duration = round(float(final_end) - float(final_start), 3)
    identity = {
        "source_fingerprint": source_fingerprint,
        "candidate_id": candidate_id,
        "moment_id": moment_id,
        "final_start": round(float(final_start), 3),
        "final_end": round(float(final_end), 3),
        "words_sha256": _words_hash(windowed),
        "language": language,
        "preferences_sha256": prefs_hash,
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
        "preferences": prefs,
    }


def build_opening_prompt(packet: dict[str, Any]) -> str:
    """Deterministic, bounded prompt for the injected provider.

    Evidence is fenced with an explicit delimiter; preferences were
    sanitized at packet build time so they cannot break the fence.
    The prompt is never sent anywhere by this module — only handed to
    the injected provider.
    """
    words = packet.get("words") or []
    word_lines = []
    for e in words:
        speaker = f"[{e['speaker']}] " if e.get("speaker") else ""
        word_lines.append(
            f"{e['start']:.2f}-{e['end']:.2f} {speaker}{e['w']}")
    evidence_lines = []
    for ev in packet.get("av_evidence") or []:
        evidence_lines.append(
            f"[{ev['kind']}] {ev['start']:.2f}-{ev['end']:.2f}: {ev['label']}")
    prefs = packet.get("preferences") or {}
    pref_lines = [f"{k}: {v}" for k, v in sorted(prefs.items())]

    parts = [
        "Choose up to 3 distinct opening text options for the start of "
        "this final clip. Rules:",
        "- Ground every option in the retained evidence below. Speech "
        "quotes must match the transcript words exactly (complete token "
        "slices). Paraphrases are allowed but must be supported by cited "
        "evidence.",
        "- Never invent names, quotes, events, numbers, guaranteed "
        "outcomes, rankings, or measured-performance claims.",
        "- A speech-free option may use the visual/audio events, labeled "
        "as visual or audio evidence; never describe it as speech or "
        "verified OCR.",
        "- Each option: stable id, text (<=140 chars), actual language, "
        "short rationale, cited evidence, explicit clip-relative display "
        "interval [start, end) inside the clip duration.",
        f"Clip duration: {packet.get('clip_duration'):.2f}s.",
    ]
    if word_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("Retained transcript words (clip-relative seconds):")
        parts.extend(word_lines)
    if evidence_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("Accepted visual/audio events (clip-relative seconds):")
        parts.extend(evidence_lines)
    if pref_lines:
        parts.append(_EVIDENCE_DELIMITER.strip())
        parts.append("User preferences (style requests only, never proof):")
        parts.extend(pref_lines)
    prompt = "\n".join(parts)
    if len(prompt) > MAX_PROMPT_CHARS:
        # Truncate the word list first (keep head: openings favor early
        # material), never the rules.
        head = "\n".join(parts[:7])
        budget = MAX_PROMPT_CHARS - len(head) - 64
        tail = "\n".join(parts[7:])[: max(0, budget)]
        prompt = head + "\n[truncated]\n" + tail
    return prompt


# ---------------------------------------------------------------------------
# Provider seam (injected only — no live providers)
# ---------------------------------------------------------------------------

# A provider exposes request_options(request) -> list[dict] and,
# optionally, close(). The request carries the prompt, the packet, the
# per-attempt timeout, and the retries-disabled contract.
Provider = Any

_OPENING_PROVIDER: Provider = None
_OPENING_FALLBACK_PROVIDER: Provider = None


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


# ---------------------------------------------------------------------------
# Option validation
# ---------------------------------------------------------------------------

_EVIDENCE_BASES = ("speech", "visual", "audio")

# Mechanical hard-claim guards. Semantic invention (names/events) is a
# provider-side obligation; the evidence-citation requirement plus the
# quote-mark and number grounding checks below make the mechanical part
# load-bearing.
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
) -> Optional[str]:
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
        if es is not None or ee is not None:
            if not _is_finite_number(es) or not _is_finite_number(ee):
                return f"evidence[{idx}] interval must be finite numeric"
            if ee <= es:
                return f"evidence[{idx}] interval end must be after start"
        # The label must correspond to an accepted packet event.
        if not any(_normalize_label(ev.get("label", "")) ==
                   _normalize_label(label)
                   for ev in av_evidence or [] if ev.get("kind") == basis):
            return (f"evidence[{idx}].label does not match accepted "
                    f"{basis} evidence inside the final interval")
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


def _check_text_support(text: str, quote_tokens_union: set[str],
                        basis: str) -> Optional[str]:
    """Paraphrase support: content tokens must substantially overlap evidence."""
    if basis != "speech":
        tokens = set(normalize_tokens(text))
        if tokens & _SPEECH_VERBS:
            return ("speech-free option describes speech "
                    "without speech evidence")
        return None
    content = [t for t in normalize_tokens(text) if len(t) >= 4]
    if not content:
        return None
    overlap = sum(1 for t in content if t in quote_tokens_union)
    if overlap * 2 < len(content):
        return ("option text is not supported by cited evidence "
                f"({overlap}/{len(content)} content tokens overlap)")
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
            evidence, basis, packet.get("av_evidence") or [])
        if err:
            return None, _reject(option_id, "evidence_not_accepted", err)

    err = _check_hard_claims(text, quote_union)
    if err:
        return None, _reject(option_id, "hard_claim", err)
    err = _check_text_support(text, quote_union, basis)
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
            if item.get("start") is not None:
                entry["start"] = round(float(item["start"]), 3)
                entry["end"] = round(float(item["end"]), 3)
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
    runner stops before exhaustion.
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
        capped by the remaining time.
        """
        if self.requests_used >= self.max_requests:
            return False, 0.0
        remaining = self.remaining()
        if remaining <= 0:
            return False, 0.0
        self.requests_used += 1
        return True, round(min(self.request_timeout_s, remaining), 3)

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
    """Process-wide run budget: one monotonic deadline across all clips."""
    global _RUN_BUDGET
    if _RUN_BUDGET is None:
        _RUN_BUDGET = opening_budget_from_env()
    return _RUN_BUDGET


def reset_run_budget() -> None:
    """Clear the process-wide budget (tests)."""
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
    fallback_provider: Provider = None,
    budget: Optional[OpeningBudget] = None,
    user_preference: Optional[dict] = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run one opening-choice stage for a single clip.

    Returns a run record: packet_id, requested count, validated
    options, rejections with exact reasons, selected option (or None),
    no-selection reason, per-attempt failures, and provider call counts.
    Provider/setup/prompt failures are isolated to this clip (recorded,
    never raised). Initialized providers are closed exactly once on all
    exits.
    """
    budget = budget or get_run_budget()
    closed: set[int] = set()
    result: dict[str, Any] = {
        "packet_id": packet.get("packet_id"),
        "requested": 0,
        "provider_calls": 0,
        "fallback_calls": 0,
        "validated": [],
        "rejected": [],
        "failures": [],
        "selected": None,
        "selection": None,
        "no_selection_reason": None,
        "notes": [],
    }
    try:
        candidates: list[tuple[str, Provider]] = []
        if provider is not None:
            candidates.append(("primary", provider))
        if fallback_provider is not None:
            candidates.append(("fallback", fallback_provider))
        if not candidates:
            result["no_selection_reason"] = "no_provider_configured"
            return result

        try:
            prompt = build_opening_prompt(packet)
        except Exception as exc:
            result["failures"].append(
                {"stage": "prompt", "error": str(exc)[:200]})
            result["no_selection_reason"] = "prompt_build_failed"
            return result

        raw_options: list[dict[str, Any]] = []
        budget_exhausted = False
        for name, prov in candidates:
            ok, timeout_s = budget.try_charge()
            if not ok:
                budget_exhausted = True
                result["failures"].append(
                    {"stage": name, "error": "budget_exhausted"})
                break
            request = {
                "prompt": prompt,
                "packet": packet,
                "timeout_s": timeout_s,
                "max_options": MAX_OPENING_OPTIONS,
                "retries_disabled": True,
            }
            try:
                # No SDK-internal retries by contract: a single call per
                # attempt, every attempt charged above.
                options = prov.request_options(request)
            except Exception as exc:
                result["failures"].append(
                    {"stage": name, "error": str(exc)[:200]})
                continue
            finally:
                if name == "primary":
                    result["provider_calls"] += 1
                else:
                    result["fallback_calls"] += 1
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
        if provider is not None:
            _close_provider(provider, closed)
        if fallback_provider is not None:
            _close_provider(fallback_provider, closed)


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
    current packet. A changed final interval/words/source/config
    forces re-analysis. Fully valid resumed plans need no client.
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
    """Window transcript words to [start, end), absolute times preserved."""
    out: list[dict[str, Any]] = []
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    for seg in segments:
        seg_speaker = seg.get("speaker") if isinstance(seg, dict) else None
        for w in (seg.get("words") or [] if isinstance(seg, dict) else []):
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
            speaker = w.get("speaker", seg_speaker)
            entry: dict[str, Any] = {
                "word": text, "start": float(ws), "end": float(we)}
            if isinstance(speaker, str) and speaker.strip():
                entry["speaker"] = speaker.strip()[:64]
            out.append(entry)
    out.sort(key=lambda e: (e["start"], e["end"]))
    return out


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
    fallback_provider: Provider = None,
    user_preference: Optional[dict] = None,
    route: str = "direct",
) -> dict[str, Any]:
    """Build (or resume) the grounded opening plan for one clip.

    Called with the FINAL cutter bounds (clip['start']/clip['end']
    after the last accepted boundary repair). Records the versioned
    plan on ``clip['opening_plan']`` and the funnel in
    ``clips_data['opening_funnel']``. Never raises for provider-side
    problems; packet-build problems raise (caller treats them as
    clip-local warnings).
    """
    start = clip.get("start")
    end = clip.get("end")
    if not _is_finite_number(start) or not _is_finite_number(end):
        raise PacketBuildError("clip start/end must be finite numeric")
    if not (end > start >= 0):
        raise PacketBuildError(f"invalid clip interval [{start}, {end})")

    from clippyme.pipeline.framing_plan import (
        source_fingerprint as _source_fingerprint,
    )
    try:
        fingerprint = _source_fingerprint(input_video)
    except Exception as exc:
        raise PacketBuildError(f"cannot fingerprint source: {exc}")

    transcript = clips_data.get("transcript") or {}
    words = _window_transcript_words(transcript, float(start), float(end))
    packet = build_opening_packet(
        source_fingerprint=fingerprint,
        final_start=float(start),
        final_end=float(end),
        words=words,
        candidate_id=clip.get("candidate_id"),
        moment_id=clip.get("moment_id"),
        av_evidence=clip.get("opening_av_evidence"),
        language=clip.get("language") or clips_data.get("language"),
        preferences=user_preference or {},
    )

    clip_key = f"clip-{index}"
    funnel = clips_data.get("opening_funnel")
    if not isinstance(funnel, dict):
        funnel = new_opening_funnel()
        clips_data["opening_funnel"] = funnel

    # Explicit user-authored override: wins, marked, no provider call.
    override = user_authored_override(clip)
    if override is not None:
        plan = {
            "schema_version": OPENING_PLAN_SCHEMA_VERSION,
            "packet_id": packet.get("packet_id"),
            "packet_identity": packet.get("identity"),
            "user_authored": True,
            "text": override["text"],
            "route": route,
        }
        clip["opening_plan"] = plan
        record_opening_result(
            funnel, clip_key,
            {"packet_id": packet.get("packet_id"), "requested": 0,
             "validated": [], "rejected": [], "selected": None,
             "selection": None,
             "no_selection_reason": "user_authored_override",
             "failures": [], "notes": []},
            user_authored=True,
        )
        return plan

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
        fallback_provider=fallback_provider,
        budget=budget or get_run_budget(),
        user_preference=user_preference,
    )
    plan = {
        "schema_version": OPENING_PLAN_SCHEMA_VERSION,
        "packet_id": packet.get("packet_id"),
        "packet_identity": packet.get("identity"),
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
