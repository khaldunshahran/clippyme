"""Robust parser for Gemini viral-moment JSON responses.

Implements a 5-level fallback chain designed to make the clip-detection
pipeline resilient to the kinds of malformations Gemini occasionally
produces (stray backslashes, smart quotes, code fences, reasoning
prose mixed with the JSON body, trailing commas, etc.):

    1. strict   - json.loads on the section after the ``### JSON ###``
                  delimiter (or on the full text if absent).
    2. clean    - deterministic cleanup pass (smart quotes, trailing
                  commas, lone backslashes, control chars) then retry.
    3. json_repair - delegate to the ``json_repair`` library if
                  installed.
    4. retry    - one additional round-trip to Gemini with the decoder
                  error as context, asking it to emit JSON only.
    5. fallback - give up, return ``ParseResult.data = None`` so the
                  caller can degrade gracefully (whole-video mode).

Post-parse, ``validate_and_dedupe`` runs the result through the
``ViralClipsResponse`` Pydantic model and removes clips whose time
ranges overlap more than ``overlap_threshold`` (IoU) with a
higher-scoring neighbour.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import ValidationError

from clippyme.schemas import ViralClipsResponse

logger = logging.getLogger(__name__)

JSON_DELIMITER = "### JSON ###"

# The delimiter in the prompt is canonical, but Gemini (especially flash)
# occasionally emits tiny variations: different spacing, lowercase, extra
# hashes, or markdown bold wrapping. Match them all with a single regex.
_DELIMITER_PATTERN = re.compile(
    r"(?:\*{0,2})#{2,4}\s*json\s*#{2,4}(?:\*{0,2})",
    re.IGNORECASE,
)

_CODE_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_CODE_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_LONE_BACKSLASH = re.compile(r'\\(?!["\\/bfnrtu])')
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass
class ParseResult:
    """Outcome of ``parse_gemini_response``.

    ``parse_path`` tells downstream logging which level of the chain
    succeeded (or ``"fallback"`` if every level failed).
    """
    data: Optional[Dict[str, Any]]
    parse_path: str  # strict | clean | json_repair | retry | fallback
    duration_ms: float
    error: Optional[str] = None


def _extract_json_section(text: str) -> str:
    """Isolate the JSON body from reasoning + code fences.

    If a delimiter matching ``_DELIMITER_PATTERN`` is present,
    everything before the LAST occurrence is discarded (chain-of-
    thought reasoning). The "last occurrence" rule matters because
    Gemini occasionally echoes the delimiter inside its own reasoning.

    Then any ``` ```json ... ``` ``` fence is stripped defensively so
    the rest of the pipeline sees raw JSON whatever the model emits.
    """
    matches = list(_DELIMITER_PATTERN.finditer(text))
    if matches:
        text = text[matches[-1].end():]
    text = _CODE_FENCE_OPEN.sub("", text)
    text = _CODE_FENCE_CLOSE.sub("", text)
    return text.strip()


def _clean_json(raw: str) -> str:
    """Deterministic JSON repair for the most common Gemini mistakes.

    * curly/smart quotes -> straight ASCII quotes
    * trailing comma before ``}``/``]`` -> removed
    * lone backslash not part of a valid escape -> doubled
    * ASCII control chars (except ``\\n \\r \\t``) -> stripped
    """
    raw = (
        raw.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    raw = _TRAILING_COMMA.sub(r"\1", raw)
    raw = _LONE_BACKSLASH.sub(r"\\\\", raw)
    raw = _CONTROL_CHARS.sub("", raw)
    return raw


def parse_gemini_response(
    text: str,
    retry_fn: Optional[Callable[[str], str]] = None,
    request_id: str = "",
) -> ParseResult:
    """Run the 5-level parsing chain on a Gemini response.

    Parameters
    ----------
    text:
        The raw text returned by ``response.text`` from the
        google-genai SDK.
    retry_fn:
        Optional callable that takes the last decoder error message
        and returns a fresh Gemini response (as text) for one retry.
        Pass ``None`` to disable level 4.
    request_id:
        Opaque ID to correlate logs across the parse chain.
    """
    t0 = time.time()
    section = _extract_json_section(text)

    # Level 1: strict ------------------------------------------------
    try:
        data = json.loads(section)
        return ParseResult(data, "strict", (time.time() - t0) * 1000)
    except json.JSONDecodeError as e1:
        strict_err = f"{e1.msg}: line {e1.lineno} col {e1.colno}"

    # Level 2: deterministic clean ----------------------------------
    cleaned = _clean_json(section)
    try:
        data = json.loads(cleaned)
        return ParseResult(data, "clean", (time.time() - t0) * 1000)
    except json.JSONDecodeError:
        pass

    # Level 3: json_repair library ----------------------------------
    try:
        from json_repair import repair_json  # type: ignore

        data = json.loads(repair_json(section))
        return ParseResult(data, "json_repair", (time.time() - t0) * 1000)
    except Exception:  # pragma: no cover - json_repair may be missing
        pass

    # Level 4: one retry with error context -------------------------
    if retry_fn is not None:
        try:
            retry_text = retry_fn(strict_err)
            retry_section = _extract_json_section(retry_text)
            data = json.loads(_clean_json(retry_section))
            return ParseResult(data, "retry", (time.time() - t0) * 1000)
        except Exception as e:
            return ParseResult(
                None,
                "fallback",
                (time.time() - t0) * 1000,
                f"retry failed: {e}; original: {strict_err}",
            )

    return ParseResult(None, "fallback", (time.time() - t0) * 1000, strict_err)


def _viral_reason_is_generic(reason: str) -> bool:
    """Heuristic: does ``viral_reason`` look like a placeholder?

    We already reject reasons shorter than 20 chars via Pydantic, but
    a 25-char generic line ("this is a cool moment in the video") can
    still slip through. Detect the most common hedging phrases and
    flag them so the caller can either drop the clip or downgrade the
    score.
    """
    lowered = reason.lower().strip()
    generic_markers = (
        "interesting point",
        "cool moment",
        "great content",
        "good clip",
        "this moment",
        "this part",
        "nice segment",
        "important point",
    )
    if any(m in lowered for m in generic_markers):
        return True
    # A reason with no digits AND no quoted fragment almost never
    # cites a specific hook or payoff — treat as weak signal.
    has_digit = any(ch.isdigit() for ch in lowered)
    has_quote = '"' in reason or "'" in reason
    return (not has_digit) and (not has_quote) and len(lowered.split()) < 8


# Timestamp coercion now lives in clippyme.api.schemas.ViralClip as a
# @field_validator('start','end', mode='before'). The legacy helpers
# _coerce_timestamp() and _normalize_clip_timestamps() that used to
# pre-walk the dict before Pydantic were removed in this commit — they
# were unused after schema validation took over. If you need to expand
# the coercion rules, add the logic inside ViralClip._coerce_timestamp
# so every downstream caller gets it automatically.


def _extract_key_entities(text: str) -> set[str]:
    """Extract notable proper nouns / entities (capitalized names, dollar amounts, large numbers)."""
    if not text:
        return set()
    tokens = re.findall(r"\b(?:[A-Z][a-z0-9_]{3,}|\$\d+(?:,\d+)*(?:\.\d+)?[kKmMbB]?|\d+[kKmMbB])\b", text)
    stopwords = {
        "this", "that", "what", "when", "where", "which", "with", "from",
        "about", "after", "before", "then", "they", "there", "here", "some",
        "just", "more", "your", "their", "have", "been", "opens", "incredible",
        "shocking", "speaker", "video", "short", "reel", "tiktok", "youtube",
        "reveals", "detail", "pricing", "secret", "tactic", "first", "second",
        "everyone", "anyone", "someone", "nothing", "everything", "always"
    }
    return {t.lower() for t in tokens if t.lower() not in stopwords}


def _has_narrative_continuity(c1: Any, c2: Any) -> bool:
    """Check if c2 is an explicit continuation of c1's narrative arc."""
    d1 = c1.model_dump() if hasattr(c1, "model_dump") else c1
    d2 = c2.model_dump() if hasattr(c2, "model_dump") else c2

    t1_title = (d1.get("video_title_for_youtube_short") or "").strip()
    t2_title = (d2.get("video_title_for_youtube_short") or "").strip()

    # Discourse continuation cues in c2 (e.g. Giuliani responded, then sent photo, etc.)
    t2_text = f"{t2_title} {d2.get('viral_reason', '')} {d2.get('viral_hook_text', '')}".lower()
    continuation_cues = (
        "responded by", "replied by", "then he", "then she", "then they",
        "he lied about", "she lied about", "forgot about the photo",
        "then sent", "later told", "turns out", "twist where",
        "and he laughed", "and they published", "part 2", "continuation",
    )
    if any(cue in t2_text for cue in continuation_cues):
        return True

    # Shared proper entities between titled candidates
    if t1_title and t2_title:
        spk1 = (d1.get("speaker_name") or "").strip().lower()
        spk2 = (d2.get("speaker_name") or "").strip().lower()
        if spk1 and spk2 and spk1 != spk2:
            return False

        e1 = _extract_key_entities(f"{t1_title} {d1.get('viral_reason', '')}")
        e2 = _extract_key_entities(f"{t2_title} {d2.get('viral_reason', '')}")
        shared = e1.intersection(e2)
        if len(shared) >= 1:
            return True

    return False


def consolidate_narrative_continuations(
    clips: List[Any],
    max_gap_sec: float = 25.0,
    max_combined_sec: float = 180.0,
) -> List[Any]:
    """Merge adjacent candidate clips that belong to the same continuous story arc."""
    if len(clips) < 2:
        return clips

    def _get_start(c):
        return c.start if hasattr(c, "start") else c.get("start", 0.0)

    def _get_end(c):
        return c.end if hasattr(c, "end") else c.get("end", 0.0)

    sorted_clips = sorted(clips, key=_get_start)
    merged: List[Any] = []

    i = 0
    while i < len(sorted_clips):
        curr = sorted_clips[i]
        while i + 1 < len(sorted_clips):
            nxt = sorted_clips[i + 1]
            c_start, c_end = _get_start(curr), _get_end(curr)
            n_start, n_end = _get_start(nxt), _get_end(nxt)
            gap = n_start - c_end
            combined_dur = n_end - c_start

            if 0.0 <= gap <= max_gap_sec and combined_dur <= max_combined_sec and _has_narrative_continuity(curr, nxt):
                logger.info(
                    "consolidate_narrative_continuations: fusing story fragments [%.1f–%.1f] and [%.1f–%.1f] into unified %.1fs clip",
                    c_start, c_end, n_start, n_end, combined_dur,
                )
                is_model = hasattr(curr, "model_copy")
                if is_model:
                    from clippyme.schemas import ViralClip
                    d1 = curr.model_dump()
                    d2 = nxt.model_dump()
                else:
                    d1 = dict(curr)
                    d2 = dict(nxt)

                t1 = d1.get("video_title_for_youtube_short", "")
                t2 = d2.get("video_title_for_youtube_short", "")
                title = t1 if len(t1) >= len(t2) else t2
                if t1 and t2 and t1 != t2 and len(f"{t1} — {t2}") <= 100:
                    title = f"{t1} — {t2}"

                tier = "short" if combined_dur <= 60.0 else ("mid" if combined_dur <= 120.0 else "extended")

                merged_dict = {
                    **d1,
                    "start": c_start,
                    "end": n_end,
                    "viral_score": max(d1.get("viral_score", 0), d2.get("viral_score", 0)),
                    "video_title_for_youtube_short": title,
                    "viral_hook_text": d1.get("viral_hook_text") or d2.get("viral_hook_text", ""),
                    "viral_reason": f"{d1.get('viral_reason', '')} Continues with: {d2.get('viral_reason', '')}".strip(),
                    "duration_tier": tier,
                }
                if is_model:
                    curr = ViralClip.model_validate(merged_dict)
                else:
                    curr = merged_dict
                i += 1
            else:
                break

        merged.append(curr)
        i += 1

    return merged


# --- 2F deterministic score cross-check -----------------------------------
# The LLM's viral_score is a self-assessment with no grounding. These pure
# helpers compute an independent heuristic score from the transcript (and,
# when a media path is available, the clip's audio energy) so a confident
# but flat clip can't sail through on LLM vibes alone. A disagreement above
# SCORE_DISAGREEMENT_THRESHOLD flags the candidate for review and down-ranks
# it in sort order — the LLM score itself is never silently replaced.

SCORE_DISAGREEMENT_THRESHOLD = 25
SCORE_REVIEW_RANK_PENALTY = 10


def _norm_word(w: Dict[str, Any]):
    """Read a transcript word in either {word,start,end} or {w,s,e} shape."""
    if not isinstance(w, dict):
        return None, None, None
    text = w.get("word", w.get("w"))
    start = w.get("start", w.get("s"))
    end = w.get("end", w.get("e"))
    try:
        s = float(start) if start is not None else None
        e = float(end) if end is not None else None
    except (TypeError, ValueError):
        return None, None, None
    return text, s, e


def _overlap_words(transcript_words, start: float, end: float) -> List[Dict[str, Any]]:
    out = []
    for w in transcript_words or []:
        text, s, e = _norm_word(w)
        if s is None or e is None:
            continue
        if e > start and s < end:
            out.append({"word": text, "start": s, "end": e})
    return out


def compute_deterministic_features(
    candidate: Dict[str, Any],
    transcript_words=None,
    audio_energy_variance: Optional[float] = None,
) -> Dict[str, Any]:
    """Deterministic virality features for one clip candidate (pure).

    - speech_rate_wpm: words overlapping [start, end] per minute
    - question/exclamation_density: ? and ! per overlapping word
    - audio_energy_variance: per-window dBFS variance when provided
    """
    try:
        start = float(candidate.get("start", 0) or 0)
        end = float(candidate.get("end", 0) or 0)
    except (TypeError, ValueError):
        start, end = 0.0, 0.0
    dur = max(0.1, end - start)
    words = _overlap_words(transcript_words, start, end)
    n = len(words)
    text = " ".join(str(w.get("word") or "") for w in words)
    var = None
    if audio_energy_variance is not None:
        try:
            var = float(audio_energy_variance)
        except (TypeError, ValueError):
            var = None
    return {
        "word_count": n,
        "duration": round(dur, 2),
        "speech_rate_wpm": round((n / dur) * 60.0, 1),
        "question_density": round(text.count("?") / max(1, n), 4),
        "exclamation_density": round(text.count("!") / max(1, n), 4),
        "audio_energy_variance": round(var, 3) if var is not None else None,
    }


#: Deterministic scoring features eligible for virality weight calibration
#: (Phase 3A). Each key scales its named delta in ``deterministic_score``.
SCORE_FEATURE_KEYS = [
    "speech_rate_wpm",
    "question_density",
    "exclamation_density",
    "audio_energy_variance",
    "duration",
]


def deterministic_score(
    features: Dict[str, Any],
    weight_overrides: Optional[Dict[str, float]] = None,
) -> int:
    """Heuristic 1-100 virality score from deterministic features (pure).

    Rewards: conversational speech rate (140-185 wpm), question/exclamation
    hooks, dynamic audio delivery, and the 20-45s short-form sweet spot.
    Penalises: rushed/dragging speech, flat monotone audio, sub-12s stubs.

    Phase 3A: each named per-feature delta is multiplied by its override
    (``weight_overrides.get(key, 1.0)``). ``weight_overrides=None``
    lazy-loads the approved calibration file via
    ``virality_calibration.get_active_weight_overrides()`` (``{}`` when no
    approved file exists); an explicit ``{}`` reproduces the Phase-2
    defaults exactly.
    """
    if weight_overrides is None:
        try:
            from clippyme.domain.virality_calibration import (
                get_active_weight_overrides,
            )
            weight_overrides = get_active_weight_overrides()
        except Exception:  # noqa: BLE001 - calibration must never break scoring
            weight_overrides = {}
    w = lambda key: float(weight_overrides.get(key, 1.0))  # noqa: E731

    feats = features or {}
    score = 50.0

    wpm = feats.get("speech_rate_wpm") or 0
    if 140 <= wpm <= 185:
        wpm_delta = 15.0
    elif 110 <= wpm < 140 or 185 < wpm <= 220:
        wpm_delta = 6.0
    else:
        wpm_delta = -10.0
    score += wpm_delta * w("speech_rate_wpm")

    question_delta = 8.0 if (feats.get("question_density") or 0) >= 0.03 else 0.0
    score += question_delta * w("question_density")

    exclamation_delta = 8.0 if (feats.get("exclamation_density") or 0) >= 0.03 else 0.0
    score += exclamation_delta * w("exclamation_density")

    var = feats.get("audio_energy_variance")
    if var is None:
        energy_delta = 0.0
    elif var >= 25:
        energy_delta = 10.0
    elif var >= 10:
        energy_delta = 4.0
    else:
        energy_delta = -6.0
    score += energy_delta * w("audio_energy_variance")

    dur = feats.get("duration") or 0
    if 20 <= dur <= 45:
        duration_delta = 10.0
    elif dur < 12:
        duration_delta = -8.0
    else:
        duration_delta = 0.0
    score += duration_delta * w("duration")

    return max(1, min(100, int(round(score))))


def cross_check_scores(
    candidates: List[Dict[str, Any]],
    transcript_words=None,
    media_path: Optional[str] = None,
    energy_fn=None,
    weight_overrides: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Attach deterministic cross-check fields to candidate dicts.

    Each candidate gains ``deterministic_features``, ``deterministic_score``,
    ``score_disagreement`` (|LLM - deterministic|) and
    ``score_review_required``. Flagged candidates are down-ranked by
    ``SCORE_REVIEW_RANK_PENALTY`` in sort order only — the LLM
    ``viral_score`` itself is never replaced. ``energy_fn`` is injectable
    for tests (defaults to
    ``media_probe.compute_audio_energy_variance``); a failing energy probe
    degrades to transcript-only features. ``weight_overrides`` (Phase 3A)
    is forwarded to ``deterministic_score``; ``None`` loads the approved
    calibration file, ``{}`` keeps Phase-2 defaults.
    """
    if energy_fn is None:
        try:
            from clippyme.pipeline.media_probe import compute_audio_energy_variance
            energy_fn = compute_audio_energy_variance
        except Exception:  # noqa: BLE001 - import must never break scoring
            energy_fn = lambda *a, **k: None  # noqa: E731

    out = []
    for c in candidates:
        if not isinstance(c, dict):
            out.append(c)
            continue
        try:
            c_start = float(c.get("start", 0) or 0)
            c_end = float(c.get("end", 0) or 0)
        except (TypeError, ValueError):
            c_start, c_end = 0.0, 0.0
        var = None
        if media_path:
            try:
                var = energy_fn(media_path, c_start, c_end)
            except Exception:  # noqa: BLE001 - degrade to transcript-only
                var = None
        feats = compute_deterministic_features(
            c, transcript_words, audio_energy_variance=var)
        det = deterministic_score(feats, weight_overrides)
        try:
            llm = int(c.get("viral_score"))
        except (TypeError, ValueError):
            llm = det
        disagreement = abs(llm - det)
        review = disagreement > SCORE_DISAGREEMENT_THRESHOLD
        c = dict(c)
        c["deterministic_features"] = feats
        c["deterministic_score"] = det
        c["score_disagreement"] = disagreement
        c["score_review_required"] = bool(review)
        if review:
            logger.info(
                "cross_check_scores: clip %.1f-%.1fs LLM=%d deterministic=%d "
                "disagreement=%d > %d — flagged for review (down-ranked, score kept)",
                c_start, c_end, llm, det, disagreement,
                SCORE_DISAGREEMENT_THRESHOLD,
            )
        out.append(c)
    out.sort(key=lambda c: -(
        (c.get("viral_score") or 0)
        - (SCORE_REVIEW_RANK_PENALTY if isinstance(c, dict) and c.get("score_review_required") else 0)
    ))
    return out


def validate_and_dedupe(
    data: Dict[str, Any],
    video_duration: Optional[float] = None,
    overlap_threshold: float = 0.7,
    drop_generic: bool = False,
    transcript_words=None,
    media_path: Optional[str] = None,
    weight_overrides: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Pydantic-validate then remove overlapping clips.

    Clips are sorted by ``viral_score`` desc; for each candidate, if
    its intersection-over-union with any already-kept clip exceeds
    ``overlap_threshold`` it's dropped. When ``video_duration`` is
    given, clips whose ``end`` exceeds it are also dropped.

    ``transcript_words`` / ``media_path`` (both optional, 2F): feed the
    deterministic score cross-check — candidates whose LLM viral_score
    disagrees with the deterministic heuristic by more than
    ``SCORE_DISAGREEMENT_THRESHOLD`` are flagged (``score_review_required``)
    and down-ranked; the LLM score itself is never replaced.

    Raises
    ------
    pydantic.ValidationError
        Only if the top-level response shape is malformed (e.g. ``shorts``
        is missing or not a list). Individual clip validation failures
        no longer nuke the whole batch — invalid clips are logged and
        silently dropped so the pipeline proceeds with whatever Gemini
        got right. This is critical because Gemini occasionally returns
        ~15 clips where 14 are valid and 1 has a duration of 2.96s; the
        old behaviour rejected all 15 and fell back to whole-video mode.
    """
    # Timestamp coercion lives entirely in ViralClip's @field_validator
    # (mode='before'). The legacy _normalize_clip_timestamps() helper was
    # removed to keep the pipeline single-source-of-truth — if you need a
    # log of coerced fields, add it inside the validator.

    # Per-clip resilience: iterate manually instead of relying on
    # ViralClipsResponse's List[ViralClip] fail-fast behaviour.
    raw_shorts = (data or {}).get("shorts") if isinstance(data, dict) else None
    if not isinstance(raw_shorts, list):
        # Top-level shape wrong — let Pydantic emit the authoritative error.
        ViralClipsResponse.model_validate(data)
        return []

    from clippyme.schemas import ViralClip  # neutral module, no api dependency

    candidates: List = []
    dropped_invalid = 0
    for i, raw in enumerate(raw_shorts):
        try:
            candidates.append(ViralClip.model_validate(raw))
        except Exception as exc:
            dropped_invalid += 1
            # Log the first validation error message verbatim so debugging
            # which field failed is one grep away.
            msg = str(exc).splitlines()[0] if str(exc) else "unknown"
            logger.warning(
                "validate_and_dedupe: dropping clip #%d — %s (input: start=%s end=%s)",
                i, msg[:200], raw.get("start") if isinstance(raw, dict) else "?",
                raw.get("end") if isinstance(raw, dict) else "?",
            )
    if dropped_invalid:
        logger.info(
            "validate_and_dedupe: %d/%d clip(s) rejected by per-clip validation, "
            "%d survived",
            dropped_invalid, len(raw_shorts), len(candidates),
        )
    if video_duration is not None:
        candidates = [c for c in candidates if c.end <= video_duration + 0.5]

    if drop_generic:
        before = len(candidates)
        candidates = [c for c in candidates if not _viral_reason_is_generic(c.viral_reason)]
        dropped = before - len(candidates)
        if dropped:
            logger.info(
                "validate_and_dedupe: dropped %d clip(s) with generic viral_reason", dropped
            )

    # Narrative Completeness: fuse adjacent story fragments into full arcs
    candidates = consolidate_narrative_continuations(candidates)

    candidates.sort(key=lambda c: -c.viral_score)

    kept: List = []
    for clip in candidates:
        overlaps = False
        for k in kept:
            inter = max(0.0, min(clip.end, k.end) - max(clip.start, k.start))
            union = max(clip.end, k.end) - min(clip.start, k.start)
            if union > 0 and (inter / union) > overlap_threshold:
                overlaps = True
                break
        if not overlaps:
            kept.append(clip)

    dumped = [c.model_dump() for c in kept]
    # 2F: deterministic cross-check — flag (and down-rank, never silently
    # replace) LLM scores that disagree with transcript/audio features.
    # Phase 3A: approved virality weight overrides flow through here.
    return cross_check_scores(
        dumped, transcript_words=transcript_words, media_path=media_path,
        weight_overrides=weight_overrides)


def _truncate_words(text: str, n: int = 10) -> str:
    """Normalize whitespace and truncate to the first ``n`` words."""
    parts = (text or "").strip().split()
    return " ".join(parts[:n]).strip()


def backfill_hook_text(
    clips: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
    fallback_title: str = "",
) -> List[Dict[str, Any]]:
    """Ensure EVERY clip has a non-empty ``viral_hook_text``.

    IMPORTANT: The hook is designed to be a SCROLL-STOPPING overlay, not
    a transcript echo. The prompt in ``main.py`` explicitly forbids quoting
    the first spoken words, so this backfill also does NOT fall back to
    transcript words — doing so would reintroduce the exact bug the new
    prompt is meant to fix.

    Strategy (first success wins):
      1. Keep Gemini's hook if non-empty (normalized + truncated to 8 words).
      2. Derive a teaser-style hook from the YouTube title if present.
      3. Use ``fallback_title`` (source title) similarly truncated.
      4. Hard-coded generic teaser so the field is never empty.

    The ``words`` parameter is kept for backward compatibility with
    callers in ``main.py`` and ``job_results.py``; it is intentionally
    unused.

    This function is idempotent and mutates each clip dict in place,
    returning the same list for convenience.
    """
    del words  # unused — see docstring

    for clip in clips:
        existing = _truncate_words(clip.get("viral_hook_text") or "", 8)
        if existing:
            clip["viral_hook_text"] = existing
            continue

        title_hook = _truncate_words(
            clip.get("video_title_for_youtube_short", "") or fallback_title,
            8,
        )
        if title_hook:
            clip["viral_hook_text"] = title_hook
            continue

        clip["viral_hook_text"] = "You need to see this"

    return clips


def cap_clips_by_score(clips: list[dict], max_clips: int, min_score: int = 0) -> list[dict]:
    """Select a segment's clips by ``viral_score``.

    ``min_score`` (auto selection) first drops every clip scoring below the
    floor — the count then follows the material instead of being fixed, and a
    segment with nothing good yields nothing. ``max_clips`` then keeps the
    top-N by ``viral_score`` (desc) as a ceiling; a non-positive ``max_clips``
    (or a count already at/under it) is a no-op. Ties keep their original
    relative order (stable sort).
    """
    if min_score and min_score > 0:
        clips = [c for c in clips if c.get("viral_score", 0) >= min_score]
    if not max_clips or max_clips <= 0 or len(clips) <= max_clips:
        return clips
    ranked = sorted(clips, key=lambda c: c.get("viral_score", 0), reverse=True)
    return ranked[:max_clips]


def drop_wordless_clips(clips: list[dict], words: list[dict]) -> list[dict]:
    """Keep only clips whose [start,end] overlaps at least one transcript word.

    A clip with zero words in range is either a Gemini hallucination (empty
    transcript) or would crash subtitle compose ("No words found") — drop it.
    Same half-open overlap test as generate_ass_karaoke.

    Accepts both word shapes the pipeline uses: the transcript's
    ``{"start","end",...}`` and the TOON-encoded prompt payload's
    ``{"s","e",...}`` (``build_viral_prompt``). Words missing a start/end are
    skipped, never raised on.
    """
    kept = []
    for c in clips:
        cs, ce = float(c.get("start", 0)), float(c.get("end", 0))
        hit = False
        for w in words:
            ws = w.get("start", w.get("s"))
            we = w.get("end", w.get("e"))
            if ws is None or we is None:
                continue
            if we > cs and ws < ce:
                hit = True
                break
        if hit:
            kept.append(c)
    return kept


__all__ = [
    "JSON_DELIMITER",
    "ParseResult",
    "parse_gemini_response",
    "validate_and_dedupe",
    "compute_deterministic_features",
    "deterministic_score",
    "cross_check_scores",
    "SCORE_DISAGREEMENT_THRESHOLD",
    "backfill_hook_text",
    "drop_wordless_clips",
]
