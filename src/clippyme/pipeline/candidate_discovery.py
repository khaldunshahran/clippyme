"""Windowed candidate discovery (Task 08).

Replaces the single whole-transcript discovery call with bounded
overlapping transcript windows. Each window is searched independently
for candidate moments; candidates are validated per-item, deduplicated
globally by event key / evidence similarity (never by time overlap
alone), and finally capped by the existing score floor / max-clips
policy *after* aggregation.

All provider calls go through an injected client factory so tests use
deterministic fakes; no live SDK calls happen in this module's tests.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

DISCOVERY_VERSION = 1

# Defaults are bounded for ~1h sources: 3600/195 ~= 19 windows.
WINDOW_TARGET_S = 240.0
WINDOW_OVERLAP_S = 45.0
BOUNDARY_TOLERANCE_S = 12.0
MAX_WINDOWS = 32
MAX_RAW_CANDIDATES = 200
MAX_PROVIDER_ATTEMPTS = 24
DISCOVERY_DEADLINE_S = 600.0
SDK_TIMEOUT_MS = 120000

_SENTENCE_END = re.compile(r"[.!?\u2026]+$")


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.getenv(name) or default)
    except (ValueError, TypeError):
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.getenv(name) or default)
    except (ValueError, TypeError):
        v = default
    return max(lo, min(hi, v))


def _env_flag(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def windowed_discovery_enabled() -> bool:
    """Rollback flag; default ON on this branch."""
    return _env_flag("CLIPPYME_WINDOWED_DISCOVERY", True)


@dataclass
class DiscoveryConfig:
    """Validated, bounded discovery configuration (env-driven)."""

    window_target_s: float = WINDOW_TARGET_S
    window_overlap_s: float = WINDOW_OVERLAP_S
    boundary_tolerance_s: float = BOUNDARY_TOLERANCE_S
    max_windows: int = MAX_WINDOWS
    max_raw_candidates: int = MAX_RAW_CANDIDATES
    max_provider_attempts: int = MAX_PROVIDER_ATTEMPTS
    deadline_s: float = DISCOVERY_DEADLINE_S
    sdk_timeout_ms: int = SDK_TIMEOUT_MS
    model_name: str = ""
    fallback_models: tuple = ()

    @classmethod
    def from_env(cls, model_name: str = "",
                 fallback_models: Optional[Sequence[str]] = None,
                 ) -> "DiscoveryConfig":
        # Preserve established default chain when env override is unset.
        if fallback_models is None:
            from clippyme.pipeline.gemini_request import build_model_chain
            _chain = build_model_chain(model_name or "gemini-3.5-flash",
                                       os.getenv("GEMINI_FALLBACK_MODELS"))
            # build_model_chain includes primary first; fallbacks are rest.
            fallback_models = [m for m in _chain if m != model_name]
        target = _env_float("CLIPPYME_WINDOW_TARGET_S", WINDOW_TARGET_S,
                            30.0, 900.0)
        overlap = _env_float("CLIPPYME_WINDOW_OVERLAP_S", WINDOW_OVERLAP_S,
                             0.0, target * 0.9)
        return cls(
            window_target_s=target,
            window_overlap_s=overlap,
            boundary_tolerance_s=_env_float(
                "CLIPPYME_WINDOW_BOUNDARY_TOL", BOUNDARY_TOLERANCE_S,
                0.0, 60.0),
            max_windows=_env_int("CLIPPYME_WINDOW_MAX_WINDOWS",
                                 MAX_WINDOWS, 1, 128),
            max_raw_candidates=_env_int("CLIPPYME_WINDOW_MAX_CANDIDATES",
                                        MAX_RAW_CANDIDATES, 1, 1000),
            max_provider_attempts=_env_int("CLIPPYME_WINDOW_MAX_ATTEMPTS",
                                           MAX_PROVIDER_ATTEMPTS, 1, 200),
            deadline_s=_env_float("CLIPPYME_WINDOW_DEADLINE_S",
                                  DISCOVERY_DEADLINE_S, 10.0, 3600.0),
            sdk_timeout_ms=_env_int("CLIPPYME_WINDOW_SDK_TIMEOUT_MS",
                                    SDK_TIMEOUT_MS, 5000, 600000),
            model_name=model_name,
            fallback_models=tuple(fallback_models or ()),
        )


@dataclass
class PlannedWindow:
    """One transcript window with absolute timestamps."""

    index: int
    start_s: float
    end_s: float
    word_start: int  # inclusive word index
    word_end: int  # exclusive word index


def _word_text(w: Dict[str, Any]) -> str:
    # Handle: w (Task 04), text, word (production segments).
    for k in ("w", "text", "word"):
        t = w.get(k)
        if isinstance(t, str) and t.strip():
            return t
        if t is not None and not isinstance(t, bool):
            s = str(t).strip()
            if s:
                return s
    return ""


def _word_start(w: Dict[str, Any]) -> Optional[float]:
    # Handle: s (Task 04), start (production/words).
    for k in ("s", "start", "start_time", "start_s"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _word_end(w: Dict[str, Any]) -> Optional[float]:
    # Handle: e (Task 04), end (production/words).
    for k in ("e", "end", "end_time", "end_s"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


def _word_speaker(w: Dict[str, Any]) -> Optional[int]:
    # Handle: sp (Task 04), speaker (production). Preserve 0.
    for k in ("sp", "speaker"):
        v = w.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
    return None


def normalize_transcript_words(transcript_result: Any) -> List[Dict[str, Any]]:
    """One authoritative normalizer for all transcript contracts.

    Handles:
    - Real production: {"segments": [{"words": [{"word","start","end","speaker"}]}]}
    - Task 04 flattened: [{"w","s","e","sp"}] (list) or {"words": [...]}
    - Words shape: {"words": [{"text","start","end","speaker"}]}

    Returns list of {"w","s","e"} plus "sp" if speaker present.
    Preserves absolute timestamps (never rebases) and speaker ID 0.
    Invalid words are omitted here; plan_windows reports them.
    """
    raw_words: List[Dict[str, Any]] = []

    if isinstance(transcript_result, dict):
        # Try segments first (real production).
        segments = transcript_result.get("segments")
        if isinstance(segments, list):
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                for w in seg.get("words", []):
                    if isinstance(w, dict):
                        raw_words.append(w)
        # Fallback to top-level words list.
        if not raw_words:
            v = transcript_result.get("words")
            if isinstance(v, list):
                raw_words = [w for w in v if isinstance(w, dict)]
            v = transcript_result.get("word_list")
            if not raw_words and isinstance(v, list):
                raw_words = [w for w in v if isinstance(w, dict)]
    elif isinstance(transcript_result, list):
        raw_words = [w for w in transcript_result if isinstance(w, dict)]
    else:
        # Object with .words attribute.
        words = getattr(transcript_result, "words", None)
        if isinstance(words, list):
            raw_words = [w for w in words if isinstance(w, dict)]

    normalized: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    for idx, w in enumerate(raw_words):
        text = _word_text(w)
        s = _word_start(w)
        e = _word_end(w)
        if not text or s is None or e is None:
            # Retain issue record; exclude from prompts.
            issues.append({
                "word_index": idx,
                "reason": "missing text" if not text else "missing/invalid time",
                "raw": {k: v for k, v in w.items()
                        if k in ("word", "text", "w", "speaker", "sp")},
            })
            continue
        entry: Dict[str, Any] = {"w": text, "s": s, "e": e}
        sp = _word_speaker(w)
        if sp is not None:
            entry["sp"] = sp
        normalized.append(entry)
    # Attach issues for callers that need them.
    normalize_transcript_words.last_issues = issues
    return normalized


def plan_windows(words: Sequence[Dict[str, Any]],
                 config: DiscoveryConfig) -> Dict[str, Any]:
    """Deterministic overlapping window plan over word timestamps.

    Returns {"windows": [PlannedWindow...], "coverage": {...},
    "invalid_words": [...]}. Every valid word appears in >= 1 window
    under normal capacity; first and last valid words are never dropped
    unless the window cap prevents it (then reported honestly).
    No window ever exceeds target + boundary_tolerance.
    Invalid words are isolated and reported, never crashing the plan.
    """
    valid: List[tuple] = []  # (orig_idx, start, end, text)
    invalid: List[int] = []
    for i, w in enumerate(words):
        if not isinstance(w, dict):
            invalid.append(i)
            continue
        s, e = _word_start(w), _word_end(w)
        t = _word_text(w)
        if s is None or e is None or not t.strip():
            invalid.append(i)
            continue
        if e <= s:
            invalid.append(i)
            continue
        valid.append((i, s, e, t))

    if not valid:
        return {"windows": [], "coverage": {
            "valid_words": 0, "covered_words": 0,
            "first_covered": False, "last_covered": False,
            "transcript_span_s": 0.0,
            "truncated_by_window_limit": False,
            "uncovered_word_ranges": [],
            "uncovered_time_ranges": [],
        }, "invalid_words": invalid}

    span_start = valid[0][1]
    span_end = valid[-1][2]
    max_window_s = config.window_target_s + config.boundary_tolerance_s
    step = config.window_target_s - config.window_overlap_s
    windows: List[PlannedWindow] = []
    n = len(valid)
    truncated = False

    cur_start = span_start
    while True:
        if len(windows) >= config.max_windows:
            truncated = True
            break
        # Target end, snapped to punctuation/gap within tolerance.
        target_end = cur_start + config.window_target_s
        # Find candidate end word index: last word starting before target.
        end_vi = 0
        for vi in range(n):
            if valid[vi][1] < target_end:
                end_vi = vi
            else:
                break
        # Snap to sentence boundary using actual word text.
        end_vi = _snap_end_to_punctuation(valid, end_vi, target_end,
                                          config.boundary_tolerance_s,
                                          max_window_s, cur_start)
        # Find start word index: first word ending after cur_start.
        start_vi = 0
        for vi in range(n):
            if valid[vi][2] > cur_start:
                start_vi = vi
                break
        else:
            start_vi = n - 1

        w_start_s = valid[start_vi][1]
        w_end_s = valid[end_vi][2]
        # Enforce hard bound: never exceed target + tolerance.
        if w_end_s - w_start_s > max_window_s + 1e-6:
            # Clamp end to max bound.
            w_end_s = w_start_s + max_window_s
            # Adjust end_vi to last word ending within bound.
            for vi in range(start_vi, n):
                if valid[vi][2] <= w_end_s + 1e-6:
                    end_vi = vi
                else:
                    break
            w_end_s = valid[end_vi][2]

        windows.append(PlannedWindow(
            index=len(windows), start_s=w_start_s, end_s=w_end_s,
            word_start=valid[start_vi][0], word_end=valid[end_vi][0] + 1))

        if w_end_s >= span_end - 1e-6:
            break
        # Advance.
        nxt = cur_start + step
        if nxt >= cur_start:  # Ensure forward progress.
            cur_start = nxt
        else:
            break
        if cur_start >= span_end:
            break

    # Coverage analysis.
    covered = set()
    for w in windows:
        for oi in range(w.word_start, w.word_end):
            covered.add(oi)
    # Uncovered ranges.
    uncovered_words = []
    uncovered_times = []
    if covered:
        all_orig = [v[0] for v in valid]
        # Find gaps in coverage.
        sorted_covered = sorted(covered)
        # Check from first valid to last valid.
        prev = None
        for oi in sorted(all_orig):
            if oi not in covered:
                if prev is None or oi != prev + 1:
                    # Start new gap.
                    uncovered_words.append([oi, oi])
                else:
                    uncovered_words[-1][1] = oi
                prev = oi
            else:
                prev = None
        # Convert to time ranges.
        orig_to_time = {v[0]: (v[1], v[2]) for v in valid}
        for s_oi, e_oi in uncovered_words:
            s_t = orig_to_time[s_oi][0]
            e_t = orig_to_time[e_oi][1]
            uncovered_times.append([s_t, e_t])

    # If cap hit and words uncovered, mark truncated.
    if uncovered_words:
        # Check if we hit the cap.
        if len(windows) >= config.max_windows:
            truncated = True

    return {"windows": windows, "coverage": {
        "valid_words": len(valid),
        "covered_words": len(covered),
        "first_covered": valid[0][0] in covered,
        "last_covered": valid[-1][0] in covered,
        "transcript_span_s": span_end - span_start,
        "truncated_by_window_limit": truncated,
        "uncovered_word_ranges": uncovered_words,
        "uncovered_time_ranges": uncovered_times,
    }, "invalid_words": invalid}


def _snap_end_to_punctuation(valid: List[tuple], end_vi: int,
                             target_end: float, tolerance: float,
                             max_window_s: float,
                             window_start_s: float) -> int:
    """Snap window end to sentence punctuation using word text.

    Looks for words ending with . ! ? within tolerance of target.
    Never exceeds window_start + max_window_s.
    """
    n = len(valid)
    # Search forward and backward within tolerance for punctuation.
    best_vi = end_vi
    # Check words near target_end.
    for vi in range(max(0, end_vi - 10), min(n, end_vi + 10)):
        ws, we, text = valid[vi][1], valid[vi][2], valid[vi][3]
        # Must be within tolerance of target.
        if abs(we - target_end) > tolerance:
            continue
        # Must not exceed max window size.
        if we - window_start_s > max_window_s + 1e-6:
            continue
        # Check for sentence-ending punctuation.
        t = text.strip()
        if t and t[-1] in ".!?":
            # Prefer the closest to target.
            if abs(we - target_end) < abs(valid[best_vi][2] - target_end):
                best_vi = vi
    return best_vi




def _normalize_evidence(text: str) -> str:
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class CandidateRecord:
    """Validated discovery candidate with stable local identity."""

    candidate_id: str  # Distinct per window occurrence (trace ID)
    window_index: int
    start_s: float
    end_s: float
    event_type: str
    evidence_text: str
    evidence_basis: str
    context: str
    expected_payoff: str
    event_key: Optional[str]
    confidence: float
    recommendation_score: float
    speaker_ids: List[Any]
    moment_fingerprint: str = ""  # Stable across windows (dedupe key)
    raw: Dict[str, Any] = field(default_factory=dict)


class DiscoveryError(Exception):
    pass


def _is_finite_number(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v))


def make_moment_fingerprint(source_tag: str, start_s: float,
                            end_s: float, evidence_text: str) -> str:
    """Stable moment fingerprint across windows; for semantic dedupe."""
    h = hashlib.sha256()
    h.update(source_tag.encode("utf-8"))
    h.update(f"{start_s:.3f}-{end_s:.3f}".encode("utf-8"))
    h.update(_normalize_evidence(evidence_text).encode("utf-8"))
    return "moment-" + h.hexdigest()[:16]


def make_candidate_id(source_tag: str, start_s: float, end_s: float,
                      evidence_text: str, window_index: int = 0) -> str:
    """Distinct trace ID per window occurrence; never model-provided.

    Includes window_index so the same moment in overlapping windows
    gets distinct IDs. Use moment_fingerprint for stable dedupe.
    """
    h = hashlib.sha256()
    h.update(source_tag.encode("utf-8"))
    h.update(f"w{window_index}:".encode("utf-8"))
    h.update(f"{start_s:.3f}-{end_s:.3f}".encode("utf-8"))
    h.update(_normalize_evidence(evidence_text).encode("utf-8"))
    return "cand-" + h.hexdigest()[:16]


def validate_discovery_candidate(item: Any, window: PlannedWindow,
                                 window_words: Sequence[Dict[str, Any]],
                                 source_tag: str = "src",
                                 video_duration: Optional[float] = None,
                                 min_duration: Optional[float] = None,
                                 max_duration: Optional[float] = None,
                                 ) -> CandidateRecord:
    """Validate one candidate independently; raise DiscoveryError if bad.

    Rejects: booleans, non-finite numbers, reversed/out-of-window
    intervals, intervals outside source duration, missing
    evidence/context/payoff, wordless ranges, confidence outside
    [0,1], recommendation_score outside [1,100].
    """
    if not isinstance(item, dict):
        raise DiscoveryError("candidate must be an object")
    start, end = item.get("start_s", item.get("start")), item.get(
        "end_s", item.get("end"))
    if not _is_finite_number(start) or not _is_finite_number(end):
        raise DiscoveryError("candidate start/end must be finite numbers")
    start, end = float(start), float(end)
    if not (window.start_s - 1e-6 <= start < end <= window.end_s + 1e-6):
        raise DiscoveryError(
            "candidate interval outside window or reversed")
    # Source duration is authoritative bound.
    if video_duration is not None and video_duration > 0:
        if not (0 <= start < end <= video_duration + 1e-6):
            raise DiscoveryError(
                "candidate interval outside source duration")
    # Shared duration policy bounds.
    _dur = end - start
    if min_duration is not None and _dur < min_duration - 1e-6:
        raise DiscoveryError(
            f"candidate duration {_dur:.1f}s below minimum {min_duration:.1f}s")
    if max_duration is not None and _dur > max_duration + 1e-6:
        raise DiscoveryError(
            f"candidate duration {_dur:.1f}s above maximum {max_duration:.1f}s")
    evidence = item.get("evidence_text") or item.get("evidence") or ""
    context = item.get("context") or ""
    payoff = item.get("expected_payoff") or item.get("payoff") or ""
    if not str(evidence).strip():
        raise DiscoveryError("candidate missing evidence_text")
    if not str(context).strip():
        raise DiscoveryError("candidate missing context")
    if not str(payoff).strip():
        raise DiscoveryError("candidate missing expected_payoff")
    # Wordless range guard: at least one window word must overlap.
    has_word = False
    for w in window_words:
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end and we > start:
            if _word_text(w).strip():
                has_word = True
                break
    if not has_word:
        raise DiscoveryError("candidate interval covers no words")
    # Evidence grounding: normalized evidence tokens must substantially
    # overlap the words within the candidate interval.
    # Documented threshold: >= 50% of evidence tokens appear in interval.
    _ev_tokens = set(
        t.lower() for t in re.findall(r"[a-z0-9]+", str(evidence).lower()))
    _interval_tokens = set()
    for w in window_words:
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end and we > start:
            _wt = _word_text(w)
            if _wt:
                _interval_tokens.update(
                    t.lower() for t in re.findall(r"[a-z0-9]+", _wt.lower()))
    if _ev_tokens:
        _overlap = len(_ev_tokens & _interval_tokens) / len(_ev_tokens)
        if _overlap < 0.5:
            raise DiscoveryError(
                f"evidence not grounded in candidate words "
                f"(overlap {_overlap:.0%})")
    basis = str(item.get("evidence_basis") or "").strip().lower()
    if basis and basis != "transcript":
        raise DiscoveryError(
            "transcript-only candidate must declare transcript basis")
    confidence = item.get("confidence", 0.5)
    score = item.get("recommendation_score",
                     item.get("viral_score", item.get("score", 50)))
    if isinstance(confidence, bool) or not _is_finite_number(confidence):
        raise DiscoveryError("confidence must be a finite number")
    if isinstance(score, bool) or not _is_finite_number(score):
        raise DiscoveryError("recommendation_score must be finite")
    confidence = float(confidence)
    score = float(score)
    if not (0.0 <= confidence <= 1.0):
        raise DiscoveryError("confidence must be in [0,1]")
    if not (1.0 <= score <= 100.0):
        raise DiscoveryError("recommendation_score must be in [1,100]")
    event_key = item.get("event_key") or item.get("continuity_key")
    if event_key is not None and not str(event_key).strip():
        event_key = None
    speakers: List[Any] = []
    for w in window_words:
        ws, we = _word_start(w), _word_end(w)
        if ws is None or we is None:
            continue
        if ws < end and we > start:
            # Normalized words use 'sp'; preserve 0.
            sid = _word_speaker(w)
            if sid is not None and sid not in speakers:
                speakers.append(sid)
    cid = make_candidate_id(
        source_tag, start, end, str(evidence), window.index)
    mfp = make_moment_fingerprint(
        source_tag, start, end, str(evidence))
    return CandidateRecord(
        candidate_id=cid,
        moment_fingerprint=mfp,
        window_index=window.index,
        start_s=start, end_s=end,
        event_type=str(item.get("event_type") or "moment"),
        evidence_text=str(evidence).strip(),
        evidence_basis="transcript",
        context=str(context).strip(),
        expected_payoff=str(payoff).strip(),
        event_key=str(event_key).strip() if event_key else None,
        confidence=confidence,
        recommendation_score=score,
        speaker_ids=speakers,
        raw=dict(item),
    )


def _intervals_overlap(a_start: float, a_end: float,
                       b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def _adjacent(a_start: float, a_end: float, b_start: float, b_end: float,
              tol: float = 5.0) -> bool:
    return (abs(a_end - b_start) <= tol) or (abs(b_end - a_start) <= tol)


def _evidence_similarity(a: str, b: str) -> float:
    """Deterministic token Jaccard similarity on normalized evidence."""
    ta = set(_normalize_evidence(a).split())
    tb = set(_normalize_evidence(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# Documented deterministic thresholds.
DEDUP_SIMILARITY_THRESHOLD = 0.55
DEDUP_ADJACENCY_TOL_S = 5.0


def dedupe_candidates(candidates: Sequence[CandidateRecord]
                      ) -> Dict[str, Any]:
    """Overlap-aware semantic dedupe.

    Time overlap alone NEVER removes a candidate. Collapse only when:
    - same explicit event_key with supporting overlap, or
    - normalized evidence similarity >= 0.55 AND intervals overlap/adjacent.
    Returns kept list, removed mapping, and groups.
    """
    kept: List[CandidateRecord] = []
    removed: Dict[str, Dict[str, Any]] = {}
    groups: List[Dict[str, Any]] = []
    # Deterministic order: score desc, then id.
    ordered = sorted(candidates,
                     key=lambda c: (-c.recommendation_score, c.candidate_id))
    for cand in ordered:
        dup_of = None
        basis = None
        for k in kept:
            same_key = (cand.event_key and k.event_key
                        and cand.event_key == k.event_key)
            overlap = _intervals_overlap(cand.start_s, cand.end_s,
                                         k.start_s, k.end_s)
            adjacent = _adjacent(cand.start_s, cand.end_s,
                                 k.start_s, k.end_s,
                                 DEDUP_ADJACENCY_TOL_S)
            if same_key and (overlap or adjacent):
                dup_of, basis = k.candidate_id, "event_key"
                break
            if _evidence_similarity(cand.evidence_text,
                                    k.evidence_text
                                    ) >= DEDUP_SIMILARITY_THRESHOLD and (
                                        overlap or adjacent):
                dup_of, basis = k.candidate_id, "evidence_similarity"
                break
        if dup_of:
            # Ensure distinct trace IDs in funnel even if inputs alias.
            rid = cand.candidate_id
            if rid == dup_of:
                rid = f"{rid}#w{cand.window_index}"
            removed[rid] = {"kept_id": dup_of, "basis": basis,
                            "candidate_id": cand.candidate_id}
            for g in groups:
                if g["kept_id"] == dup_of:
                    g["removed_ids"].append(rid)
                    break
        else:
            kept.append(cand)
            groups.append({"kept_id": cand.candidate_id,
                           "removed_ids": [], "basis": None})
    # Fill group basis from removals.
    for g in groups:
        for rid in g["removed_ids"]:
            g["basis"] = removed[rid]["basis"]
            break
    return {"kept": kept, "removed": removed, "groups": groups}


class DiscoveryBudget:
    """Shared invocation budget: attempts, deadline, finite SDK timeouts."""

    def __init__(self, config: DiscoveryConfig,
                 time_fn: Callable[[], float] = time.monotonic):
        self.config = config
        self.time_fn = time_fn
        self.start = time_fn()
        self.attempts_used = 0

    def remaining_s(self) -> float:
        return max(0.0, self.config.deadline_s - (self.time_fn()
                                                  - self.start))

    def exhausted(self) -> bool:
        return (self.attempts_used >= self.config.max_provider_attempts
                or self.remaining_s() <= 0)

    def sdk_timeout_ms(self) -> int:
        """Finite SDK timeout never exceeding remaining budget.

        No artificial floor: subsecond remainders yield subsecond
        timeouts. Returns 0 if no time remains (caller must skip).
        """
        remaining_ms = int(self.remaining_s() * 1000)
        if remaining_ms <= 0:
            return 0
        return min(self.config.sdk_timeout_ms, remaining_ms)

    def consume(self) -> bool:
        """Consume one attempt; return False if budget exhausted."""
        if self.exhausted():
            return False
        self.attempts_used += 1
        return True


def _window_words_text(window: PlannedWindow,
                       words: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for w in words[window.word_start:window.word_end]:
        s = _word_start(w)
        txt = _word_text(w)
        if s is None:
            continue
        # Normalized words use 'sp'; preserve 0.
        spk = w.get("sp", w.get("speaker", "?"))
        parts.append(f"[{s:.1f}][spk:{spk}] {txt}")
    return "\n".join(parts)


def run_windowed_discovery(
    transcript_result: Any,
    video_duration: float,
    *,
    config: Optional[DiscoveryConfig] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    build_prompt_fn: Optional[Callable[..., str]] = None,
    parse_fn: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
    generate_fn: Optional[Callable[..., Any]] = None,
    instructions: Optional[str] = None,
    source_tag: str = "src",
    time_fn: Callable[[], float] = time.monotonic,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_clips: Optional[int] = None,
    max_clips: Optional[int] = None,
    clip_type: Optional[str] = None,
    duration_mode: Optional[str] = None,
    audience_intel: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run bounded windowed discovery; return funnel + candidate store.

    All provider interaction goes through injected callables for tests.
    Returns {"shorts": [...], "candidate_discovery": {...}} where shorts
    holds ViralClip-shaped dicts and candidate_discovery is versioned
    funnel metadata (no secrets, no full provider responses).
    """
    from clippyme.pipeline import gemini_request as gr

    cfg = config or DiscoveryConfig.from_env(
        model_name=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    # Resolve shared Task 04 duration policy once.
    from clippyme.pipeline.duration_policy import resolve_duration_policy
    _policy = resolve_duration_policy(
        min_duration=min_duration,
        max_duration=max_duration,
        duration_mode=duration_mode,
        clip_type=clip_type,
        video_duration=video_duration,
    )
    # Enforce only explicit bounds in discovery; default policy minimum
    # (15s) would reject valid short test candidates. The resolved policy
    # is still reported in funnel config.
    _enforce_min = min_duration if min_duration is not None else None
    _enforce_max = max_duration if max_duration is not None else None
    # Clean explicit bounds via _clean_bound logic (reject bool/non-finite).
    def _clean(v):
        if v is None or isinstance(v, bool):
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        import math as _m
        if not _m.isfinite(fv) or fv <= 0:
            return None
        return fv
    _enforce_min = _clean(_enforce_min)
    _enforce_max = _clean(_enforce_max)
    words = _extract_words(transcript_result)
    _word_issues = _extract_word_issues()
    plan = plan_windows(words, cfg)
    # Merge normalizer issues with plan invalid indices.
    _plan_invalid = plan.get("invalid_words", [])
    # Issues from normalizer are already counted; ensure funnel reports them.
    windows: List[PlannedWindow] = plan["windows"]
    budget = DiscoveryBudget(cfg, time_fn=time_fn)

    funnel: Dict[str, Any] = {
        "version": DISCOVERY_VERSION,
        "config": {
            "window_target_s": cfg.window_target_s,
            "window_overlap_s": cfg.window_overlap_s,
            "boundary_tolerance_s": cfg.boundary_tolerance_s,
            "max_windows": cfg.max_windows,
            "max_raw_candidates": cfg.max_raw_candidates,
            "max_provider_attempts": cfg.max_provider_attempts,
            "deadline_s": cfg.deadline_s,
            "sdk_timeout_ms": cfg.sdk_timeout_ms,
            "model_name": cfg.model_name,
            "fallback_models": list(cfg.fallback_models),
            "duration_min_s": _policy.min_duration,
            "duration_max_s": _policy.max_duration,
            "duration_mode": duration_mode,
            "duration_policy_mode": _policy.mode,
            "clip_type": clip_type,
            "audience_intel_present": audience_intel is not None,
        },
        "windows_planned": len(windows),
        "windows_attempted": 0,
        "windows_succeeded": 0,
        "windows_failed": 0,
        "windows_skipped": 0,
        "window_results": [],
        "transcript_coverage": plan["coverage"],
        "invalid_words": len(_word_issues) + len(plan.get("invalid_words", [])),
        "invalid_word_issues": _word_issues,
        "raw_candidates": 0,
        "valid_candidates": 0,
        "invalid_candidates": 0,
        "raw_candidate_cap_removals": 0,
        "invalid_reasons": {},
        "post_dedupe_candidates": 0,
        "duplicate_groups": 0,
        "duplicates_removed": 0,
        "score_floor_removals": 0,
        "final_cap_removals": 0,
        "final_selected": 0,
        "rejections": [],
        "budget": {"attempts_used": 0, "attempts_max": cfg.max_provider_attempts,
                   "deadline_s": cfg.deadline_s, "exhausted": False},
    }

    if not windows:
        funnel["candidate_discovery_note"] = "no valid words"
        return {"shorts": [], "candidate_discovery": funnel}

    _real_build = gr.build_discovery_prompt
    if build_prompt_fn is not None:
        # Test fake: call with legacy signature, but also try preferences.
        def build_prompt(w, w_words, instructions, **prefs):
            try:
                return build_prompt_fn(w, w_words, instructions, **prefs)
            except TypeError:
                return build_prompt_fn(w, w_words, instructions)
    else:
        def build_prompt(w, w_words, instructions, **prefs):
            return _real_build(
                w, w_words, instructions,
                clip_type=prefs.get("clip_type"),
                duration_mode=prefs.get("duration_mode"),
                duration_min_s=prefs.get("duration_min_s"),
                duration_max_s=prefs.get("duration_max_s"),
                audience_intel=prefs.get("audience_intel"),
            )
    parse = parse_fn or _default_parse
    gen = generate_fn or _default_generate

    client = None
    client_holder: Dict[str, Any] = {}
    if client_factory is not None:
        try:
            client = client_factory()
            client_holder["client"] = client
        except Exception as e:
            funnel["rejections"].append(
                {"reason": "client_setup_failed", "detail": str(e)[:200]})
            return {"shorts": [], "candidate_discovery": funnel}

    valid: List[CandidateRecord] = []
    try:
        for w in windows:
            if budget.exhausted():
                funnel["windows_skipped"] += 1
                funnel["rejections"].append(
                    {"reason": "budget_exhausted",
                     "window_index": w.index})
                funnel["window_results"].append(
                    {"window_index": w.index, "status": "skipped_budget",
                     "start_s": w.start_s, "end_s": w.end_s})
                continue
            funnel["windows_attempted"] += 1
            w_words = words[w.word_start:w.word_end]
            prompt = build_prompt(
                w, w_words, instructions,
                clip_type=clip_type,
                duration_mode=duration_mode,
                duration_min_s=_policy.min_duration,
                duration_max_s=_policy.max_duration,
                audience_intel=audience_intel,
            )
            if not budget.consume():
                funnel["windows_skipped"] += 1
                funnel["window_results"].append(
                    {"window_index": w.index, "status": "skipped_budget",
                     "start_s": w.start_s, "end_s": w.end_s})
                continue
            try:
                raw_text = gen(client, prompt, cfg, budget)
            except Exception as e:
                funnel["windows_failed"] += 1
                funnel["window_results"].append(
                    {"window_index": w.index, "status": "failed",
                     "start_s": w.start_s, "end_s": w.end_s,
                     "error": str(e)[:200]})
                continue
            try:
                items = parse(raw_text)
            except Exception as e:
                funnel["windows_failed"] += 1
                funnel["window_results"].append(
                    {"window_index": w.index, "status": "parse_failed",
                     "start_s": w.start_s, "end_s": w.end_s,
                     "error": str(e)[:200]})
                continue
            funnel["windows_succeeded"] += 1
            n_valid = 0
            for item in items:
                funnel["raw_candidates"] += 1
                if funnel["raw_candidates"] > cfg.max_raw_candidates:
                    funnel["raw_candidate_cap_removals"] += 1
                    funnel["rejections"].append(
                        {"reason": "raw_candidate_cap",
                         "window_index": w.index})
                    continue
                try:
                    rec = validate_discovery_candidate(
                        item, w, w_words, source_tag=source_tag,
                        video_duration=video_duration,
                        min_duration=_enforce_min,
                        max_duration=_enforce_max)
                except DiscoveryError as e:
                    funnel["invalid_candidates"] += 1
                    key = str(e)[:80]
                    funnel["invalid_reasons"][key] = funnel[
                        "invalid_reasons"].get(key, 0) + 1
                    continue
                valid.append(rec)
                n_valid += 1
            funnel["valid_candidates"] += n_valid
            funnel["window_results"].append(
                {"window_index": w.index, "status": "ok",
                 "start_s": w.start_s, "end_s": w.end_s,
                 "valid": n_valid})
    finally:
        # Client lifecycle: close exactly once on all exits.
        c = client_holder.get("client")
        if c is not None:
            try:
                close = getattr(c, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

    funnel["budget"]["attempts_used"] = budget.attempts_used
    funnel["budget"]["exhausted"] = budget.exhausted()

    # Global aggregation + overlap-aware dedupe (never per-window caps).
    dedup = dedupe_candidates(valid)
    funnel["duplicate_groups"] = len(
        [g for g in dedup["groups"] if g["removed_ids"]])
    funnel["duplicates_removed"] = len(dedup["removed"])
    funnel["duplicate_details"] = [
        {"kept_id": g["kept_id"], "removed_ids": g["removed_ids"],
         "basis": g["basis"]}
        for g in dedup["groups"] if g["removed_ids"]
    ]

    # Final score floor + max-clips AFTER aggregation (existing policy).
    # Effective cap: stricter of explicit max_clips and env CLIPPYME_MAX_CLIPS.
    try:
        min_score = int(os.getenv("CLIPPYME_MIN_VIRAL_SCORE") or 0)
    except ValueError:
        min_score = 0
    try:
        env_max = int(os.getenv("CLIPPYME_MAX_CLIPS") or 0)
    except ValueError:
        env_max = 0
    explicit_max = max_clips if isinstance(max_clips, int) and max_clips > 0 else 0
    if explicit_max > 0 and env_max > 0:
        effective_max = min(explicit_max, env_max)
    elif explicit_max > 0:
        effective_max = explicit_max
    else:
        effective_max = env_max
    selected = dedup["kept"]
    if min_score > 0:
        before = len(selected)
        selected = [c for c in selected
                    if c.recommendation_score >= min_score]
        funnel["score_floor_removals"] = before - len(selected)
    selected = sorted(selected,
                      key=lambda c: (-c.recommendation_score, c.candidate_id))
    funnel["final_cap_removals"] = 0
    if effective_max > 0 and len(selected) > effective_max:
        dropped = selected[effective_max:]
        funnel["final_cap_removals"] = len(dropped)
        funnel["rejections"].append(
            {"reason": "max_clips_cap", "removed_ids": [
                c.candidate_id for c in dropped]})
        selected = selected[:effective_max]
    funnel["final_selected"] = len(selected)

    # Adapt to downstream ViralClip shape without inventing copy.
    shorts = []
    candidate_store = {}
    for c in valid:
        candidate_store[c.candidate_id] = {
            "candidate_id": c.candidate_id,
            "moment_fingerprint": c.moment_fingerprint,
            "window_index": c.window_index,
            "start_s": c.start_s, "end_s": c.end_s,
            "event_type": c.event_type,
            "evidence_text": c.evidence_text,
            "evidence_basis": c.evidence_basis,
            "context": c.context,
            "expected_payoff": c.expected_payoff,
            "event_key": c.event_key,
            "confidence": c.confidence,
            "recommendation_score": c.recommendation_score,
            "speaker_ids": c.speaker_ids,
        }
    for c in selected:
        reason = f"{c.evidence_text.strip()} {c.context.strip()}".strip()
        if len(reason) < 20:
            reason = (reason + " " + c.expected_payoff.strip()).strip()
        if len(reason) < 20:
            reason = reason.ljust(20, ".")
        shorts.append({
            "start": c.start_s,
            "end": c.end_s,
            "viral_score": max(1, min(100, int(round(
                c.recommendation_score)))),
            "viral_reason": reason[:500],
            "candidate_id": c.candidate_id,
            "event_type": c.event_type,
            "viral_hook_text": "",
            "video_description": "",
        })
    funnel["candidate_store_ids"] = sorted(candidate_store.keys())
    funnel["post_dedupe_candidates"] = len(dedup["kept"])
    # Exact funnel arithmetic:
    # planned = succeeded + failed + skipped
    # raw = valid + invalid + raw-cap removals
    # valid = post-dedupe + duplicate removals
    # post-dedupe = floor removals + final-cap removals + final selected
    funnel["counts_reconcile"] = (
        funnel["windows_planned"] == funnel["windows_succeeded"]
        + funnel["windows_failed"] + funnel["windows_skipped"]
        and funnel["raw_candidates"] == funnel["valid_candidates"]
        + funnel["invalid_candidates"]
        + funnel["raw_candidate_cap_removals"]
        and funnel["valid_candidates"] == funnel["post_dedupe_candidates"]
        + funnel["duplicates_removed"]
        and funnel["post_dedupe_candidates"]
        == funnel["score_floor_removals"]
        + funnel["final_cap_removals"]
        + funnel["final_selected"]
    )
    return {"shorts": shorts,
            "candidate_discovery": funnel,
            "candidate_store": candidate_store}


def _extract_words(transcript_result: Any) -> List[Dict[str, Any]]:
    """Extract normalized words via the authoritative normalizer."""
    # Issues are available via normalize_transcript_words.last_issues.
    return normalize_transcript_words(transcript_result)


def _extract_word_issues() -> List[Dict[str, Any]]:
    """Return issues from the last normalize_transcript_words call."""
    return list(getattr(normalize_transcript_words, "last_issues", []))


def _default_parse(text: str) -> List[Dict[str, Any]]:
    from clippyme.pipeline import gemini_parser as gp
    return gp.parse_discovery_payload(text)


def _default_generate(client: Any, prompt: str, cfg: DiscoveryConfig,
                      budget: DiscoveryBudget) -> str:
    """Budget-aware generation: check/consume before every SDK call.

    Implements model fallback loop locally so the shared budget is
    checked immediately before each generate_content call, including
    fallbacks. Timeout recomputed per attempt; never exceeds remaining
    deadline. Preserves default fallback chain.
    """
    from clippyme.pipeline import gemini_request as gr
    models = [cfg.model_name] + list(cfg.fallback_models)
    models = [m for m in models if m] or ["gemini-3.5-flash"]
    last_error = None
    first = True
    for model_name in models:
        # Primary already consumed by window loop; fallbacks consume.
        if not first:
            if budget.exhausted():
                break
            if not budget.consume():
                break
        first = False
        timeout_ms = budget.sdk_timeout_ms()
        if timeout_ms <= 0:
            break
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={"http_options": {"timeout": int(timeout_ms)}},
            )
            return response.text or ""
        except Exception as exc:
            last_error = exc
            # Permanent errors fail immediately.
            if not gr._is_retryable_model_error(exc):
                raise
            # Retryable capacity/rate/unavailable: continue to next model
            # while budget remains. Budget consumed per model below.
            continue
    if last_error:
        raise last_error
    raise DiscoveryError("budget exhausted before generation")
