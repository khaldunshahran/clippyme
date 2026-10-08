"""Grounded editorial refinement (Task 09).

Second bounded half of transcript-based Milestone 2. After windowed
discovery (Task 08) produces deduplicated candidates, this pass builds a
bounded transcript-context packet per candidate and requests ONE
validated edit decision per candidate: ``accept``, ``adjust``
(word-aligned), or ``reject``.

Grounding rules (validated independently of the provider):
- quote evidence must be verbatim contiguous transcript slices inside the
  proposed interval; fabricated quotes, invented names, and
  visual/audio evidence claims are rejected;
- proposed intervals stay inside the packet context and the source,
  overlap the original candidate, satisfy the resolved Task 04 duration
  policy, cover real words, and align to word boundaries;
- adjustments may expand or trim but never delete essential evidence,
  end before the grounded payoff, cut through a word, or steal a
  neighboring candidate's interval (explicit guard);
- duration conflicts keep the discovery interval as ``unrefined`` with a
  structured conflict instead of greedy truncation;
- continuity merges on the quality path never use the legacy
  entity/title heuristics; they require explicit candidate IDs,
  grounded bridge evidence from the intervening transcript, compatible
  event identity, source order, a bounded gap, and a policy-compliant
  merged interval.

All provider calls go through injected factories/callables so tests use
deterministic fakes; no live SDK calls happen in this module's tests.
"""
from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from clippyme.pipeline.candidate_discovery import (
    _env_flag,
    _env_float,
    _env_int,
    _is_finite_number,
    _normalize_evidence,
    _word_end,
    _word_speaker,
    _word_start,
    _word_text,
)

EDITORIAL_VERSION = "1"

EDIT_CTX_WORDS = 60
EDIT_CTX_SECS = 90.0
EDIT_MAX_ATTEMPTS = 24
EDIT_DEADLINE_S = 600.0
EDIT_SDK_TIMEOUT_MS = 120000

# Word-boundary alignment tolerance (seconds). A proposed boundary must
# sit within this distance of a word edge; sitting strictly inside a
# word beyond this tolerance is "cutting through a word".
WORD_ALIGN_TOL_S = 0.5
# Maximum transcript gap between continuity-linked candidates.
CONTINUITY_MAX_GAP_S = 120.0

QUALITY_FIELDS = [
    "opening_strength",
    "context_sufficiency",
    "payoff_delivery",
    "useful_value",
    "entertaining_value",
    "transcript_support",
    "editability",
    "confidence",
]

QUOTE_FIELDS = (
    "opening_evidence",
    "premise_evidence",
    "payoff_evidence",
    "ending_evidence",
)


class EditorialError(Exception):
    pass


def editorial_refinement_enabled() -> bool:
    """Rollback flag for the Task 09 quality path; default ON."""
    return _env_flag("CLIPPYME_EDITORIAL_REFINEMENT", True)


@dataclass
class EditorialConfig:
    """Validated, bounded editorial configuration (env-driven)."""

    context_words: int = EDIT_CTX_WORDS
    context_seconds: float = EDIT_CTX_SECS
    max_provider_attempts: int = EDIT_MAX_ATTEMPTS
    deadline_s: float = EDIT_DEADLINE_S
    sdk_timeout_ms: int = EDIT_SDK_TIMEOUT_MS
    model_name: str = ""
    fallback_models: tuple = ()

    @classmethod
    def from_env(cls, model_name: str = "",
                 fallback_models: Optional[Sequence[str]] = None,
                 ) -> "EditorialConfig":
        # Preserve established default chain when env override is unset.
        if fallback_models is None:
            from clippyme.pipeline.gemini_request import build_model_chain
            _chain = build_model_chain(model_name or "gemini-3.5-flash",
                                       os.getenv("GEMINI_FALLBACK_MODELS"))
            # build_model_chain includes primary first; fallbacks are rest.
            fallback_models = [m for m in _chain if m != model_name]
        return cls(
            context_words=_env_int("CLIPPYME_EDIT_CTX_WORDS",
                                   EDIT_CTX_WORDS, 0, 400),
            context_seconds=_env_float("CLIPPYME_EDIT_CTX_SECS",
                                       EDIT_CTX_SECS, 0.0, 600.0),
            max_provider_attempts=_env_int("CLIPPYME_EDIT_MAX_ATTEMPTS",
                                           EDIT_MAX_ATTEMPTS, 1, 200),
            deadline_s=_env_float("CLIPPYME_EDIT_DEADLINE_S",
                                  EDIT_DEADLINE_S, 10.0, 3600.0),
            sdk_timeout_ms=_env_int("CLIPPYME_EDIT_SDK_TIMEOUT_MS",
                                    EDIT_SDK_TIMEOUT_MS, 5000, 600000),
            model_name=model_name,
            fallback_models=tuple(fallback_models or ()),
        )


class EditorialBudget:
    """Shared invocation budget: attempts, deadline, finite SDK timeouts."""

    def __init__(self, config: EditorialConfig,
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
        """Finite SDK timeout never exceeding remaining budget."""
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


def _packet_word(i: int, words: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    w = words[i]
    d: Dict[str, Any] = {"t": _word_text(w),
                         "s": _word_start(w), "e": _word_end(w)}
    sp = _word_speaker(w)
    if sp is not None:
        d["sp"] = sp  # preserve speaker 0
    return d


def _valid_word(w: Any) -> bool:
    return (isinstance(w, dict) and bool(_word_text(w).strip())
            and _word_start(w) is not None and _word_end(w) is not None)


def build_context_packet(candidate: Mapping[str, Any],
                         all_candidates: Sequence[Mapping[str, Any]],
                         words: Sequence[Dict[str, Any]],
                         video_duration: float,
                         policy_min: float, policy_max: float,
                         config: EditorialConfig) -> Dict[str, Any]:
    """Build a deterministic bounded context packet for one candidate.

    The packet carries the candidate interval plus bounded surrounding
    transcript context. All timestamps are ABSOLUTE (never rebased).
    Context is bounded by ``context_words`` per side and additionally by
    ``context_seconds`` per side, clipped to ``[0, video_duration]``.
    ``context_before_available`` / ``context_after_available`` report
    honestly when the source edge (not the budget) truncates a side.
    """
    cid = candidate["candidate_id"]
    start_s = float(candidate["start_s"])
    end_s = float(candidate["end_s"])

    iv = [i for i, w in enumerate(words)
          if _valid_word(w) and _word_start(w) < end_s
          and _word_end(w) > start_s]
    before: List[int] = []
    after: List[int] = []
    if iv:
        i = iv[0] - 1
        while i >= 0 and len(before) < config.context_words:
            w = words[i]
            if not _valid_word(w):
                i -= 1
                continue
            if start_s - _word_start(w) > config.context_seconds:
                break
            before.append(i)
            i -= 1
        before.reverse()
        i = iv[-1] + 1
        while i < len(words) and len(after) < config.context_words:
            w = words[i]
            if not _valid_word(w):
                i += 1
                continue
            if _word_end(w) - end_s > config.context_seconds:
                break
            after.append(i)
            i += 1

    ctx_idxs = before + iv + after
    if ctx_idxs:
        ctx_start = min(
            start_s, min(_word_start(words[i]) for i in ctx_idxs))
        ctx_end = max(
            end_s, max(_word_end(words[i]) for i in ctx_idxs))
    else:
        ctx_start, ctx_end = start_s, end_s
    ctx_start = max(0.0, ctx_start)
    if video_duration and video_duration > 0:
        ctx_end = min(float(video_duration), ctx_end)

    # Availability is False only when the SOURCE edge truncates the side
    # (no words exist there); budget-bounded sides still report True.
    before_available = any(
        _valid_word(w) and _word_end(w) <= start_s for w in words)
    after_available = any(
        _valid_word(w) and _word_start(w) >= end_s for w in words)

    others = [c for c in (all_candidates or [])
              if c.get("candidate_id") != cid]
    adj_before = sorted(
        (c for c in others if float(c["end_s"]) <= start_s),
        key=lambda c: float(c["end_s"]))[-2:]
    adj_after = sorted(
        (c for c in others if float(c["start_s"]) >= end_s),
        key=lambda c: float(c["start_s"]))[:2]

    # Deterministic key order; timestamps absolute, never rebased.
    return {
        "candidate_id": cid,
        "moment_fingerprint": candidate.get("moment_fingerprint", ""),
        "start_s": start_s,
        "end_s": end_s,
        "context_start_s": ctx_start,
        "context_end_s": ctx_end,
        "context_before_available": bool(before_available),
        "context_after_available": bool(after_available),
        "context_before": [_packet_word(i, words) for i in before],
        "candidate_words": [_packet_word(i, words) for i in iv],
        "context_after": [_packet_word(i, words) for i in after],
        "evidence_text": candidate.get("evidence_text", ""),
        "evidence_context": candidate.get("context", ""),
        "expected_payoff": candidate.get("expected_payoff", ""),
        "event_key": candidate.get("event_key"),
        "duration_min_s": policy_min,
        "duration_max_s": policy_max,
        "adjacent_before": [c["candidate_id"] for c in adj_before],
        "adjacent_after": [c["candidate_id"] for c in adj_after],
        "speaker_ids": list(candidate.get("speaker_ids") or []),
        "evidence_disclosure": "transcript-only",
    }


@dataclass
class EditorialPlan:
    """Validated per-candidate editorial plan (JSON-serializable)."""

    candidate_id: str
    verdict: str  # accept | adjust | reject | unrefined
    start_s: Optional[float]
    end_s: Optional[float]
    quotes: Dict[str, str] = field(default_factory=dict)
    quality: Dict[str, Optional[float]] = field(default_factory=dict)
    recommendation_score: Optional[float] = None
    reason: str = ""
    continuity_links: List[str] = field(default_factory=list)
    bridge_evidence: Optional[str] = None
    status: str = "refined"  # refined | unrefined
    conflict: Optional[str] = None
    moment_fingerprint: str = ""
    continuity_group: Optional[str] = None
    components: List[str] = field(default_factory=list)
    plan_version: str = EDITORIAL_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_version": EDITORIAL_VERSION,
            "candidate_id": self.candidate_id,
            "verdict": self.verdict,
            "status": self.status,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "quotes": dict(self.quotes),
            "quality": dict(self.quality),
            "recommendation_score": self.recommendation_score,
            "reason": self.reason,
            "continuity_links": list(self.continuity_links),
            "bridge_evidence": self.bridge_evidence,
            "conflict": self.conflict,
            "moment_fingerprint": self.moment_fingerprint,
            "continuity_group": self.continuity_group,
            "components": list(self.components),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EditorialPlan":
        return cls(
            candidate_id=d.get("candidate_id", ""),
            verdict=d.get("verdict", "unrefined"),
            start_s=d.get("start_s"),
            end_s=d.get("end_s"),
            quotes=dict(d.get("quotes") or {}),
            quality=dict(d.get("quality") or {}),
            recommendation_score=d.get("recommendation_score"),
            reason=d.get("reason") or "",
            continuity_links=list(d.get("continuity_links") or []),
            bridge_evidence=d.get("bridge_evidence"),
            status=d.get("status") or "unrefined",
            conflict=d.get("conflict"),
            moment_fingerprint=d.get("moment_fingerprint") or "",
            continuity_group=d.get("continuity_group"),
            components=list(d.get("components") or []),
        )


def _validate_prior_plan(raw: Any, candidate: Mapping[str, Any],
                         video_duration: float,
                         policy_min: float, policy_max: float
                         ) -> Optional[str]:
    """Validate a persisted prior plan against the current candidate.

    Returns None when the plan may be reused as completed work, else a
    short reason. Stale identity is never trusted: the embedded
    candidate ID and moment fingerprint must match the current
    candidate, the schema version must be current, and boundaries,
    verdict/status, and duration/source bounds must still hold.
    """
    if not isinstance(raw, Mapping):
        return "not an object"
    if raw.get("plan_version") != EDITORIAL_VERSION:
        return "schema version mismatch"
    if raw.get("candidate_id") != candidate.get("candidate_id"):
        return "candidate_id mismatch"
    if (raw.get("moment_fingerprint") or "") != (
            candidate.get("moment_fingerprint") or ""):
        return "moment_fingerprint mismatch"
    if raw.get("verdict") not in ("accept", "adjust", "reject",
                                  "unrefined"):
        return "unknown verdict"
    if raw.get("status") not in ("refined", "unrefined"):
        return "unknown status"
    ss, se = raw.get("start_s"), raw.get("end_s")
    if not _is_finite_number(ss) or not _is_finite_number(se):
        return "non-finite boundaries"
    ss, se = float(ss), float(se)
    if not ss < se:
        return "reversed boundaries"
    if video_duration and video_duration > 0:
        if not (0 <= ss and se <= video_duration + 1e-6):
            return "outside source bounds"
    dur = se - ss
    if dur < policy_min - 1e-6 or dur > policy_max + 1e-6:
        return "duration policy violation"
    return None


def _blank_quality() -> Dict[str, Optional[float]]:
    return {f: None for f in QUALITY_FIELDS}


def _validate_quality(q: Any) -> Dict[str, Optional[float]]:
    """Bounded quality vector: each field in [0,1] or None.

    Unknown stays unknown: missing, boolean, non-finite, or
    out-of-range values become None rather than failing the decision.
    """
    out = _blank_quality()
    if not isinstance(q, dict):
        return out
    for f in QUALITY_FIELDS:
        v = q.get(f)
        if v is None or isinstance(v, bool) or not _is_finite_number(v):
            continue
        fv = float(v)
        if 0.0 <= fv <= 1.0:
            out[f] = fv
    return out


def compute_editorial_score(quality: Optional[Mapping[str, Any]]
                            ) -> Optional[float]:
    """Editorial recommendation score: mean of validated present fields.

    Returns mean(present fields) * 100 rounded to 1 decimal, or None
    when no field validated. This is an editorial sorting estimate;
    it is never a virality probability or measured audience response.
    """
    vals: List[float] = []
    for v in (quality or {}).values():
        if v is None or isinstance(v, bool):
            continue
        if _is_finite_number(v):
            vals.append(float(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals) * 100, 1)


def _ground_quote_in_interval(quote: str, words: Sequence[Dict[str, Any]],
                              start: float, end: float
                              ) -> Optional[tuple]:
    """Contiguous normalized token-slice match inside [start, end).

    Returns the matched word span (span_start, span_end) or None.
    Character substrings inside larger tokens do NOT count; complete
    normalized tokens only (mirrors discovery's contiguous rule).
    """
    qtokens = _normalize_evidence(quote).split()
    if not qtokens:
        return None
    idx_tokens: List[tuple] = []  # (word_index, token)
    for i, w in enumerate(words):
        if not _valid_word(w):
            continue
        if _word_start(w) < end and _word_end(w) > start:
            for tok in _normalize_evidence(_word_text(w)).split():
                idx_tokens.append((i, tok))
    toks = [t for _, t in idx_tokens]
    n = len(qtokens)
    for j in range(len(toks) - n + 1):
        if toks[j:j + n] == qtokens:
            wi0 = idx_tokens[j][0]
            wi1 = idx_tokens[j + n - 1][0]
            return (_word_start(words[wi0]), _word_end(words[wi1]))
    return None


def _check_no_invented_names(quote: str, words: Sequence[Dict[str, Any]],
                             start: float, end: float, field: str) -> None:
    """Reject capitalized tokens in quotes absent from the interval words.

    Simple and deterministic: a title-cased token in a quote must appear
    verbatim (case-sensitive) among the raw interval word tokens;
    otherwise the provider invented a name. Grounding is
    case-insensitive; this check keeps the verbatim-case honest.
    """
    interval_tokens = set()
    for w in words:
        if not _valid_word(w):
            continue
        if _word_start(w) < end and _word_end(w) > start:
            for raw in re.findall(r"[A-Za-z][A-Za-z']*",
                                  _word_text(w)):
                interval_tokens.add(raw.strip("'"))
    for raw in re.findall(r"[A-Za-z][A-Za-z']*", quote):
        tok = raw.strip("'")
        if len(tok) > 1 and tok[0].isupper() and tok[1:].islower():
            if tok not in interval_tokens:
                raise EditorialError(f"invented name in {field}: {tok}")


def _snap_boundary(p: float, words: Sequence[Dict[str, Any]],
                   is_start: bool) -> float:
    """Snap a proposed boundary to the nearest exact word edge.

    The contract tolerates recognizing the intended edge within
    WORD_ALIGN_TOL_S, but the persisted plan must never store a
    mid-word fractional value: the boundary is stored as the exact
    edge. Anything farther than tolerance from every edge (or
    strictly inside a word beyond tolerance from both edges) raises
    EditorialError.
    """
    kind = "start" if is_start else "end"
    edges: List[float] = []
    for w in words:
        if not _valid_word(w):
            continue
        s, e = _word_start(w), _word_end(w)
        edges.append(s)
        edges.append(e)
        if s < p < e and min(p - s, e - p) > WORD_ALIGN_TOL_S:
            raise EditorialError("proposed boundary cuts through a word")
    if not edges:
        raise EditorialError("proposed interval covers no words")
    best = min(edges, key=lambda v: abs(v - p))
    if abs(best - p) > WORD_ALIGN_TOL_S + 1e-9:
        raise EditorialError(f"proposed {kind} not word-aligned")
    return best


def _unrefined_plan(candidate: Mapping[str, Any],
                    quality: Dict[str, Optional[float]],
                    score: Optional[float],
                    reason: str,
                    conflict: Optional[str] = None) -> EditorialPlan:
    """Keep the discovery interval explicitly unrefined (never approved)."""
    return EditorialPlan(
        candidate_id=candidate["candidate_id"],
        verdict="unrefined",
        start_s=float(candidate["start_s"]),
        end_s=float(candidate["end_s"]),
        quotes={},
        quality=quality,
        recommendation_score=score,
        reason=reason,
        continuity_links=[],
        bridge_evidence=None,
        status="unrefined",
        conflict=conflict,
        moment_fingerprint=candidate.get("moment_fingerprint", ""),
    )


def _passes_selection_gates(
        quality: Mapping[str, Optional[float]]) -> bool:
    """Context/payoff gates for every refined selected edit.

    Applies to accept AND adjust: a strong opening cannot hide an
    incomplete clip, and continuity may only use plans that passed.
    """
    cs = quality.get("context_sufficiency")
    pd = quality.get("payoff_delivery")
    return (cs is not None and cs >= 0.5
            and pd is not None and pd >= 0.5)


def validate_editorial_decision(
    decision: Any,
    packet: Mapping[str, Any],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    words: Sequence[Dict[str, Any]],
    video_duration: float,
    policy_min: float,
    policy_max: float,
    neighbor_guards: Sequence[Sequence[float]],
) -> EditorialPlan:
    """Validate one provider decision independently; raise EditorialError.

    Returns a refined EditorialPlan for accept/adjust/reject, or an
    ``unrefined`` plan (kept discovery interval, never editor-approved)
    for duration conflicts and failed accept gates. One malformed
    decision raises without affecting other candidates (the caller
    isolates it per candidate).
    """
    if not isinstance(decision, dict):
        raise EditorialError("decision must be an object")
    cid = decision.get("candidate_id")
    if not isinstance(cid, str) or cid not in candidates_by_id:
        raise EditorialError(f"unknown candidate_id: {cid!r}")
    if cid != packet.get("candidate_id"):
        raise EditorialError("candidate_id does not match packet")
    verdict = decision.get("verdict")
    if verdict not in ("accept", "adjust", "reject"):
        raise EditorialError(f"invalid verdict: {verdict!r}")

    candidate = candidates_by_id[cid]
    orig_start = float(candidate["start_s"])
    orig_end = float(candidate["end_s"])
    # Packet timestamps are absolute and never rebased: they must equal
    # the candidate interval exactly.
    if (abs(float(packet.get("start_s", orig_start)) - orig_start) > 1e-6
            or abs(float(packet.get("end_s", orig_end)) - orig_end) > 1e-6):
        raise EditorialError(
            "packet timestamps do not match candidate interval")
    quality = _validate_quality(decision.get("quality"))
    score = compute_editorial_score(quality)
    # Strict decision contract: a transcript-only evidence basis and an
    # explicit bounded reason are required for every verdict, including
    # reject. Nothing is manufactured on the provider's behalf.
    basis = decision.get("evidence_basis")
    if not isinstance(basis, str) or basis.strip().lower() != "transcript":
        raise EditorialError('evidence_basis must be "transcript"')
    reason = decision.get("decision_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise EditorialError("decision_reason must be a non-empty string")
    reason = reason.strip()[:500]
    links = decision.get("continuity_links") or []
    if not isinstance(links, list):
        raise EditorialError("continuity_links must be a list")
    links = [l for l in links if isinstance(l, str) and l]
    bridge = decision.get("bridge_evidence")
    if bridge is not None and not isinstance(bridge, str):
        raise EditorialError("bridge_evidence must be a string")

    # Transcript-only disclosure: applies to every verdict.
    basis = decision.get("evidence_basis")
    if basis is not None and str(basis).strip().lower() != "transcript":
        raise EditorialError('evidence_basis must be "transcript"')

    if verdict == "reject":
        # Honest rejection needs no interval checks beyond identity.
        return EditorialPlan(
            candidate_id=cid, verdict="reject",
            start_s=orig_start, end_s=orig_end,
            quotes={}, quality=quality, recommendation_score=score,
            reason=reason,
            continuity_links=[], bridge_evidence=None,
            status="refined",
            moment_fingerprint=candidate.get("moment_fingerprint", ""))

    ps = decision.get("proposed_start_s")
    pe = decision.get("proposed_end_s")
    if not _is_finite_number(ps) or not _is_finite_number(pe):
        raise EditorialError("proposed start/end must be finite numbers")
    ps, pe = float(ps), float(pe)
    if not ps < pe:
        raise EditorialError("proposed interval reversed or empty")
    ctx_start = float(packet["context_start_s"])
    ctx_end = float(packet["context_end_s"])
    if not (ctx_start - 1e-6 <= ps and pe <= ctx_end + 1e-6):
        raise EditorialError("proposed interval outside packet context")
    if video_duration and video_duration > 0:
        if not (0 <= ps and pe <= video_duration + 1e-6):
            raise EditorialError("proposed interval outside source duration")
    if not (ps < orig_end and orig_start < pe):
        raise EditorialError(
            "proposed interval does not overlap original candidate")
    if not any(_valid_word(w) and _word_start(w) < pe
               and _word_end(w) > ps for w in words):
        raise EditorialError("proposed interval covers no words")
    # Snap tolerated boundaries to exact word edges: the persisted plan
    # must never store a mid-word fractional value.
    ps = _snap_boundary(ps, words, is_start=True)
    pe = _snap_boundary(pe, words, is_start=False)
    if not ps < pe:
        raise EditorialError("proposed interval reversed or empty")
    # Re-run the spatial checks against the snapped interval (snapping
    # moves each boundary by at most the tolerance).
    if not (ctx_start - 1e-6 <= ps and pe <= ctx_end + 1e-6):
        raise EditorialError("proposed interval outside packet context")
    if video_duration and video_duration > 0:
        if not (0 <= ps and pe <= video_duration + 1e-6):
            raise EditorialError("proposed interval outside source duration")
    if not (ps < orig_end and orig_start < pe):
        raise EditorialError(
            "proposed interval does not overlap original candidate")
    if not any(_valid_word(w) and _word_start(w) < pe
               and _word_end(w) > ps for w in words):
        raise EditorialError("proposed interval covers no words")

    # Quote grounding before duration-conflict analysis (the story's
    # required span decides conflict vs. hard violation).
    quotes: Dict[str, str] = {}
    spans: Dict[str, tuple] = {}
    for f in QUOTE_FIELDS:
        q = decision.get(f)
        if not isinstance(q, str) or not q.strip():
            raise EditorialError(f"quote missing or empty: {f}")
        span = _ground_quote_in_interval(q, words, ps, pe)
        if span is None:
            raise EditorialError(f"quote not grounded: {f}")
        _check_no_invented_names(q, words, ps, pe, f)
        quotes[f] = q.strip()
        spans[f] = span

    dur = pe - ps
    req_start = min(s[0] for s in spans.values())
    req_end = max(s[1] for s in spans.values())
    if dur > policy_max + 1e-6:
        if req_end - req_start > policy_max + 1e-6:
            # The complete story cannot fit: keep discovery interval,
            # structured conflict, never greedy truncation.
            return _unrefined_plan(
                candidate, quality, score,
                reason=("duration_conflict: complete story exceeds maximum "
                        f"duration ({policy_max:.1f}s); kept discovery "
                        "interval"),
                conflict="duration_conflict")
        raise EditorialError(
            f"proposed duration {dur:.1f}s above maximum {policy_max:.1f}s")
    if dur < policy_min - 1e-6:
        max_reachable = ctx_end - ctx_start
        if video_duration and video_duration > 0:
            max_reachable = min(max_reachable, float(video_duration))
        if max_reachable < policy_min - 1e-6:
            return _unrefined_plan(
                candidate, quality, score,
                reason=("duration_conflict: complete story shorter than "
                        f"minimum duration ({policy_min:.1f}s); kept "
                        "discovery interval"),
                conflict="duration_conflict")
        raise EditorialError(
            f"proposed duration {dur:.1f}s below minimum {policy_min:.1f}s")

    if verdict == "accept":
        if abs(ps - orig_start) > 1e-6 or abs(pe - orig_end) > 1e-6:
            raise EditorialError(
                "accept verdict requires proposed interval to equal "
                "candidate interval")
        # Gates: a strong opening cannot hide an incomplete clip.
        if not _passes_selection_gates(quality):
            return _unrefined_plan(
                candidate, quality, score,
                reason="gate_failed: context/payoff incomplete")
        return EditorialPlan(
            candidate_id=cid, verdict="accept",
            start_s=ps, end_s=pe,
            quotes=quotes, quality=quality, recommendation_score=score,
            reason=reason,
            continuity_links=links,
            bridge_evidence=bridge.strip() if bridge and bridge.strip()
            else None,
            status="refined",
            moment_fingerprint=candidate.get("moment_fingerprint", ""))

    # verdict == "adjust": content-preserving rules. The selection
    # gates protect every refined edit, not just accept.
    if not _passes_selection_gates(quality):
        return _unrefined_plan(
            candidate, quality, score,
            reason="gate_failed: context/payoff incomplete")
    tol = WORD_ALIGN_TOL_S
    # Payoff first: ending before the grounded payoff gets the specific
    # reason (it implies the span check below, which stays for the other
    # fields).
    payoff_end = spans["payoff_evidence"][1]
    if pe < payoff_end - tol:
        raise EditorialError("ends before grounded payoff")
    for f, (ss, se) in spans.items():
        if not (ss >= ps - tol and se <= pe + tol):
            raise EditorialError(
                f"quote span outside proposed interval: {f}")
    for g in neighbor_guards or []:
        gs, ge = float(g[0]), float(g[1])
        if ps < ge and gs < pe:
            # Pre-existing overlap (kept by discovery dedupe) is not a
            # violation; only NEW overlap introduced by the adjustment
            # steals a neighbor's interval.
            if not (orig_start < ge and gs < orig_end):
                raise EditorialError("violates neighbor guard")
    return EditorialPlan(
        candidate_id=cid, verdict="adjust",
        start_s=ps, end_s=pe,
        quotes=quotes, quality=quality, recommendation_score=score,
        reason=reason,
        continuity_links=links,
        bridge_evidence=bridge.strip() if bridge and bridge.strip()
        else None,
        status="refined",
        moment_fingerprint=candidate.get("moment_fingerprint", ""))


def plan_continuity_group(
    link_ids: Sequence[str],
    plans_by_id: Mapping[str, EditorialPlan],
    candidates_by_id: Mapping[str, Mapping[str, Any]],
    words: Sequence[Dict[str, Any]],
    video_duration: float,
    policy_min: float,
    policy_max: float,
) -> EditorialPlan:
    """Merge explicitly linked candidates into one continuity plan.

    Never uses entity/title heuristics: requires valid distinct IDs,
    source order, a bounded gap, grounded bridge evidence from the
    intervening transcript, compatible event keys or moment identity,
    and a policy-compliant merged interval covering every component
    word and bridge word. Raises EditorialError otherwise.
    """
    ids = list(link_ids)
    if len(set(ids)) != len(ids):
        raise EditorialError("continuity: duplicate candidate id")
    if len(ids) < 2:
        raise EditorialError(
            "continuity: need at least two distinct candidates")
    for cid in ids:
        if cid not in candidates_by_id or cid not in plans_by_id:
            raise EditorialError(
                f"continuity: unknown candidate id {cid!r}")
    comps = [(cid, plans_by_id[cid]) for cid in ids]
    comps.sort(key=lambda t: (t[1].start_s, t[0]))

    keys = {candidates_by_id[cid].get("event_key") for cid, _ in comps}
    fps = {candidates_by_id[cid].get("moment_fingerprint")
           for cid, _ in comps}
    if not (len(keys) == 1 and next(iter(keys))):
        # Fall back to moment identity: identical fingerprint means the
        # same moment observed twice.
        if len(fps) != 1 or not next(iter(fps)):
            raise EditorialError("continuity: incompatible event identity")

    # Gaps and bridge evidence between consecutive components.
    for (cid_a, pa), (cid_b, pb) in zip(comps, comps[1:]):
        gap = pb.start_s - pa.end_s
        if gap > CONTINUITY_MAX_GAP_S + 1e-6:
            raise EditorialError("continuity: gap exceeds maximum "
                                 f"({CONTINUITY_MAX_GAP_S:.0f}s)")
        gap_words = [w for w in words
                     if _valid_word(w)
                     and _word_start(w) >= pa.end_s - 1e-6
                     and _word_end(w) <= pb.start_s + 1e-6]
        if not gap_words:
            continue  # adjacent: no intervening transcript to bridge
        bridge = (pa.bridge_evidence or "").strip() \
            or (pb.bridge_evidence or "").strip()
        if not bridge:
            raise EditorialError(
                "continuity: missing bridge evidence for intervening "
                "transcript")
        span = _ground_quote_in_interval(bridge, words,
                                         pa.end_s, pb.start_s)
        if span is None:
            raise EditorialError(
                "continuity: bridge evidence not grounded")
        ss, se = span
        if not (ss >= pa.end_s - 1e-6 and se <= pb.start_s + 1e-6):
            raise EditorialError(
                "continuity: bridge evidence outside intervening span")

    mstart = min(p.start_s for _, p in comps)
    mend = max(p.end_s for _, p in comps)
    mdur = mend - mstart
    if video_duration and video_duration > 0:
        if not (0 <= mstart and mend <= video_duration + 1e-6):
            raise EditorialError(
                "continuity: merged interval outside source duration")
    if mdur < policy_min - 1e-6 or mdur > policy_max + 1e-6:
        raise EditorialError(
            "continuity: merged interval violates duration policy")
    if not any(_valid_word(w) and _word_start(w) < mend
               and _word_end(w) > mstart for w in words):
        raise EditorialError("continuity: merged interval covers no words")

    # Revalidate every component quote inside the merged interval.
    merged_quotes: Dict[str, str] = {}
    for cid, p in comps:
        for f, q in p.quotes.items():
            if _ground_quote_in_interval(q, words, mstart, mend) is None:
                raise EditorialError(
                    "continuity: component quote not grounded in merged "
                    f"interval: {cid}/{f}")
            merged_quotes[f"{cid}::{f}"] = q

    # Recompute quality as the per-field mean (None if any None).
    quality: Dict[str, Optional[float]] = {}
    for f in QUALITY_FIELDS:
        vals = [p.quality.get(f) for _, p in comps]
        if all(v is not None for v in vals):
            quality[f] = round(sum(vals) / len(vals), 4)
        else:
            quality[f] = None
    score = compute_editorial_score(quality)

    gid = "group:" + "+".join(cid for cid, _ in comps)
    fp = next(iter(fps)) if len(fps) == 1 and next(iter(fps)) else ""
    return EditorialPlan(
        candidate_id=gid, verdict="adjust",
        start_s=mstart, end_s=mend,
        quotes=merged_quotes, quality=quality,
        recommendation_score=score,
        reason=f"continuity merge of {len(comps)} candidates",
        continuity_links=[], bridge_evidence=None,
        status="refined",
        moment_fingerprint=fp,
        components=[cid for cid, _ in comps])


def _plan_fingerprint(p: Any) -> str:
    if isinstance(p, EditorialPlan):
        return p.moment_fingerprint or ""
    if isinstance(p, Mapping):
        return p.get("moment_fingerprint") or ""
    return ""


def _plan_id(p: Any) -> str:
    if isinstance(p, EditorialPlan):
        return p.candidate_id
    if isinstance(p, Mapping):
        return p.get("candidate_id", "")
    return ""


def group_variants(plans: Sequence[Any]) -> Dict[str, Any]:
    """Group plans by moment fingerprint.

    Candidates sharing a non-empty fingerprint are variants of ONE
    distinct moment; they must not inflate the distinct-moment count.
    A missing fingerprint is unknown identity: each such plan stays
    its own distinct moment and is never collapsed under a shared
    empty key.
    Returns {"distinct_moments": int, "variants": {fp: [ids]}} where
    variants lists only fingerprints with >1 plan.
    """
    by_key: Dict[Any, List[str]] = {}
    for p in plans:
        fp = _plan_fingerprint(p)
        key = fp if fp else ("__unknown__", _plan_id(p))
        by_key.setdefault(key, []).append(_plan_id(p))
    return {
        "distinct_moments": len(by_key),
        "variants": {fp: ids for fp, ids in by_key.items()
                     if isinstance(fp, str) and len(ids) > 1},
    }


def disabled_editorial_record(reason: str) -> Dict[str, Any]:
    """Reconciled empty funnel for disabled/empty editorial runs."""
    return {
        "version": EDITORIAL_VERSION,
        "enabled": False,
        "reason": reason,
        "config": {
            "model_name": "",
            "fallback_models": [],
            "context_words": 0,
            "context_seconds": 0.0,
            "max_provider_attempts": 0,
            "deadline_s": 0.0,
            "sdk_timeout_ms": 0,
            "duration_min_s": 0.0,
            "duration_max_s": 0.0,
        },
        "packets_planned": 0,
        "packets_attempted": 0,
        "packets_succeeded": 0,
        "packets_failed": 0,
        "packets_skipped": 0,
        "accept_count": 0,
        "adjust_count": 0,
        "reject_count": 0,
        "unrefined_count": 0,
        "continuity_proposed": 0,
        "continuity_accepted": 0,
        "continuity_rejected": 0,
        "continuity_rejected_reasons": [],
        "duration_conflicts": 0,
        "distinct_moments": 0,
        "selected_variants": 0,
        "rejections": [],
        "budget": {"attempts_used": 0, "attempts_max": 0,
                   "deadline_s": 0.0, "exhausted": False},
        "counts_reconcile": True,
    }


def _default_editorial_prompt(packet: Mapping[str, Any],
                              instructions: Optional[str],
                              **prefs: Any) -> str:
    from clippyme.pipeline import gemini_request as gr
    return gr.build_editorial_prompt(
        packet, instructions,
        clip_type=prefs.get("clip_type"),
        duration_mode=prefs.get("duration_mode"),
        audience_intel=prefs.get("audience_intel"))


def _default_editorial_parse(text: str) -> Dict[str, Any]:
    from clippyme.pipeline import gemini_parser as gp
    return gp.parse_editorial_payload(text)


def _default_editorial_generate(client: Any, prompt: str,
                                cfg: EditorialConfig,
                                budget: EditorialBudget) -> str:
    """Budget-aware generation mirroring discovery's fallback loop.

    Model fallback loop with the shared budget checked immediately
    before each generate_content call; per-attempt finite timeout;
    SDK retries disabled ({"attempts": 1}); only retryable/unavailable
    model errors move to the next model.
    """
    from clippyme.pipeline import gemini_request as gr
    models = [cfg.model_name] + list(cfg.fallback_models)
    models = [m for m in models if m] or ["gemini-3.5-flash"]
    last_error = None
    first = True
    for model_name in models:
        # Primary already consumed by the candidate loop; fallbacks
        # consume one attempt each.
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
                config={"http_options": {
                    "timeout": int(timeout_ms),
                    "retry_options": {"attempts": 1},
                }},
            )
            return response.text or ""
        except Exception as exc:
            last_error = exc
            if not (gr._is_retryable_model_error(exc)
                    or gr._is_unavailable_model_error(exc)):
                raise
            continue
    if last_error:
        raise last_error
    raise EditorialError("budget exhausted before generation")


def _close_quietly(client: Any) -> None:
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def run_editorial_refinement(
    discovery_result: Optional[Mapping[str, Any]],
    words: Sequence[Dict[str, Any]],
    video_duration: float,
    *,
    config: Optional[EditorialConfig] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    build_prompt_fn: Optional[Callable[..., str]] = None,
    parse_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    generate_fn: Optional[Callable[..., str]] = None,
    instructions: Optional[str] = None,
    clip_type: Optional[str] = None,
    duration_mode: Optional[str] = None,
    audience_intel: Optional[Any] = None,
    policy_min: Optional[float] = None,
    policy_max: Optional[float] = None,
    time_fn: Callable[[], float] = time.monotonic,
    prior_plans: Optional[Mapping[str, Mapping[str, Any]]] = None,
    _known_client: Any = None,
) -> Dict[str, Any]:
    """Run the editorial pass over post-dedupe discovery candidates.

    Returns {"editorial_plans": [...], "editorial_refinement": {...}}.
    Plans are JSON-serializable dicts. One malformed decision never
    erases other candidates; failures keep the discovery interval as
    explicitly ``unrefined`` (never editor-approved).

    ``_known_client`` is private wiring: a client instance the caller
    already owns (and closes). A factory-created instance identical to
    it is borrowed, never double-closed.
    """
    from clippyme.pipeline import gemini_request as gr  # noqa: F401

    cfg = config or EditorialConfig.from_env(
        model_name=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))
    if policy_min is None or policy_max is None:
        from clippyme.pipeline.duration_policy import (
            resolve_duration_policy)
        _pol = resolve_duration_policy(video_duration=video_duration)
        if policy_min is None:
            policy_min = _pol.min_duration
        if policy_max is None:
            policy_max = _pol.max_duration

    store = (discovery_result or {}).get("candidate_store") or {}
    dfunnel = (discovery_result or {}).get("candidate_discovery") or {}
    removed: set = set()
    for g in dfunnel.get("duplicate_details") or []:
        for rid in g.get("removed_ids") or []:
            removed.add(rid)
    kept_ids = [cid for cid in store if cid not in removed]
    candidates_by_id = {cid: store[cid] for cid in kept_ids}
    # Deterministic processing order.
    kept_ids.sort(
        key=lambda cid: (float(store[cid].get("start_s", 0.0)), cid))

    funnel: Dict[str, Any] = {
        "version": EDITORIAL_VERSION,
        "enabled": True,
        "config": {
            "model_name": cfg.model_name,
            "fallback_models": list(cfg.fallback_models),
            "context_words": cfg.context_words,
            "context_seconds": cfg.context_seconds,
            "max_provider_attempts": cfg.max_provider_attempts,
            "deadline_s": cfg.deadline_s,
            "sdk_timeout_ms": cfg.sdk_timeout_ms,
            "duration_min_s": policy_min,
            "duration_max_s": policy_max,
            "duration_mode": duration_mode,
            "clip_type": clip_type,
            "audience_intel_present": audience_intel is not None,
        },
        "packets_planned": len(kept_ids),
        "packets_attempted": 0,
        "packets_succeeded": 0,
        "packets_failed": 0,
        "packets_skipped": 0,
        "accept_count": 0,
        "adjust_count": 0,
        "reject_count": 0,
        "unrefined_count": 0,
        "continuity_proposed": 0,
        "continuity_accepted": 0,
        "continuity_rejected": 0,
        "continuity_rejected_reasons": [],
        "duration_conflicts": 0,
        "distinct_moments": 0,
        "selected_variants": 0,
        "rejections": [],
        "budget": {"attempts_used": 0,
                   "attempts_max": cfg.max_provider_attempts,
                   "deadline_s": cfg.deadline_s, "exhausted": False},
    }

    def _count_verdict(plan: EditorialPlan) -> None:
        if plan.verdict == "accept":
            funnel["accept_count"] += 1
        elif plan.verdict == "adjust":
            funnel["adjust_count"] += 1
        elif plan.verdict == "reject":
            funnel["reject_count"] += 1
        else:
            funnel["unrefined_count"] += 1

    def _unrefined(candidate: Mapping[str, Any], reason: str,
                   conflict: Optional[str] = None) -> EditorialPlan:
        return _unrefined_plan(candidate, _blank_quality(), None,
                               reason, conflict)

    def _finish(plans: List[EditorialPlan]) -> Dict[str, Any]:
        funnel["budget"]["attempts_used"] = budget.attempts_used
        funnel["budget"]["exhausted"] = budget.exhausted()
        c1 = (funnel["packets_planned"]
              == funnel["packets_succeeded"] + funnel["packets_failed"]
              + funnel["packets_skipped"])
        c2 = (funnel["accept_count"] + funnel["adjust_count"]
              + funnel["reject_count"] + funnel["unrefined_count"]
              == funnel["packets_planned"])
        c3 = (funnel["continuity_proposed"]
              == funnel["continuity_accepted"]
              + funnel["continuity_rejected"])
        funnel["counts_reconcile"] = bool(c1 and c2 and c3)
        return {"editorial_plans": [p.to_dict() for p in plans],
                "editorial_refinement": funnel}

    budget = EditorialBudget(cfg, time_fn=time_fn)

    if not editorial_refinement_enabled():
        plans = [_unrefined(c, "refinement_disabled") for c in
                 (candidates_by_id[cid] for cid in kept_ids)]
        funnel["enabled"] = False
        funnel["packets_skipped"] = len(kept_ids)
        for p in plans:
            _count_verdict(p)
        return _finish(plans)

    # Client lifecycle: create exactly one owned client via the factory,
    # close it exactly once on all exits. A factory result identical to
    # the caller's already-owned instance is borrowed, never closed.
    client = None
    owned = False
    if client_factory is not None:
        try:
            client = client_factory()
            owned = True
        except Exception as e:
            plans = [_unrefined(c, "client_setup_failed")
                     for c in (candidates_by_id[cid]
                               for cid in kept_ids)]
            funnel["packets_skipped"] = len(kept_ids)
            funnel["rejections"].append(
                {"reason": "client_setup_failed",
                 "detail": str(e)[:200]})
            for p in plans:
                _count_verdict(p)
            return _finish(plans)
    if client is None:
        # Enabled but no client available: zero provider calls.
        plans = [_unrefined(c, "no_client_factory")
                 for c in (candidates_by_id[cid] for cid in kept_ids)]
        funnel["packets_skipped"] = len(kept_ids)
        for p in plans:
            _count_verdict(p)
        return _finish(plans)

    _real_build = gr.build_editorial_prompt
    if build_prompt_fn is not None:
        def build_prompt(packet, instructions, **prefs):
            try:
                return build_prompt_fn(packet, instructions, **prefs)
            except TypeError:
                return build_prompt_fn(packet, instructions)
    else:
        def build_prompt(packet, instructions, **prefs):
            return _real_build(
                packet, instructions,
                clip_type=prefs.get("clip_type"),
                duration_mode=prefs.get("duration_mode"),
                audience_intel=prefs.get("audience_intel"))
    parse = parse_fn or _default_editorial_parse
    gen = generate_fn or _default_editorial_generate

    plans: List[EditorialPlan] = []
    group_plans: List[EditorialPlan] = []
    try:
        all_cands = [candidates_by_id[cid] for cid in kept_ids]
        for cid in kept_ids:
            cand = candidates_by_id[cid]
            # Resume: never repeat a completed plan — but never
            # trust stale identity either. An invalid prior plan is
            # reprocessed from scratch (recorded honestly) instead of
            # being preserved as completed work.
            if prior_plans and cid in prior_plans:
                raw = prior_plans[cid]
                bad = _validate_prior_plan(raw, cand, video_duration,
                                           policy_min, policy_max)
                if bad is None:
                    plan = EditorialPlan.from_dict(raw)
                    plans.append(plan)
                    funnel["packets_skipped"] += 1
                    _count_verdict(plan)
                    continue
                funnel["rejections"].append(
                    {"candidate_id": cid,
                     "reason": f"invalid_prior_plan:{bad}"})
            try:
                packet = build_context_packet(
                    cand, all_cands, words, video_duration,
                    policy_min, policy_max, cfg)
            except Exception as e:
                plans.append(_unrefined(
                    cand, f"error:packet:{type(e).__name__}:{str(e)[:80]}"))
                funnel["packets_failed"] += 1
                _count_verdict(plans[-1])
                continue
            if not budget.consume():
                plans.append(_unrefined(cand, "budget_skipped"))
                funnel["packets_skipped"] += 1
                _count_verdict(plans[-1])
                continue
            funnel["packets_attempted"] += 1
            try:
                prompt = build_prompt(
                    packet, instructions,
                    clip_type=clip_type, duration_mode=duration_mode,
                    audience_intel=audience_intel)
            except Exception as e:
                plans.append(_unrefined(
                    cand, f"prompt_failed:{type(e).__name__}:"
                          f"{str(e)[:100]}"))
                funnel["packets_failed"] += 1
                _count_verdict(plans[-1])
                continue
            try:
                raw_text = gen(client, prompt, cfg, budget)
            except Exception as e:
                plans.append(_unrefined(
                    cand, f"generate_failed:{type(e).__name__}:"
                          f"{str(e)[:100]}"))
                funnel["packets_failed"] += 1
                _count_verdict(plans[-1])
                continue
            try:
                decision = parse(raw_text)
            except Exception:
                plans.append(_unrefined(cand, "parse_failed"))
                funnel["packets_failed"] += 1
                _count_verdict(plans[-1])
                continue
            neighbor_guards = [
                (float(candidates_by_id[o]["start_s"]),
                 float(candidates_by_id[o]["end_s"]))
                for o in kept_ids if o != cid
            ]
            try:
                plan = validate_editorial_decision(
                    decision, packet, candidates_by_id, words,
                    video_duration, policy_min, policy_max,
                    neighbor_guards)
            except EditorialError as e:
                plans.append(_unrefined(
                    cand, f"invalid_decision:{e}"))
                funnel["packets_failed"] += 1
                _count_verdict(plans[-1])
                continue
            plans.append(plan)
            _count_verdict(plan)
            if plan.status == "refined":
                funnel["packets_succeeded"] += 1
            else:
                funnel["packets_failed"] += 1
            if plan.conflict == "duration_conflict":
                funnel["duration_conflicts"] += 1

        # Continuity: explicit links between refined accept/adjust plans.
        refined = [p for p in plans
                   if p.status == "refined"
                   and p.verdict in ("accept", "adjust")]
        by_id = {p.candidate_id: p for p in refined}
        parent: Dict[str, str] = {}

        def _find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for p in refined:
            parent.setdefault(p.candidate_id, p.candidate_id)
            for link in p.continuity_links:
                if link == p.candidate_id:
                    continue
                if link in by_id:
                    parent.setdefault(link, link)
                    ra, rb = _find(p.candidate_id), _find(link)
                    if ra != rb:
                        parent[ra] = rb
        comps: Dict[str, List[str]] = {}
        for p in refined:
            comps.setdefault(_find(p.candidate_id), []).append(
                p.candidate_id)
        # Deterministic proposal order: source order of first member.
        ordered = sorted(
            (v for v in comps.values() if len(v) >= 2),
            key=lambda ids: (min(float(candidates_by_id[c]["start_s"])
                                 for c in ids), ids))

        proposals: List[tuple] = [("group", ids) for ids in ordered]
        seen_bad = set()
        for p in refined:
            for link in p.continuity_links:
                if link == p.candidate_id or link in by_id:
                    continue
                key = (p.candidate_id, link)
                if key in seen_bad:
                    continue
                seen_bad.add(key)
                proposals.append(("badlink", [p.candidate_id, link]))
        for kind, ids in proposals:
            funnel["continuity_proposed"] += 1
            if kind == "badlink":
                funnel["continuity_rejected"] += 1
                funnel["continuity_rejected_reasons"].append(
                    {"group": ids,
                     "reason": "continuity: unknown candidate id"})
                continue
            try:
                gplan = plan_continuity_group(
                    ids, by_id, candidates_by_id, words,
                    video_duration, policy_min, policy_max)
            except EditorialError as e:
                funnel["continuity_rejected"] += 1
                funnel["continuity_rejected_reasons"].append(
                    {"group": ids, "reason": str(e)})
                continue
            funnel["continuity_accepted"] += 1
            for cid in ids:
                by_id[cid].continuity_group = gplan.candidate_id
            group_plans.append(gplan)

        # Final selected editorial metrics describe selected
        # outcomes only: refined accept/adjust plans plus explicit
        # unrefined discovery candidates (failure semantics keep them
        # selected). Rejects are excluded. Accepted continuity groups
        # collapse their selected components exactly once.
        selected = [p for p in plans
                    if p.verdict in ("accept", "adjust", "unrefined")]
        grouped_cids = {c for g in group_plans for c in g.components}
        ungrouped = [p for p in selected
                     if p.candidate_id not in grouped_cids]
        gv = group_variants(ungrouped)
        funnel["distinct_moments"] = (
            gv["distinct_moments"] + len(group_plans))
        funnel["selected_variants"] = sum(
            len(v) - 1 for v in gv["variants"].values())
    finally:
        if owned and client is not None and client is not _known_client:
            _close_quietly(client)

    return _finish(plans + group_plans)
