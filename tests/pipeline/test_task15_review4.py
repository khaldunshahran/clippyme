"""Task 15 review-04: promoted probes + regressions + load-bearing mutations.

Covers the one remaining area — claim meaning must fail closed:

1. Sensory verb gate: uninflected base forms in verb position
   ("red ball can explode") are action claims, not descriptive
   content. Novel sensory content is grounded only through the
   explicit bounded manner-paraphrase authorization; unknown support
   is rejected, never grounded.
2. Negation scope: polarity is validated per claim/subject, not per
   quote, so "coffee does work but tea does not work" cannot borrow
   the single negation of "coffee does not work but tea does work".

Injected providers only; synthetic fixtures; no live providers, no
downloads, no new dependencies.
"""
import pytest

from clippyme.pipeline import opening_plan as op


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _packet(**kwargs):
    args = dict(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=[{"word": "coffee", "start": 11., "end": 11.4},
               {"word": "works", "start": 11.4, "end": 12.}],
        language="en")
    args.update(kwargs)
    return op.build_opening_packet(**args)


def _option(**kwargs):
    args = dict(
        id="x", text="coffee works", language="en",
        rationale="retained words", evidence_basis="speech",
        evidence=[{"basis": "speech", "quote": "coffee works"}],
        display={"start": 0., "end": 2.})
    args.update(kwargs)
    return args


def _visual_packet(label="red ball rolls", event_id="v1"):
    event = {"kind": "visual", "label": label,
             "start": 11., "end": 12., "event_id": event_id}
    pkt = op.build_opening_packet(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=[], av_evidence=[event], language="en")
    return pkt, event


def _visual_option(text, event):
    return _option(
        text=text, evidence_basis="visual",
        evidence=[{"basis": "visual", "label": event["label"],
                   "start": event["start"], "end": event["end"],
                   "event_id": event["event_id"]}])


def _mutate(name, value):
    real = getattr(op, name)
    setattr(op, name, value)
    return real


# ---------------------------------------------------------------------------
# Promoted probe 1: uninflected sensory action is not descriptive content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["red ball can explode", "red ball is poisonous"])
def test_unverifiable_sensory_content_is_not_grounded(text):
    pkt, event = _visual_packet()
    accepted, rejected = op.validate_opening_option(
        _visual_option(text, event), packet=pkt)
    assert accepted is None, \
        "novel unsupported action/property blessed grounded because the verb gate missed it"
    assert rejected["code"] == "text_unsupported"


def test_uninflected_action_after_modal_is_rejected():
    pkt, event = _visual_packet()
    accepted, _ = op.validate_opening_option(
        _visual_option("red ball will detonate", event), packet=pkt)
    assert accepted is None, \
        "uninflected action after modal treated as descriptive"


def test_supported_manner_paraphrase_still_passes():
    pkt, event = _visual_packet(label="crowd cheers")
    norm, rej = op.validate_opening_option(
        _visual_option("the crowd erupts", event), packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_supported_manner_base_form_still_passes():
    pkt, event = _visual_packet(label="crowd cheers")
    norm, rej = op.validate_opening_option(
        _visual_option("the crowd can erupt", event), packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_evidence_supported_sensory_text_still_passes():
    pkt, event = _visual_packet()
    norm, rej = op.validate_opening_option(
        _visual_option("red ball rolls", event), packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


# ---------------------------------------------------------------------------
# Promoted probe 2: negation scope is validated per claim/subject
# ---------------------------------------------------------------------------

def _negation_packet():
    quote = "coffee does not work but tea does work"
    words = [{"word": w, "start": 11 + i * 0.2, "end": 11 + (i + 1) * 0.2}
             for i, w in enumerate(quote.split())]
    return op.build_opening_packet(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=words, language="en"), quote


def test_equal_negation_counts_cannot_transfer_negation_to_other_claim():
    pkt, quote = _negation_packet()
    accepted, rejected = op.validate_opening_option(
        _option(text="coffee does work but tea does not work",
                evidence=[{"basis": "speech", "quote": quote}]),
        packet=pkt)
    assert accepted is None, \
        "one negation count preserved while both actual claims reversed"
    assert rejected["code"] == "text_unsupported"


def test_matching_multi_clause_polarity_still_passes():
    pkt, quote = _negation_packet()
    norm, rej = op.validate_opening_option(
        _option(text="coffee does not work and tea does work",
                evidence=[{"basis": "speech", "quote": quote}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_partial_clause_selection_still_passes():
    pkt, quote = _negation_packet()
    norm, rej = op.validate_opening_option(
        _option(text="tea does work",
                evidence=[{"basis": "speech", "quote": quote}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_negated_partial_clause_selection_still_passes():
    pkt, quote = _negation_packet()
    norm, rej = op.validate_opening_option(
        _option(text="coffee does not work",
                evidence=[{"basis": "speech", "quote": quote}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_contraction_polarity_still_passes():
    words = [{"word": w, "start": 11 + i * 0.2, "end": 11 + (i + 1) * 0.2}
             for i, w in enumerate("coffee does not work".split())]
    pkt = op.build_opening_packet(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=words, language="en")
    norm, rej = op.validate_opening_option(
        _option(text="coffee doesn't work",
                evidence=[{"basis": "speech", "quote": "coffee does not work"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_dropped_negation_still_rejected():
    words = [{"word": w, "start": 11 + i * 0.2, "end": 11 + (i + 1) * 0.2}
             for i, w in enumerate("coffee does not work".split())]
    pkt = op.build_opening_packet(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=words, language="en")
    accepted, rejected = op.validate_opening_option(
        _option(text="coffee does work",
                evidence=[{"basis": "speech", "quote": "coffee does not work"}]),
        packet=pkt)
    assert accepted is None
    assert rejected["code"] == "text_unsupported"


def test_added_negation_still_rejected():
    accepted, rejected = op.validate_opening_option(
        _option(text="coffee never works",
                evidence=[{"basis": "speech", "quote": "coffee works"}]),
        packet=_packet())
    assert accepted is None
    assert rejected["code"] == "text_unsupported"


# ---------------------------------------------------------------------------
# Load-bearing mutations
# ---------------------------------------------------------------------------

def test_mutation_verb_position_gate_is_load_bearing():
    pkt, event = _visual_packet()
    bad = _visual_option("red ball can explode", event)
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    # MUTATION: drop the verb-position markers -> the uninflected
    # action is no longer detected and the option passes.
    real = _mutate("_VERB_POSITION_MARKERS", frozenset())
    try:
        accepted, _ = op.validate_opening_option(bad, packet=pkt)
        assert accepted is not None, "mutation did not break the gate"
    finally:
        _mutate("_VERB_POSITION_MARKERS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None


def test_mutation_clause_scope_gate_is_load_bearing():
    pkt, quote = _negation_packet()
    bad = _option(text="coffee does work but tea does not work",
                  evidence=[{"basis": "speech", "quote": quote}])
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    # MUTATION: drop the clause boundaries -> whole-quote negation
    # counts match (1 == 1) and the swapped claims pass.
    real = _mutate("_CLAUSE_BOUNDARY_TOKENS", frozenset())
    try:
        accepted, _ = op.validate_opening_option(bad, packet=pkt)
        assert accepted is not None, "mutation did not break the gate"
    finally:
        _mutate("_CLAUSE_BOUNDARY_TOKENS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None


def test_mutation_manner_paraphrase_gate_still_load_bearing():
    pkt, event = _visual_packet(label="crowd cheers")
    good = _visual_option("the crowd erupts", event)
    accepted, _ = op.validate_opening_option(good, packet=pkt)
    assert accepted is not None
    # MUTATION: drop the manner-paraphrase mapping -> "erupts" becomes
    # an unverifiable novel action.
    real = _mutate("_SENSORY_MANNER_VERBS", {})
    try:
        accepted, _ = op.validate_opening_option(good, packet=pkt)
        assert accepted is None, "mutation did not break the gate"
    finally:
        _mutate("_SENSORY_MANNER_VERBS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(good, packet=pkt)
    assert accepted is not None
