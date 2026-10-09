"""Review-05: unknown sensory claim support must fail closed.

The sensory material-content gate is default-deny: every novel
non-glue claim token needs explicit bounded support (evidence
membership or an authorized transformation). Unknown descriptive
properties are claims too; being grammatically an adjective does
not establish evidence support. The catch-all permission for
unrecognized modifiers/actions is removed.

Known paraphrases pass through their explicit authorizations
(the curated manner-verb allowlist); supported multilingual text
passes via evidence membership.

Injected providers only; synthetic fixtures; no live providers,
no downloads, no new dependencies.
"""
import pytest

from clippyme.pipeline import opening_plan as op


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _packet(**kwargs):
    args = dict(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=[], language="en")
    args.update(kwargs)
    return op.build_opening_packet(**args)


def _option(**kwargs):
    args = dict(
        id="x", text="red ball rolls", language="en",
        rationale="retained evidence", evidence_basis="visual",
        evidence=[], display={"start": 0., "end": 2.})
    args.update(kwargs)
    return args


def _visual_packet(label="red ball rolls", event_id="v1", language="en"):
    event = {"kind": "visual", "label": label,
             "start": 11., "end": 12., "event_id": event_id}
    pkt = op.build_opening_packet(
        source_fingerprint="src", final_start=10., final_end=20.,
        words=[], av_evidence=[event], language=language)
    return pkt, event


def _visual_option(text, event, language="en"):
    return _option(
        text=text, language=language, evidence_basis="visual",
        evidence=[{"basis": "visual", "label": event["label"],
                   "start": event["start"], "end": event["end"],
                   "event_id": event["event_id"]}])


def _mutate(name, value):
    real = getattr(op, name)
    setattr(op, name, value)
    return real


# ---------------------------------------------------------------------------
# Promoted review-05 probes: unknown sensory meaning is not grounded
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "red ball will suddenly explode",  # modal/adverb action (was already rejected)
    "red ball is explosive",           # invented property (the failing case)
])
def test_unknown_sensory_meaning_is_not_grounded(text):
    pkt, event = _visual_packet()
    accepted, rejected = op.validate_opening_option(
        _visual_option(text, event), packet=pkt)
    assert accepted is None, \
        "unsupported sensory action/property bypassed default-deny and was labelled grounded"
    assert rejected["code"] == "text_unsupported"


# ---------------------------------------------------------------------------
# Negative controls: other invented properties/actions fail closed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "red ball is shiny",
    "red ball is enormous",
    "red ball rolls quickly",  # invented manner adverb, not in the curated allowlist
    "red ball vanishes",
    "red ball is dangerous",
])
def test_other_invented_sensory_properties_are_rejected(text):
    pkt, event = _visual_packet()
    accepted, _ = op.validate_opening_option(
        _visual_option(text, event), packet=pkt)
    assert accepted is None, \
        f"invented sensory content {text!r} was marked grounded"


# ---------------------------------------------------------------------------
# Positive controls: evidence, authorized paraphrase, multilingual
# ---------------------------------------------------------------------------

def test_evidence_supported_sensory_text_still_passes():
    pkt, event = _visual_packet()
    norm, rej = op.validate_opening_option(
        _visual_option("red ball rolls", event), packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


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


def test_supported_multilingual_sensory_text_still_passes():
    pkt, event = _visual_packet(label="pelota roja rueda", language="es")
    norm, rej = op.validate_opening_option(
        _visual_option("pelota roja rueda", event, language="es"), packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True
    assert norm["language"] == "es"


# ---------------------------------------------------------------------------
# Load-bearing mutation: restoring the catch-all must fail the gate
# ---------------------------------------------------------------------------

def test_mutation_default_deny_is_load_bearing():
    pkt, event = _visual_packet()
    bad = _visual_option("red ball is explosive", event)
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    # MUTATION: restore the legacy catch-all permission for
    # unrecognized modifiers -> the invented property passes.
    real = _mutate("_SENSORY_DEFAULT_DENY_UNKNOWN", False)
    try:
        accepted, _ = op.validate_opening_option(bad, packet=pkt)
        assert accepted is not None, "mutation did not break the gate"
        assert accepted["grounded"] is True
    finally:
        _mutate("_SENSORY_DEFAULT_DENY_UNKNOWN", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None


def test_mutation_default_deny_backstops_action_gate():
    # Even when the structural action-verb gate is fully disabled,
    # default-deny still rejects the unlisted novel action.
    pkt, event = _visual_packet()
    bad = _visual_option("red ball explodes", event)
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    real_markers = _mutate("_VERB_POSITION_MARKERS", frozenset())
    try:
        accepted, _ = op.validate_opening_option(bad, packet=pkt)
        assert accepted is None, "default-deny backstop failed"
    finally:
        _mutate("_VERB_POSITION_MARKERS", real_markers)
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
