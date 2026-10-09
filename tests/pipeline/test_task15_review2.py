"""Task 15 review-02: durable promotion of the 11 coordinator probes.

Every probe from work/muse-handoff/task-15/coordinator-review2-remaining.py
is promoted here, PLUS real-route tests and load-bearing fail-then-restore
mutations for the four remaining areas:

1. Claim grounding: short claims, polarity, and meaning are validated
   consistently across languages/modalities; invented people/events are
   rejected while genuinely supported paraphrases pass.
2. Retained sensory evidence: frame/event references actually inside the
   retained interval, with provenance/source binding; event_id + label +
   interval bound together.
3. Fail-closed emission: current packet/schema/identity required, complete
   normalized membership, source/config rechecked; user overrides on a
   separate authorship path.
4. Production adapter: configured fallback chain executed through the
   shared budget, timeout never rounded above the remainder.

Injected providers only; synthetic fixtures; no live providers, no
downloads, no new dependencies.
"""
import os
import subprocess

import pytest

from clippyme.pipeline import opening_plan as op
from clippyme.pipeline.project_emit import (
    _grounded_hook_overlay,
    _provider_config_is_current,
)


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


def _budget(**kw):
    args = dict(deadline_s=60., max_requests=8, request_timeout_s=10.)
    args.update(kw)
    return op.OpeningBudget(**args)


class _FakeProvider:
    def __init__(self, options=None, error=None):
        self._options = options if options is not None else []
        self._error = error
        self.calls = 0
        self.closed = 0

    def request_options(self, request):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._options

    def close(self):
        self.closed += 1


@pytest.fixture
def av_job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    js = str(job)
    src = os.path.join(js, "source_clip0.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "color=blue:s=320x180:r=30:d=6",
         "-f", "lavfi", "-i",
         "sine=frequency=880:sample_rate=48000:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", src],
        capture_output=True)
    assert r.returncode == 0, r.stderr[-500:]
    final = os.path.join(js, "clip0.mp4")
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", final],
        capture_output=True)
    assert r.returncode == 0, r.stderr[-500:]
    return js, src, final


# ---------------------------------------------------------------------------
# Promoted probes 1-2: short claims and negation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["Bob won", "coffee never works"])
def test_short_claims_and_negation_cannot_bypass_grounding(text):
    accepted, _ = op.validate_opening_option(
        _option(text=text), packet=_packet())
    assert accepted is None, f"unsupported or opposite claim accepted: {text}"


def test_negation_in_evidence_stays_supported():
    pkt = _packet(words=[
        {"word": "coffee", "start": 11., "end": 11.4},
        {"word": "never", "start": 11.4, "end": 11.7},
        {"word": "works", "start": 11.7, "end": 12.}])
    norm, rej = op.validate_opening_option(
        _option(text="coffee never works",
                evidence=[{"basis": "speech", "quote": "coffee never works"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


# ---------------------------------------------------------------------------
# Promoted probes 3-4: non-English and visual claim support
# ---------------------------------------------------------------------------

def test_nonenglish_claims_not_blessed_by_half_token_overlap():
    pkt = _packet(language="es", words=[
        {"word": "café", "start": 11., "end": 11.4},
        {"word": "funciona", "start": 11.4, "end": 12.}])
    accepted, _ = op.validate_opening_option(
        _option(text="café funciona Alicia campeonato", language="es",
                rationale="r",
                evidence=[{"basis": "speech", "quote": "café funciona"}]),
        packet=pkt)
    assert accepted is None, \
        "invented non-English person/event still marked grounded"


def test_visual_claims_not_blessed_by_half_token_overlap():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls across field",
         "start": 11., "end": 12., "event_id": "a"}])
    accepted, _ = op.validate_opening_option(
        _option(text="red ball rolls across field Alice championship",
                evidence_basis="visual",
                evidence=[{"basis": "visual",
                           "label": "red ball rolls across field",
                           "start": 11., "end": 12., "event_id": "a"}]),
        packet=pkt)
    assert accepted is None, \
        "invented visual person/event still marked grounded"


# ---------------------------------------------------------------------------
# Claim grounding: supported controls that must keep passing
# ---------------------------------------------------------------------------

def test_supported_paraphrase_controls_still_pass():
    pkt = _packet(words=[
        {"word": "hello", "start": 10.5, "end": 11.0},
        {"word": "world", "start": 11.0, "end": 11.5},
        {"word": "this", "start": 12.0, "end": 12.3},
        {"word": "is", "start": 12.3, "end": 12.4},
        {"word": "amazing", "start": 12.4, "end": 13.0}])
    norm, rej = op.validate_opening_option(
        _option(text="hello, this is amazing",
                evidence=[{"basis": "speech",
                           "quote": "hello world this is amazing"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_supported_visual_paraphrase_control_still_passes():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "crowd cheers",
         "start": 11., "end": 13., "event_id": "v1"}])
    norm, rej = op.validate_opening_option(
        _option(text="the crowd erupts", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "crowd cheers",
                           "start": 11., "end": 13., "event_id": "v1"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True
    assert norm["evidence"][0]["event_id"] == "v1"


def test_supported_non_english_control_still_passes():
    pkt = _packet(language="fi", words=[
        {"word": "kahvi", "start": 10.5, "end": 11.0},
        {"word": "toimii", "start": 11.0, "end": 11.5}])
    norm, rej = op.validate_opening_option(
        _option(text="kahvi toimii", language="fi", rationale="suomeksi",
                evidence=[{"basis": "speech", "quote": "kahvi toimii"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["language"] == "fi"


# ---------------------------------------------------------------------------
# Claim grounding: fail-then-restore mutations (load-bearing gates)
# ---------------------------------------------------------------------------

def _mutate(module_attr, value):
    real = getattr(op, module_attr)
    setattr(op, module_attr, value)
    return real


def test_mutation_negation_gate_is_load_bearing():
    bad = _option(text="coffee never works")
    accepted, _ = op.validate_opening_option(bad, packet=_packet())
    assert accepted is None
    # MUTATION: restore "never" as glue -> polarity flip passes.
    real = _mutate("_GLUE_TOKENS", op._GLUE_TOKENS | {"never"})
    try:
        accepted, _ = op.validate_opening_option(bad, packet=_packet())
        assert accepted is not None, "mutation did not break the gate"
    finally:
        _mutate("_GLUE_TOKENS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(bad, packet=_packet())
    assert accepted is None


def test_mutation_event_word_gate_is_load_bearing():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "a"}])
    bad = _option(text="red ball rolls championship", evidence_basis="visual",
                  evidence=[{"basis": "visual", "label": "red ball rolls",
                             "start": 11., "end": 12., "event_id": "a"}])
    accepted, rejected = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    assert "event/outcome" in rejected["reason"]
    # MUTATION: drop "championship" from the event list. Under the
    # removed catch-all permission it would pass as a descriptive
    # modifier; under default-deny it is still rejected (now as
    # unverifiable), proving the default-deny branch is the backstop.
    real = _mutate("_EVENT_OUTCOME_WORDS",
                   op._EVENT_OUTCOME_WORDS - {"championship"})
    try:
        accepted, rejected = op.validate_opening_option(bad, packet=pkt)
        assert accepted is None, "default-deny backstop failed"
        assert "unverifiable" in rejected["reason"]
    finally:
        _mutate("_EVENT_OUTCOME_WORDS", real)
    # RESTORE.
    accepted, rejected = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    assert "event/outcome" in rejected["reason"]


def test_mutation_proper_noun_gate_is_load_bearing():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "a"}])
    bad = _option(text="red ball rolls with Alice", evidence_basis="visual",
                  evidence=[{"basis": "visual", "label": "red ball rolls",
                             "start": 11., "end": 12., "event_id": "a"}])
    accepted, rejected = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    assert "person/entity" in rejected["reason"]
    # MUTATION: disable proper-noun detection. Under the removed
    # catch-all permission the invented person would pass as a
    # descriptive modifier; under default-deny it is still rejected
    # (now as unverifiable), proving the default-deny branch is the
    # backstop.
    real = _mutate("_looks_like_proper_noun", lambda raw, first: False)
    try:
        accepted, rejected = op.validate_opening_option(bad, packet=pkt)
        assert accepted is None, "default-deny backstop failed"
        assert "unverifiable" in rejected["reason"]
    finally:
        _mutate("_looks_like_proper_noun", real)
    # RESTORE.
    accepted, rejected = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    assert "person/entity" in rejected["reason"]


# ---------------------------------------------------------------------------
# Promoted probes 5-7: emission fail-closed contracts
# ---------------------------------------------------------------------------

def test_missing_current_packet_cannot_authorize_automatic_overlay():
    opt = _option()
    clip = {"opening_plan": {
        "schema_version": op.OPENING_PLAN_SCHEMA_VERSION,
        "packet_id": "stale", "validated_options": [opt], "selected": opt}}
    overlays, _ = _grounded_hook_overlay(clip, 10., current_packet=None)
    assert not overlays, \
        "current packet failure bypasses evidence/identity revalidation"


def test_selected_payload_must_match_validated_member_not_only_id():
    pkt = _packet()
    valid, _ = op.validate_opening_option(_option(), packet=pkt)
    selected = dict(valid, text="works coffee")
    clip = {"opening_plan": {
        "schema_version": op.OPENING_PLAN_SCHEMA_VERSION,
        "packet_id": pkt["packet_id"], "validated_options": [valid],
        "selected": selected}}
    overlays, _ = _grounded_hook_overlay(clip, 10., current_packet=pkt)
    assert not overlays, \
        "ID-only membership promotes a different unselected payload"


def test_stale_schema_cannot_emit_automatic_overlay():
    pkt = _packet()
    valid, _ = op.validate_opening_option(_option(), packet=pkt)
    clip = {"opening_plan": {
        "schema_version": -1, "packet_id": pkt["packet_id"],
        "validated_options": [valid], "selected": valid}}
    overlays, _ = _grounded_hook_overlay(clip, 10., current_packet=pkt)
    assert not overlays, "emission does not enforce current schema"


def test_config_change_blocks_automatic_overlay():
    pkt = _packet()
    valid, _ = op.validate_opening_option(_option(), packet=pkt)
    saved_cfg = {"kind": "gemini-opening", "model": "old-model",
                 "fallback_models": [], "prompt_version": "x"}
    clip = {"opening_plan": {
        "schema_version": op.OPENING_PLAN_SCHEMA_VERSION,
        "packet_id": pkt["packet_id"], "validated_options": [valid],
        "selected": valid, "provider_config": saved_cfg}}

    class _ChangedFactory:
        def describe(self):
            return {"kind": "gemini-opening", "model": "new-model",
                    "fallback_models": [], "prompt_version": "x"}

    op.set_opening_provider_factory(_ChangedFactory())
    try:
        overlays, opening = _grounded_hook_overlay(
            clip, 10., current_packet=pkt)
        assert not overlays,             "stale provider config still authorizes an overlay"
        assert opening["no_selection_reason"] == "provider_config_changed"
    finally:
        op.set_opening_provider_factory(None)
    # Control: a matching current config emits.
    class _SameFactory:
        def describe(self):
            return dict(saved_cfg)

    op.set_opening_provider_factory(_SameFactory())
    try:
        overlays, _ = _grounded_hook_overlay(
            clip, 10., current_packet=pkt)
        assert overlays, "current config should authorize the overlay"
    finally:
        op.set_opening_provider_factory(None)


def test_user_override_survives_without_current_packet():
    clip = {"opening_plan": {
        "schema_version": op.OPENING_PLAN_SCHEMA_VERSION,
        "user_authored": True, "text": "my hook"}}
    overlays, opening = _grounded_hook_overlay(clip, 10., current_packet=None)
    assert overlays and overlays[0]["text"] == "my hook"
    assert opening["status"] == "user_authored"


def test_provider_config_currency_unit():
    saved = {"kind": "gemini-opening", "model": "m1",
             "fallback_models": ["b"], "prompt_version": "p"}
    assert _provider_config_is_current(saved, dict(saved))
    assert _provider_config_is_current(
        saved, {"kind": "gemini-opening", "model": "m2",
                "fallback_models": ["b"], "prompt_version": "p"}) is False
    assert _provider_config_is_current(
        saved, {"kind": "other", "model": "m1",
                "fallback_models": ["b"], "prompt_version": "p"}) is False
    # Lenient when nothing saved or nothing authoritative to compare.
    assert _provider_config_is_current(None, saved)
    assert _provider_config_is_current(saved, None)
    assert _provider_config_is_current({}, {})


# ---------------------------------------------------------------------------
# Promoted probes 8-9: sensory evidence identity and scope
# ---------------------------------------------------------------------------

def test_partial_candidate_description_cannot_be_relabelled_retained():
    data = {"candidate_store": {
        "a": {"candidate_id": "a", "start_s": 10., "end_s": 20.,
              "evidence_basis": "visual",
              "visual_description": "red ball rolls then scores at the end",
              "visual_frame_ids": ["frame-at-19"],
              "source_fingerprint": "src"}}}
    events = op._opening_av_evidence_for_clip(
        {"candidate_id": "a"}, data, 10., 12.)
    assert not events, \
        "whole candidate summary retimed to a cut that excludes its payoff"


def test_retained_frame_reference_reaches_packet():
    data = {"candidate_store": {
        "a": {"candidate_id": "a", "start_s": 10., "end_s": 20.,
              "evidence_basis": "visual",
              "visual_description": "red ball rolls",
              "visual_frame_ids": ["frame-at-11"],
              "source_fingerprint": "src"}}}
    events = op._opening_av_evidence_for_clip(
        {"candidate_id": "a"}, data, 10., 12.)
    assert events, "retained frame reference declined"
    assert events[0]["kind"] == "visual"
    assert events[0]["source_fingerprint"] == "src"
    assert events[0]["start"] == 10. and events[0]["end"] == 12.


def test_specific_sensory_event_identity_is_required():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "a"}])
    accepted, _ = op.validate_opening_option(
        _option(text="red ball rolls", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 11., "end": 12., "event_id": "forged"}]),
        packet=pkt)
    assert accepted is None, "sensory citation event_id ignored"


def test_repeated_label_events_stay_distinct():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "first"},
        {"kind": "visual", "label": "red ball rolls",
         "start": 15., "end": 16., "event_id": "second"}])
    # A citation for the later event binds the later event, not the
    # first label match.
    norm, rej = op.validate_opening_option(
        _option(text="red ball rolls", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 15., "end": 16., "event_id": "second"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["evidence"][0]["event_id"] == "second"
    # The first event's id with the second event's interval is forged.
    accepted, _ = op.validate_opening_option(
        _option(text="red ball rolls", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 15., "end": 16., "event_id": "first"}]),
        packet=pkt)
    assert accepted is None, "event_id/interval mismatch accepted"


def test_foreign_source_evidence_rejected():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "a",
         "source_fingerprint": "other-source"}])
    accepted, rej = op.validate_opening_option(
        _option(text="red ball rolls", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 11., "end": 12., "event_id": "a"}]),
        packet=pkt)
    assert accepted is None, "foreign-source sensory evidence accepted"
    assert rej["code"] == "evidence_not_accepted"


def test_mutation_event_id_binding_is_load_bearing():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "a"}])
    forged = _option(text="red ball rolls", evidence_basis="visual",
                     evidence=[{"basis": "visual", "label": "red ball rolls",
                                "start": 11., "end": 12.,
                                "event_id": "forged"}])
    accepted, _ = op.validate_opening_option(forged, packet=pkt)
    assert accepted is None
    # MUTATION: drop the event_id requirement -> forged id passes.
    real_fn = op._validate_sensory_evidence

    def _no_id_binding(items, basis, av_evidence, source_fingerprint=None):
        stripped = []
        for it in items:
            it = dict(it)
            it.pop("event_id", None)
            stripped.append(it)
        return real_fn(stripped, basis, av_evidence, source_fingerprint)

    op._validate_sensory_evidence = _no_id_binding
    try:
        accepted, _ = op.validate_opening_option(forged, packet=pkt)
        assert accepted is not None, "mutation did not break the gate"
    finally:
        op._validate_sensory_evidence = real_fn
    # RESTORE.
    accepted, _ = op.validate_opening_option(forged, packet=pkt)
    assert accepted is None


def test_prompt_carries_event_ids():
    pkt = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12., "event_id": "evt-7"}])
    prompt = op.build_opening_prompt(pkt)
    assert "evt-7" in prompt


# ---------------------------------------------------------------------------
# Promoted probes 10-11: production adapter timeout and fallback
# ---------------------------------------------------------------------------

def test_production_adapter_timeout_does_not_exceed_budget():
    # Review-03: a sub-millisecond remainder cannot represent a strictly
    # positive SDK timeout, so the adapter must stop BEFORE the SDK call
    # with explicit budget exhaustion — never send timeout=0.
    captured = []
    closes = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            captured.append(kw)
            return Response()

    class Client:
        models = Models()

        def close(self):
            closes.append(1)

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="injected")
    with pytest.raises(op.OpeningBudgetExhausted):
        provider.request_options({"timeout_s": .0004, "prompt": "test",
                                  "retries_disabled": True})
    assert captured == [], "SDK must not be called with timeout=0"
    assert closes == [1], "client still closes exactly once"


def test_adapter_never_rounds_fractional_timeout_up():
    captured = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            captured.append(kw)
            return Response()

    class Client:
        models = Models()

        def close(self):
            pass

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="injected")
    provider.request_options({"timeout_s": 1.9999, "prompt": "test",
                              "retries_disabled": True})
    assert captured[0]["config"]["http_options"]["timeout"] <= 1999


def test_adapter_rechecks_remaining_after_setup():
    calls = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            return Response()

    class Client:
        models = Models()

        def close(self):
            pass

    def slow_factory():
        return Client()

    provider = op.GeminiOpeningProvider(client_factory=slow_factory,
                                       model="injected")
    # Setup consumed the budget: no SDK call, explicit budget reason.
    with pytest.raises(op.OpeningBudgetExhausted):
        provider.request_options({"timeout_s": 5.0, "prompt": "test",
                                  "retries_disabled": True,
                                  "remaining_s": lambda: 0.0})
    assert calls == []


def test_configured_production_fallback_really_runs():
    calls = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            calls.append(kw["model"])
            if kw["model"] == "primary":
                raise RuntimeError("retryable primary failure")
            return Response()

    class Client:
        models = Models()

        def close(self):
            pass

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="primary",
                                       fallback_models=["backup"])
    result = op.run_opening_stage(
        packet=_packet(), provider=provider,
        budget=_budget(max_requests=4))
    assert calls == ["primary", "backup"], \
        "configured fallback chain is identity-only and never attempted"
    assert result["fallback_calls"] == 1
    assert result["provider_calls"] == 1


def test_fallback_failure_is_charged_and_recorded():
    calls = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            calls.append(kw["model"])
            raise RuntimeError("all models down")

    class Client:
        models = Models()

        def close(self):
            pass

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="primary",
                                       fallback_models=["backup"])
    result = op.run_opening_stage(
        packet=_packet(), provider=provider,
        budget=_budget(max_requests=4))
    assert calls == ["primary", "backup"]
    assert result["fallback_calls"] == 1
    assert result["selected"] is None
    assert result["no_selection_reason"] == "provider_failed"
    # Every attempt was charged exactly once.
    assert len(result["attempts"]) == 2


def test_fallback_stops_when_budget_exhausted():
    calls = []

    class Response:
        text = "[]"

    class Models:
        def generate_content(self, **kw):
            calls.append(kw["model"])
            raise RuntimeError("primary down")

    class Client:
        models = Models()

        def close(self):
            pass

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="primary",
                                       fallback_models=["backup"])
    result = op.run_opening_stage(
        packet=_packet(), provider=provider,
        budget=_budget(max_requests=1))
    # Only one charge available: the fallback never runs.
    assert calls == ["primary"]
    assert result["no_selection_reason"] == "budget_exhausted"


# ---------------------------------------------------------------------------
# Real emission/export: source mutation kills the overlay
# ---------------------------------------------------------------------------

def test_emission_source_mutation_kills_overlay(av_job, monkeypatch):
    """Real _build_project path: changing the ORIGINAL source file
    invalidates the plan at emission. The fingerprint is recomputed
    from the authoritative original asset (threaded through planning),
    never the cut slice, and never trusted from the saved plan."""
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    from clippyme.pipeline.project_emit import _build_project
    from clippyme.domain.clip_project import validate_project
    js, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1"}
    clips_data = {"transcript": {"segments": [{"words": [
        {"word": "hello", "start": 0.5, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 1.5}]}]}}
    good = dict(_option(), text="hello world",
                evidence=[{"basis": "speech", "quote": "hello world"}],
                display={"start": 0.0, "end": 2.0})
    op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=src, index=0,
        budget=_budget(), provider=_FakeProvider([good]))
    assert clip["opening_plan"]["selected"] is not None
    assert clip["opening_source_path"] == src
    data = _build_project(
        job_id="t15", index=0, start=0.0, end=6.0, clip=clip,
        clips_data=clips_data, clip_source=src, clip_final=final,
        job_dir=js)
    assert [o for o in validate_project(data).overlays
            if o.type == "hook"] != []
    # MUTATION: the original source file changed (new bytes at the same
    # path the plan was built from). The emission identity no longer
    # matches, so no automatic hook.
    with open(src, "rb") as fh:
        blob = fh.read()
    with open(src, "wb") as fh:
        fh.write(blob + b"\x00" * 70000)
    data = _build_project(
        job_id="t15", index=0, start=0.0, end=6.0, clip=clip,
        clips_data=clips_data, clip_source=src, clip_final=final,
        job_dir=js)
    project = validate_project(data)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.no_selection_reason == "current_packet_unavailable"
