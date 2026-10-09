"""Task 15 review-01: durable promotion of the 11 coordinator probes.

Every probe from work/muse-handoff/task-15/coordinator-review1-remaining.py
is promoted here verbatim, PLUS real-route tests that execute the actual
direct/queued wiring with injected I/O/SDK (static wiring probes alone are
insufficient — these run the real entry points, not just route strings).

Injected providers only; synthetic fixtures; no live providers, no
downloads, no new dependencies.
"""
import ast
import json
import os
import subprocess
import sys
import types as _types

import pytest

from clippyme.pipeline import opening_plan as op
from clippyme.pipeline.project_emit import _grounded_hook_overlay


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _packet(**kwargs):
    args = dict(
        source_fingerprint="source", final_start=10., final_end=20.,
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


class _FakeProvider:
    def __init__(self, options=None, error=None):
        self._options = options if options is not None else []
        self._error = error
        self.calls = 0
        self.closed = 0
        self.requests = []

    def request_options(self, request):
        self.calls += 1
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return self._options

    def close(self):
        self.closed += 1


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    op.set_opening_provider(None)
    op.set_opening_fallback_provider(None)
    op.set_opening_provider_factory(None)
    op.reset_run_budget()
    yield
    op.set_opening_provider(None)
    op.set_opening_fallback_provider(None)
    op.set_opening_provider_factory(None)
    op.reset_run_budget()


def _budget(**kw):
    args = dict(deadline_s=60., max_requests=8, request_timeout_s=10.)
    args.update(kw)
    return op.OpeningBudget(**args)


# ---------------------------------------------------------------------------
# Promoted probe 1: the direct CLI calls an opening stage (static)
# ---------------------------------------------------------------------------

def test_direct_cli_reaches_opening_stage():
    from pathlib import Path
    # Repo root is two levels above tests/pipeline/.
    repo = Path(__file__).resolve().parents[2]
    tree = ast.parse(
        (repo / "src/clippyme/pipeline/main.py").read_text(encoding="utf-8"))
    names = [ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call)]
    assert any("opening" in name for name in names), \
        "actual main.py CLI never calls an opening stage"


# ---------------------------------------------------------------------------
# Promoted probe 2: job entry initializes its own budget (static + real)
# ---------------------------------------------------------------------------

def test_new_job_does_not_inherit_exhausted_process_budget(monkeypatch):
    # Production must explicitly create/reset invocation ownership at job entry.
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    src = (repo / "src/clippyme/pipeline/orchestrator.py").read_text(
        encoding="utf-8")
    assert "reset_run_budget(" in src or "opening_budget_from_env(" in src, \
        "job entry never initializes its own opening budget"


def test_two_jobs_do_not_share_budget_real_behavior(
        clean_registry, monkeypatch):
    """Real route behavior: an exhausted process budget does not leak into
    the next job once the job entry resets it."""
    monkeypatch.setenv("CLIPPYME_OPENING_MAX_REQUESTS", "1")
    pkt = _packet()
    opt = _option()

    # Job 1: one request allowed; the second clip exhausts the budget.
    op.reset_run_budget()  # what orchestrator.run()/main.__main__ do
    run1 = op.run_opening_stage(packet=pkt, provider=_FakeProvider([opt]))
    assert run1["selected"] is not None
    run1b = op.run_opening_stage(packet=pkt, provider=_FakeProvider([opt]))
    assert run1b["selected"] is None
    assert run1b["no_selection_reason"] == "budget_exhausted"

    # Job 2 entry resets: the new job selects again.
    op.reset_run_budget()
    run2 = op.run_opening_stage(packet=pkt, provider=_FakeProvider([opt]))
    assert run2["selected"] is not None
    assert run2["selected"]["id"] == "x"


# ---------------------------------------------------------------------------
# Promoted probe 3: one job's shared provider survives both clips
# ---------------------------------------------------------------------------

def test_strict_client_survives_two_clips_with_correct_ownership():
    class Provider:
        closed = 0

        def request_options(self, request):
            if self.closed:
                raise RuntimeError("client closed")
            return [_option()]

        def close(self):
            self.closed += 1

    provider = Provider()
    budget = _budget()
    a = op.run_opening_stage(packet=_packet(), provider=provider,
                             budget=budget)
    b = op.run_opening_stage(packet=_packet(), provider=provider,
                             budget=budget)
    assert a["selected"] is not None
    assert b["selected"] is not None, \
        "registry reuses a client closed after the preceding clip"
    # Caller-owned: the stage never closed the shared provider.
    assert provider.closed == 0


def test_factory_built_provider_closed_exactly_once():
    built = []

    class _P:
        closed = 0

        def request_options(self, request):
            return [_option()]

        def close(self):
            self.closed += 1

    def factory():
        p = _P()
        built.append(p)
        return p

    run = op.run_opening_stage(packet=_packet(), provider_factory=factory,
                               budget=_budget())
    assert run["selected"] is not None
    assert len(built) == 1
    assert built[0].closed == 1  # stage-built: closed exactly once


def test_factory_setup_failure_isolated():
    def bad_factory():
        raise RuntimeError("no credentials")

    run = op.run_opening_stage(packet=_packet(), provider_factory=bad_factory,
                               budget=_budget())
    assert run["selected"] is None
    assert run["no_selection_reason"] == "no_provider_configured"
    assert any(f["stage"] == "primary_setup" for f in run["failures"])


def test_two_clips_two_factories_no_client_reuse():
    seen = []

    def factory():
        p = _FakeProvider([_option()])
        seen.append(p)
        return p

    budget = _budget()
    a = op.run_opening_stage(packet=_packet(), provider_factory=factory,
                             budget=budget)
    b = op.run_opening_stage(packet=_packet(), provider_factory=factory,
                             budget=budget)
    assert a["selected"] is not None and b["selected"] is not None
    assert seen[0] is not seen[1]  # never reuse a closed client
    assert all(p.closed == 1 for p in seen)


# ---------------------------------------------------------------------------
# Promoted probes 4-6: material claim support (invented people/events)
# ---------------------------------------------------------------------------

def test_speech_overlap_is_not_proof_of_invented_person_and_event():
    accepted, reason = op.validate_opening_option(
        _option(text="Alice won the coffee works championship"),
        packet=_packet())
    assert accepted is None, \
        "50% token overlap blessed an invented person/event as grounded"


def test_visual_label_is_not_proof_of_unrelated_hook():
    p = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12.}])
    accepted, reason = op.validate_opening_option(
        _option(text="Alice wins the gold medal", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 11., "end": 12.}]),
        packet=p)
    assert accepted is None, "unrelated visual hook marked grounded"


def test_sensory_citation_interval_matches_the_event():
    p = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12.}])
    accepted, reason = op.validate_opening_option(
        _option(text="red ball rolls", evidence_basis="visual",
                evidence=[{"basis": "visual", "label": "red ball rolls",
                           "start": 18., "end": 19.}]),
        packet=p)
    assert accepted is None, \
        "label-only matching accepts a fabricated evidence interval"


def test_valid_paraphrase_control_still_passes():
    pkt = op.build_opening_packet(
        source_fingerprint="fp", final_start=10., final_end=20.,
        words=[{"word": "hello", "start": 10.5, "end": 11.0},
               {"word": "world", "start": 11.0, "end": 11.5},
               {"word": "this", "start": 12.0, "end": 12.3},
               {"word": "is", "start": 12.3, "end": 12.4},
               {"word": "amazing", "start": 12.4, "end": 13.0}],
        language="en")
    norm, rej = op.validate_opening_option(
        _option(text="hello, this is amazing",
                evidence=[{"basis": "speech",
                           "quote": "hello world this is amazing"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["grounded"] is True


def test_non_english_control_still_passes():
    pkt = op.build_opening_packet(
        source_fingerprint="fp", final_start=10., final_end=20.,
        words=[{"word": "kahvi", "start": 10.5, "end": 11.0},
               {"word": "toimii", "start": 11.0, "end": 11.5}],
        language="fi")
    norm, rej = op.validate_opening_option(
        _option(text="kahvi toimii", language="fi", rationale="suomeksi",
                evidence=[{"basis": "speech", "quote": "kahvi toimii"}]),
        packet=pkt)
    assert rej is None, rej
    assert norm["language"] == "fi"


def test_unrecognized_language_rejected():
    accepted, rej = op.validate_opening_option(
        _option(language="xx-not-a-language"), packet=_packet())
    assert accepted is None
    assert rej["code"] == "bad_language"


def test_language_mismatch_rejected():
    pkt = _packet(language="fi")
    accepted, rej = op.validate_opening_option(
        _option(text="kahvi toimii", language="en", rationale="r",
                evidence=[{"basis": "speech", "quote": "coffee works"}]),
        packet=pkt)
    assert accepted is None
    assert rej["code"] == "language_mismatch"


def test_explicit_translation_preference_honored_in_prompt_and_validation():
    pkt = _packet(language="fi", preferences={"translate_to": "en"})
    prompt = op.build_opening_prompt(pkt)
    assert "'en'" in prompt and "translation" in prompt.lower()
    # An English option is accepted when translation was requested...
    norm, rej = op.validate_opening_option(
        _option(text="coffee works",
                rationale="translation of the Finnish line",
                evidence=[{"basis": "speech", "quote": "coffee works"}]),
        packet=pkt)
    assert rej is None, rej
    # ...but a Finnish option is not (expected language is 'en').
    accepted, rej = op.validate_opening_option(
        _option(text="kahvi toimii", language="fi", rationale="r",
                evidence=[{"basis": "speech", "quote": "coffee works"}]),
        packet=pkt)
    assert accepted is None
    assert rej["code"] == "language_mismatch"


def test_mutation_material_claim_gate_is_load_bearing(clean_registry):
    pkt = op.build_opening_packet(
        source_fingerprint="fp", final_start=10., final_end=20.,
        words=[{"word": "hello", "start": 10.5, "end": 11.0},
               {"word": "world", "start": 11.0, "end": 11.5}],
        language="en")
    bad = _option(text="Alice won the coffee works championship",
                  evidence=[{"basis": "speech", "quote": "hello world"}])
    accepted, _ = op.validate_opening_option(bad, packet=pkt)
    assert accepted is None
    # MUTATION: neutralize the glue-vocabulary check.
    real = op._GLUE_TOKENS
    op._GLUE_TOKENS = frozenset({"a", "the"})
    try:
        accepted, _ = op.validate_opening_option(
            _option(text="hello there",
                    evidence=[{"basis": "speech", "quote": "hello"}]),
            packet=pkt)
        assert accepted is None, "mutation did not break the gate"
    finally:
        op._GLUE_TOKENS = real
    # RESTORE.
    norm, _ = op.validate_opening_option(
        _option(text="hello there",
                evidence=[{"basis": "speech", "quote": "hello"}]),
        packet=pkt)
    assert norm is not None


# ---------------------------------------------------------------------------
# Promoted probes 7-8: packet identity binds AV + exact bounds
# ---------------------------------------------------------------------------

def test_av_identity_changes_with_evidence_time():
    a = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12.}])
    b = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 18., "end": 19.}])
    assert a["packet_id"] != b["packet_id"], \
        "AV evidence absent from packet identity"


def test_exact_cut_bounds_are_part_of_identity():
    assert _packet(final_end=20.0001)["packet_id"] != \
        _packet(final_end=20.0002)["packet_id"], \
        "rounding aliases different cutter intervals"


def test_same_label_different_events_distinguishable():
    a = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls", "start": 11.,
         "end": 12., "event_id": "ev-1"}])
    b = _packet(words=[], av_evidence=[
        {"kind": "visual", "label": "red ball rolls", "start": 11.,
         "end": 12., "event_id": "ev-2"}])
    assert a["packet_id"] != b["packet_id"]


def test_tiny_word_boundary_change_distinguishable():
    a = _packet(words=[{"word": "coffee", "start": 11.00001, "end": 11.4}])
    b = _packet(words=[{"word": "coffee", "start": 11.00002, "end": 11.4}])
    assert a["packet_id"] != b["packet_id"]


# ---------------------------------------------------------------------------
# Promoted probe 9: Task 04 speaker contract
# ---------------------------------------------------------------------------

def test_integer_speaker_zero_preserved_without_segment_label_invention():
    transcript = {"segments": [{"speaker": "segment-only", "words": [
        {"word": "coffee", "start": 11., "end": 11.4, "speaker": 0},
        {"word": "works", "start": 11.4, "end": 12.}]}]}
    words = op._window_transcript_words(transcript, 10., 20.)
    p = _packet(words=words)
    assert p["words"][0].get("speaker") == 0, "word-level speaker0 discarded"
    assert "speaker" not in p["words"][1], \
        "segment-only label copied into an unknown word"


def test_word_speaker_contract_matches_task04_normalizer():
    from clippyme.pipeline import gemini_request as gr
    cases = [0, 1, 7, -1, True, False, 1.0, 2.0, 1.5, "x", None, "0"]
    for value in cases:
        assert op._canonical_word_speaker(value) == \
            gr._canonical_speaker(value), \
            f"Task 04 parity broken for {value!r}"


def test_conflicting_segment_summary_never_wins():
    transcript = {"segments": [{"speaker": 9, "words": [
        {"word": "a", "start": 11., "end": 11.4, "speaker": 0},
        {"word": "b", "start": 11.4, "end": 12., "speaker": 3},
        {"word": "c", "start": 12., "end": 12.4}]}]}
    words = op._window_transcript_words(transcript, 10., 20.)
    assert words[0]["speaker"] == 0   # word-level wins, 0 preserved
    assert words[1]["speaker"] == 3   # explicit word label wins
    assert "speaker" not in words[2]  # missing stays unknown
    assert all(w.get("speaker") != 9 for w in words)  # segment never copied


# ---------------------------------------------------------------------------
# Promoted probe 10: emission fails closed
# ---------------------------------------------------------------------------

def test_emitter_cannot_bless_unvalidated_selected_dictionary():
    clip = {"opening_plan": {
        "selected": _option(text="Alice wins the gold medal"),
        "validated_options": []}}
    overlays, opening = _grounded_hook_overlay(clip, 10.)
    assert overlays == [], \
        "emission trusts selected.text without validated current identity/evidence"


def test_emission_membership_is_load_bearing(av_job, clean_registry,
                                             monkeypatch):
    """Real export path: a forged selection never becomes an overlay."""
    import json as _json
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    from clippyme.pipeline.project_emit import _build_project
    js, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1",
            "viral_hook_text": "legacy hook",
            "opening_plan": {
                "schema_version": op.OPENING_PLAN_SCHEMA_VERSION,
                "packet_id": "forged",
                "selected": _option(id="evil", text="Alice wins"),
                "validated_options": [],
            }}
    clips_data = {"transcript": {"segments": []}}
    data = _build_project(
        job_id="t15", index=0, start=0.0, end=6.0, clip=clip,
        clips_data=clips_data, clip_source=src, clip_final=final, job_dir=js)
    from clippyme.domain.clip_project import validate_project
    project = validate_project(data)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.status == "none"
    assert project.opening.no_selection_reason == "selection_not_validated"


def test_emission_current_bound_mutation_kills_overlay(av_job, clean_registry,
                                                      monkeypatch):
    """Real export path: planning at [0,6) then mutating the bound to 7
    invalidates the plan at emission (packet id recomputed from the
    current clip)."""
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    from clippyme.pipeline.project_emit import _build_project
    from clippyme.domain.clip_project import validate_project
    js, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1",
            "viral_hook_text": "legacy hook"}
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
    # Sanity: unmutated emission carries the overlay.
    data = _build_project(
        job_id="t15", index=0, start=0.0, end=6.0, clip=clip,
        clips_data=clips_data, clip_source=src, clip_final=final, job_dir=js)
    assert [o for o in validate_project(data).overlays
            if o.type == "hook"] != []
    # MUTATION: the current cut moved; the stale plan must not emit.
    clip["end"] = 7.0
    data = _build_project(
        job_id="t15", index=0, start=0.0, end=7.0, clip=clip,
        clips_data=clips_data, clip_source=src, clip_final=final, job_dir=js)
    project = validate_project(data)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.no_selection_reason == "packet_identity_changed"


def test_stale_automatic_plan_cleared_on_packet_failure(
        tmp_path, clean_registry):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1"}
    clips_data = {"transcript": {"segments": [{"words": [
        {"word": "hello", "start": 0.5, "end": 1.0}]}]}}
    good = dict(_option(), text="hello",
                evidence=[{"basis": "speech", "quote": "hello"}],
                display={"start": 0.0, "end": 2.0})
    op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=_FakeProvider([good]))
    assert clip["opening_plan"]["selected"] is not None
    # Packet construction now fails (non-finite bound): the stale
    # automatic plan must be cleared, not left for emission.
    clip["end"] = float("nan")
    with pytest.raises(op.PacketBuildError):
        op.plan_clip_opening(
            clip=clip, clips_data=clips_data, input_video=str(src), index=0,
            budget=_budget(), provider=_FakeProvider([good]))
    assert "opening_plan" not in clip
    funnel = clips_data["opening_funnel"]
    assert funnel["clips"]["clip-0"]["no_selection_reason"] == \
        "packet_build_failed"


def test_user_override_survives_packet_failure(tmp_path, clean_registry):
    """Override handling runs before packet construction: the authorized
    override is never blocked by a packet failure."""
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 0.0, "end": float("nan"), "candidate_id": "c1",
            "hook": "My hand-written hook"}
    clips_data = {"transcript": {"segments": []}}
    plan = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=_FakeProvider([_option()]))
    assert plan["user_authored"] is True
    assert plan["text"] == "My hand-written hook"


def test_transient_no_selection_never_resumes(tmp_path, clean_registry):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1"}
    clips_data = {"transcript": {"segments": [{"words": [
        {"word": "hello", "start": 0.5, "end": 1.0}]}]}}
    # No provider -> honest transient no-selection with empty options.
    plan1 = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=None)
    assert plan1["selected"] is None
    assert plan1["no_selection_reason"] == "no_provider_configured"
    # A second attempt must re-analyze, not resume the transient plan.
    good = dict(_option(), text="hello",
                evidence=[{"basis": "speech", "quote": "hello"}],
                display={"start": 0.0, "end": 2.0})
    plan2 = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=_FakeProvider([good]))
    assert plan2.get("resumed") is not True
    assert plan2["selected"] is not None


# ---------------------------------------------------------------------------
# Promoted probe 11: sub-millisecond budget
# ---------------------------------------------------------------------------

def test_submillisecond_budget_is_positive_and_not_above_remaining():
    budget = op.OpeningBudget(deadline_s=.0004, max_requests=1,
                              request_timeout_s=10., clock=lambda: 0.)
    ok, timeout = budget.try_charge()
    assert ok and 0 < timeout <= .0004, \
        "positive remaining deadline rounded to zero timeout"


def test_submillisecond_timeout_reaches_provider_request():
    clock = [0.0]
    budget = op.OpeningBudget(deadline_s=.0004, max_requests=1,
                              request_timeout_s=10.,
                              clock=lambda: clock[0])
    prov = _FakeProvider([])
    op.run_opening_stage(packet=_packet(), provider=prov, budget=budget)
    assert prov.calls == 1
    timeout = prov.requests[0]["timeout_s"]
    assert 0 < timeout <= .0004


# ---------------------------------------------------------------------------
# Lazy production adapter: injected SDK, no live calls
# ---------------------------------------------------------------------------

class _FakeSDKClient:
    instances = []

    def __init__(self, api_key=None, produce=None):
        self.api_key = api_key
        self.produce = produce
        self.calls = []
        self.closed = 0
        _FakeSDKClient.instances.append(self)
        self.models = self

    def generate_content(self, **kwargs):
        # Exactly one SDK call per request_options: record and answer.
        self.calls.append(kwargs)
        return self.produce(kwargs)

    def close(self):
        self.closed += 1


class _FakeSDKResponse:
    def __init__(self, text):
        self.text = text


def _fake_sdk(produce):
    class _SDK:
        @staticmethod
        def Client(api_key=None):
            return _FakeSDKClient(api_key=api_key, produce=produce)
    return _SDK


def _sdk_option_payload():
    return json.dumps([{
        "id": "sdk-1", "text": "coffee works", "language": "en",
        "rationale": "opens on the retained words",
        "evidence_basis": "speech",
        "evidence": [{"basis": "speech", "quote": "coffee works"}],
        "display": {"start": 0.0, "end": 2.0},
    }])


def test_adapter_uses_production_prompt_parsing_timeout_retry():
    pkt = _packet()
    captured = {}

    def produce(kwargs):
        captured.update(kwargs)
        return _FakeSDKResponse(_sdk_option_payload())

    sdk = _fake_sdk(produce)
    provider = op.GeminiOpeningProvider(
        client_factory=lambda: sdk.Client(api_key="test-key"),
        model="gemini-3.5-flash-lite")
    request = {"prompt": op.build_opening_prompt(pkt), "packet": pkt,
               "timeout_s": 4.5, "max_options": 3, "retries_disabled": True}
    options = provider.request_options(request)
    # Production prompt handed to the SDK verbatim.
    assert captured["contents"] == op.build_opening_prompt(pkt)
    assert captured["model"] == "gemini-3.5-flash-lite"
    # Response parsing: JSON array -> option dicts.
    assert options[0]["id"] == "sdk-1"
    assert options[0]["text"] == "coffee works"
    # Timeout from the budgeted per-attempt timeout...
    http_options = captured["config"]["http_options"]
    assert http_options["timeout"] == 4500
    # ...and retries disabled in the adapter's own call config.
    assert http_options["retry_options"] == {"attempts": 1}
    # Exactly one SDK call per request (no adapter retry loop).
    assert len(_FakeSDKClient.instances[-1].calls) == 1


def test_adapter_makes_no_retry_loop_on_transient_failure():
    pkt = _packet()
    calls = []

    def produce(kwargs):
        calls.append(kwargs)
        raise RuntimeError("transient 500")

    sdk = _fake_sdk(produce)
    provider = op.GeminiOpeningProvider(
        client_factory=lambda: sdk.Client(api_key="test-key"),
        model="gemini-3.5-flash-lite")
    with pytest.raises(RuntimeError):
        provider.request_options(
            {"prompt": "p", "packet": pkt, "timeout_s": 2.0,
             "max_options": 3, "retries_disabled": True})
    assert len(calls) == 1  # no internal retry loop
    # The client was still closed exactly once.
    assert _FakeSDKClient.instances[-1].closed == 1


def test_adapter_rejects_malformed_responses():
    pkt = _packet()
    provider = op.GeminiOpeningProvider(
        client_factory=lambda: _FakeSDKClient(produce=lambda kw: None),
        model="m")

    def run_with(text):
        client = _FakeSDKClient(
            produce=lambda kw: _FakeSDKResponse(text))
        provider._client_factory = lambda: client
        return provider.request_options(
            {"prompt": "p", "packet": pkt, "timeout_s": 2.0,
             "max_options": 3, "retries_disabled": True})

    with pytest.raises(ValueError):
        run_with("not json{{")
    with pytest.raises(ValueError):
        run_with('{"not": "an array"}')
    # Fenced JSON is accepted.
    opts = run_with("```json\n" + _sdk_option_payload() + "\n```")
    assert opts[0]["id"] == "sdk-1"


def test_default_factory_is_lazy_and_replaceable(clean_registry,
                                                 monkeypatch):
    # The default factory builds no client until a request needs one,
    # and reads the supported config (model/fallback) without network.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_FALLBACK_MODELS", "gemini-x, gemini-y")
    factory = op.default_opening_provider_factory()
    config = factory.describe()
    assert config["kind"] == "gemini-opening"
    assert config["fallback_models"] == ["gemini-x", "gemini-y"]
    assert config["prompt_version"] == op.OPENING_PROMPT_VERSION
    # Replaceable: an injected factory wins over the default.
    sentinel = object()
    op.set_opening_provider_factory(lambda: sentinel)
    assert op.get_opening_provider_factory()() is sentinel


def test_adapter_end_to_end_selects_via_stage():
    """Injected SDK -> adapter -> stage: a grounded option is selected."""
    pkt = _packet()
    sdk = _fake_sdk(lambda kw: _FakeSDKResponse(_sdk_option_payload()))
    factory = op.GeminiOpeningProviderFactory(sdk=sdk)
    run = op.run_opening_stage(packet=pkt, provider_factory=factory,
                               budget=_budget())
    assert run["selected"] is not None
    assert run["selected"]["id"] == "sdk-1"
    assert run["selected"]["grounded"] is True
    # Every attempt charged exactly once.
    assert len(run["attempts"]) == 1
    assert run["attempts"][0]["ok"] is True


# ---------------------------------------------------------------------------
# Real direct-route wiring: the actual helper the CLI loop calls
# ---------------------------------------------------------------------------

def _install_main_stubs():
    """Stub main.py's missing heavy deps (same pattern as
    test_av_acceptance); skip anything already present."""
    import sys as _sys
    import types as _t

    def _ensure(name, make):
        if name in _sys.modules:
            return
        try:
            __import__(name)
        except ImportError:
            _sys.modules[name] = make()

    def _mk_ultralytics():
        m = _t.ModuleType("ultralytics")
        m.YOLO = type("YOLO", (), {})
        return m

    def _mk_torch():
        m = _t.ModuleType("torch")
        cuda = _t.ModuleType("torch.cuda")
        cuda.is_available = lambda: False
        m.cuda = cuda
        m.device = lambda x: x
        _sys.modules["torch.cuda"] = cuda
        return m

    _ensure("ultralytics", _mk_ultralytics)
    _ensure("torch", _mk_torch)
    _ensure("mediapipe", lambda: _t.ModuleType("mediapipe"))
    # google.genai stub: keep the real `google` namespace package, only
    # register the submodule (the exercised helper never touches it).
    if "google.genai" not in _sys.modules:
        try:
            __import__("google.genai")
        except ImportError:
            import google  # noqa: F401 (real namespace package, intact)
            _genai = _t.ModuleType("google.genai")
            _genai.Client = type("Client", (), {})
            _sys.modules["google.genai"] = _genai
    if "psutil" not in _sys.modules:
        try:
            __import__("psutil")
        except ImportError:
            pm = _t.ModuleType("psutil")

            class _VM:
                total = 16 * 1024 ** 3
                available = 8 * 1024 ** 3
                percent = 50.0

            pm.virtual_memory = lambda: _VM()
            pm.cpu_count = lambda logical=True: 8
            _sys.modules["psutil"] = pm


def test_direct_route_helper_reaches_packet_with_final_bounds(
        tmp_path, clean_registry, monkeypatch):
    """Execute the actual direct-path wiring (the helper the CLI cutter
    loop calls) with injected provider I/O."""
    _install_main_stubs()
    from clippyme.pipeline.main import _plan_opening_for_direct_clip
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 1024)
    meta = str(tmp_path / "meta.json")
    clip = {"start": 10.0, "end": 20.0, "candidate_id": "c-direct"}
    clips_data = {"shorts": [clip], "transcript": {"segments": [
        {"words": [
            {"word": "coffee", "start": 11.0, "end": 11.4},
            {"word": "works", "start": 11.4, "end": 12.0},
            {"word": "leadin", "start": 5.0, "end": 5.5},
        ]}]}}
    good = dict(_option(), text="coffee works",
                evidence=[{"basis": "speech", "quote": "coffee works"}],
                display={"start": 0.0, "end": 2.0})
    prov = _FakeProvider([good])
    op.set_opening_provider(prov)
    plan = _plan_opening_for_direct_clip(
        clip, clips_data, str(src), 0, meta)
    assert plan is not None
    assert plan["route"] == "direct"
    assert prov.calls == 1
    # FINAL cutter bounds reached the packet; lead-in never leaked.
    assert plan["packet_identity"]["final_start"] == 10.0
    assert plan["packet_identity"]["final_end"] == 20.0
    assert plan["selected"]["id"] == "x"
    # Atomic metadata persistence carries the plan + funnel.
    saved = json.load(open(meta, encoding="utf-8"))
    assert saved["shorts"][0]["opening_plan"]["packet_id"] == \
        plan["packet_id"]
    assert saved["opening_funnel"]["totals"]["clips"] == 1


def test_direct_route_helper_flag_off_is_noop(tmp_path, clean_registry,
                                              monkeypatch):
    _install_main_stubs()
    from clippyme.pipeline.main import _plan_opening_for_direct_clip
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    meta = str(tmp_path / "meta.json")
    clip = {"start": 10.0, "end": 20.0}
    clips_data = {"shorts": [clip], "transcript": {"segments": []}}
    prov = _FakeProvider([_option()])
    op.set_opening_provider(prov)
    assert _plan_opening_for_direct_clip(
        clip, clips_data, str(src), 0, meta) is None
    assert prov.calls == 0
    assert "opening_plan" not in clip


# ---------------------------------------------------------------------------
# Real queued-route wiring: _render_one_clip (resume path) with injected I/O
# ---------------------------------------------------------------------------

def test_queued_render_one_clip_reaches_opening_stage(
        tmp_path, clean_registry, monkeypatch):
    """Execute the actual queued route (_render_one_clip) with injected
    I/O: the opening stage runs with route='queued' before the cutter."""
    import argparse
    from clippyme.pipeline import orchestrator as orch
    from clippyme.domain.runtime_state import RuntimeState
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")

    out = tmp_path / "job"
    out.mkdir()
    meta = str(out / "meta.json")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 2048)
    # A finished final clip -> the resume path (no ffmpeg cutter runs).
    clip_final = out / "clip0.mp4"
    clip_final.write_bytes(b"\x00" * 20000)
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c-queued",
            "clip_filename": "clip0.mp4"}
    clips_data = {"shorts": [clip], "transcript": {"segments": [
        {"words": [
            {"word": "hello", "start": 0.5, "end": 1.0},
            {"word": "world", "start": 1.0, "end": 1.5},
        ]}]}}
    good = dict(_option(), text="hello world",
                evidence=[{"basis": "speech", "quote": "hello world"}],
                display={"start": 0.0, "end": 2.0})
    prov = _FakeProvider([good])
    op.set_opening_provider(prov)
    op.reset_run_budget()

    monkeypatch.setattr(
        orch, "inspect_clip",
        lambda *a, **k: {"critical": False, "issues": [], "warnings": []})
    emitted = []
    monkeypatch.setattr(
        orch, "_maybe_emit_clip_project",
        lambda **k: emitted.append(k) or "clip-project-0.json")

    args = argparse.Namespace(reframe_mode="auto", no_zoom=False)
    state = RuntimeState(str(out), job_id="t15q")
    ok = orch._render_one_clip(
        index=0, total=1, clip=clip, input_video=str(src),
        video_title="t", output_dir=str(out), metadata_file=meta,
        clips_data=clips_data, args=args, aspect_ratio=9 / 16,
        state=state, legacy=_types.SimpleNamespace())
    assert ok is True
    plan = clip.get("opening_plan")
    assert plan is not None, "queued route never ran the opening stage"
    assert plan["route"] == "queued"
    assert prov.calls == 1
    assert plan["packet_identity"]["final_start"] == 0.0
    assert plan["packet_identity"]["final_end"] == 6.0
    assert plan["selected"]["id"] == "x"
    assert emitted, "project emission not reached on the queued route"


# ---------------------------------------------------------------------------
# Task 11 AV evidence connector
# ---------------------------------------------------------------------------

def test_task11_candidate_evidence_reaches_packet(tmp_path, clean_registry):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 10.0, "end": 20.0, "candidate_id": "avcand-1"}
    clips_data = {
        "transcript": {"segments": []},
        "candidate_store": {
            "avcand-1": {
                "candidate_id": "avcand-1",
                "evidence_basis": "visual",
                "visual_description": "red ball rolls",
                "start_s": 11.0, "end_s": 12.0,
            },
        },
    }
    plan = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=_FakeProvider([]))
    ident = plan["packet_identity"]
    # The Task 11 evidence changed the identity (av_sha present).
    assert ident["av_sha256"] != op._av_identity_hash([])
    # A speech-free visual option can now ground on the Task 11 event.
    pkt = op.build_opening_packet(
        source_fingerprint="s", final_start=10., final_end=20., words=[],
        av_evidence=op._opening_av_evidence_for_clip(
            clip, clips_data, 10., 20.),
        language="en")
    norm, rej = op.validate_opening_option(
        {"id": "v", "text": "red ball rolls", "language": "en",
         "rationale": "r", "evidence_basis": "visual",
         "evidence": [{"basis": "visual", "label": "red ball rolls",
                       "start": 11.0, "end": 12.0}],
         "display": {"start": 0.0, "end": 2.0}},
        packet=pkt)
    assert rej is None, rej


def test_legacy_av_evidence_alias_still_works(tmp_path, clean_registry):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 10.0, "end": 20.0, "candidate_id": "c1",
            "opening_av_evidence": [
                {"kind": "visual", "label": "red ball rolls",
                 "start": 11.0, "end": 12.0}]}
    clips_data = {"transcript": {"segments": []}}  # no candidate_store
    plan = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=_FakeProvider([]))
    assert plan["packet_identity"]["av_sha256"] != \
        op._av_identity_hash([])


def test_prompt_labels_av_times_as_absolute_source():
    pkt = _packet(av_evidence=[
        {"kind": "visual", "label": "red ball rolls",
         "start": 11., "end": 12.}])
    prompt = op.build_opening_prompt(pkt)
    assert "ABSOLUTE source" in prompt
    assert "abs 11.00-12.00" in prompt
    assert "clip 1.00-2.00" in prompt  # derived clip-relative display


# ---------------------------------------------------------------------------
# Funnel: no-selection with empty options is never a resumable success
# ---------------------------------------------------------------------------

def test_funnel_no_selection_empty_options_not_resumable_as_success(
        tmp_path, clean_registry):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "c1"}
    clips_data = {"transcript": {"segments": []}}
    plan = op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=_budget(), provider=None)
    assert plan["no_selection_reason"] == "no_provider_configured"
    pkt = op.build_opening_packet(
        source_fingerprint=clip["opening_source_fingerprint"],
        final_start=0., final_end=6., words=[], language=None)
    ok, reason = op.opening_plan_resumable(clip["opening_plan"], pkt)
    assert not ok, "transient empty no-selection resumed as success"


# ---------------------------------------------------------------------------
# Fail-then-restore: emission membership gate
# ---------------------------------------------------------------------------

def test_mutation_emission_membership_is_load_bearing(
        av_job, clean_registry, monkeypatch):
    from clippyme.pipeline import project_emit as pem
    js, src, final = av_job
    clip = {"opening_plan": {
        "selected": _option(text="Alice wins the gold medal"),
        "validated_options": []}}
    overlays, _ = pem._grounded_hook_overlay(clip, 10.)
    assert overlays == []
    # MUTATION: skip the membership check -> forged selection emits.
    def _no_membership(clip, duration, current_packet=None):
        plan = clip.get("opening_plan") or {}
        sel = plan.get("selected") or {}
        return ([{"type": "hook", "text": sel.get("text"),
                  "style": "viral", "start": 0.0, "end": 3.0,
                  "position": "top"}], {"status": "selected"})

    monkeypatch.setattr(pem, "_grounded_hook_overlay", _no_membership)
    overlays, _ = pem._grounded_hook_overlay(clip, 10.)
    assert overlays != [], "mutation did not break the gate"
    # RESTORE.
    monkeypatch.undo()
    overlays, _ = pem._grounded_hook_overlay(clip, 10.)
    assert overlays == []


# ---------------------------------------------------------------------------
# ffmpeg fixture for export-path tests
# ---------------------------------------------------------------------------

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
