"""Task 15 review-03: promoted probes + regressions + load-bearing mutations.

Covers the three remaining areas:
1. Original-source binding at emission; fail closed on identity failure.
2. Claim meaning: polarity preservation and sensory action grounding.
3. Sub-millisecond SDK timeouts: stop before the call, never timeout=0.

Injected providers only; synthetic AV fixtures; no live providers, no
downloads, no new dependencies.
"""
import os
import subprocess

import pytest

from clippyme.pipeline import opening_plan as op
from clippyme.pipeline.project_emit import _build_project
from clippyme.domain.clip_project import validate_project


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
    def __init__(self, options=None):
        self._options = options if options is not None else []
        self.calls = 0

    def request_options(self, request):
        self.calls += 1
        return self._options


def _mutate(name, value):
    real = getattr(op, name)
    setattr(op, name, value)
    return real


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
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", final],
        capture_output=True)
    assert r.returncode == 0, r.stderr[-500:]
    return js, src, final


def _av_words():
    return {"segments": [{"words": [
        {"word": "hello", "start": 0.5, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 1.5},
        {"word": "this", "start": 2.0, "end": 2.3},
        {"word": "is", "start": 2.3, "end": 2.4},
        {"word": "amazing", "start": 2.4, "end": 3.0},
    ]}]}


def _av_option(**kw):
    base = {
        "id": "op-1", "text": "hello world", "language": "en",
        "rationale": "opens on the exact first spoken words",
        "evidence_basis": "speech",
        "evidence": [{"quote": "hello world", "basis": "speech"}],
        "display": {"start": 2.0, "end": 4.0},
    }
    base.update(kw)
    return base


def _av_plan(clip, clips_data, src, options):
    budget = op.OpeningBudget(deadline_s=60, max_requests=4,
                              request_timeout_s=5)
    return op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=src, index=0,
        budget=budget, provider=_FakeProvider(options))


def _emit(job, src, final, clip, data, monkeypatch):
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    return _build_project(
        job_id="t15r3", index=0, start=clip["start"], end=clip["end"],
        clip=clip, clips_data=data, clip_source=src,
        clip_final=final, job_dir=job)


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout


# ---------------------------------------------------------------------------
# Promoted probe 1: real cut slice preserves the valid original-source hook
# ---------------------------------------------------------------------------

def test_real_cut_slice_preserves_valid_original_source_hook(
        av_job, monkeypatch):
    job, original, final = av_job
    clip = {"start": 0.25, "end": 5.25, "candidate_id": "cand-1"}
    data = {"transcript": _av_words()}
    _av_plan(clip, data, original, [_av_option()])
    assert clip["opening_plan"]["selected"], "positive planning control"
    assert clip["opening_source_path"] == original
    sliced = os.path.join(job, "actual_cut.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-ss", "0.25", "-i", original,
          "-t", "5", "-c:v", "libx264", "-c:a", "aac", sliced])
    p = _emit(job, sliced, final, clip, data, monkeypatch)
    hooks = [o for o in validate_project(p).overlays if o.type == "hook"]
    assert hooks != [], \
        "valid original-source plan suppressed by cut-slice identity"
    assert hooks[0].text == "hello world"


# ---------------------------------------------------------------------------
# Promoted probe 2: unreadable current source fails closed
# ---------------------------------------------------------------------------

def test_unreadable_current_source_fails_closed(av_job, monkeypatch):
    job, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1"}
    data = {"transcript": _av_words()}
    _av_plan(clip, data, src, [_av_option()])
    assert clip["opening_plan"]["selected"], "positive planning control"
    from clippyme.pipeline import framing_plan

    def unreadable(*a, **kw):
        raise OSError("injected unreadable identity")

    monkeypatch.setattr(framing_plan, "source_fingerprint", unreadable)
    p = _emit(job, src, final, clip, data, monkeypatch)
    project = validate_project(p)
    assert [o for o in project.overlays if o.type == "hook"] == [], \
        "failed current identity silently reused saved fingerprint"
    assert project.opening.no_selection_reason == "current_packet_unavailable"


# ---------------------------------------------------------------------------
# New: changed original source fails closed; missing path fails closed
# ---------------------------------------------------------------------------

def test_changed_original_source_fails_closed(av_job, monkeypatch):
    job, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1"}
    data = {"transcript": _av_words()}
    _av_plan(clip, data, src, [_av_option()])
    assert clip["opening_plan"]["selected"]
    # The original file changed after planning: new bytes, new identity.
    with open(src, "rb") as fh:
        blob = fh.read()
    with open(src, "wb") as fh:
        fh.write(blob + b"\x00" * 70000)
    p = _emit(job, src, final, clip, data, monkeypatch)
    project = validate_project(p)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.no_selection_reason == "current_packet_unavailable"


def test_missing_original_path_fails_closed(av_job, monkeypatch):
    job, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1"}
    data = {"transcript": _av_words()}
    _av_plan(clip, data, src, [_av_option()])
    assert clip["opening_plan"]["selected"]
    # The threaded original path was lost (e.g. old plan): no silent
    # trust in the saved fingerprint.
    del clip["opening_source_path"]
    p = _emit(job, src, final, clip, data, monkeypatch)
    project = validate_project(p)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.no_selection_reason == "current_packet_unavailable"


# ---------------------------------------------------------------------------
# New: resume keeps the original-source binding
# ---------------------------------------------------------------------------

def test_resume_preserves_original_source_binding(av_job, monkeypatch):
    job, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1"}
    data = {"transcript": _av_words()}
    _av_plan(clip, data, src, [_av_option()])
    assert clip["opening_plan"]["selected"]
    # Resume with no provider: the threaded original path survives and
    # emission still validates against the original asset.
    clip2 = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1",
             "opening_plan": clip["opening_plan"],
             "opening_source_fingerprint":
                 clip["opening_source_fingerprint"],
             "opening_source_path": clip["opening_source_path"]}
    data2 = {"transcript": _av_words()}
    resumed = op.plan_clip_opening(
        clip=clip2, clips_data=data2, input_video=src, index=0,
        budget=_budget(), provider=None)
    assert resumed["resumed"] is True
    assert clip2["opening_source_path"] == src
    p = _emit(job, src, final, clip2, data2, monkeypatch)
    hooks = [o for o in validate_project(p).overlays if o.type == "hook"]
    assert hooks != []
    assert hooks[0].text == "hello world"


# ---------------------------------------------------------------------------
# New: entry-point wiring threads the original path (queued route)
# ---------------------------------------------------------------------------

def test_queued_entry_point_threads_original_source_path(
        av_job, monkeypatch, tmp_path):
    from clippyme.pipeline.orchestrator import _maybe_plan_opening
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    job, src, final = av_job
    clip = {"start": 0.0, "end": 6.0, "candidate_id": "cand-1"}
    data = {"transcript": _av_words(), "shorts": [clip]}
    meta = str(tmp_path / "meta.json")
    op.reset_run_budget()
    op.set_opening_provider(_FakeProvider([_av_option()]))
    try:
        plan = _maybe_plan_opening(clip, data, src, 0, meta, route="queued")
    finally:
        op.set_opening_provider(None)
        op.reset_run_budget()
    assert plan is not None
    assert clip["opening_source_path"] == src
    assert clip["opening_plan"]["selected"] is not None


# ---------------------------------------------------------------------------
# Promoted probe 4: removing a negation is not grounded
# ---------------------------------------------------------------------------

def test_removing_negation_is_not_grounded():
    words = [{"word": w, "start": 1 + i * 0.3, "end": 1 + (i + 1) * 0.3}
             for i, w in enumerate(["coffee", "does", "not", "work"])]
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=6.,
        words=words, language="en")
    option = _option(text="coffee does work",
                     evidence=[{"basis": "speech",
                                "quote": "coffee does not work"}])
    accepted, rejected = op.validate_opening_option(option, packet=p)
    assert accepted is None, "same-token subset reversed the claim polarity"
    assert rejected["code"] == "text_unsupported"


# ---------------------------------------------------------------------------
# New: contraction paraphrase preserves polarity and passes
# ---------------------------------------------------------------------------

def test_contraction_paraphrase_preserves_polarity():
    words = [{"word": w, "start": 1 + i * 0.3, "end": 1 + (i + 1) * 0.3}
             for i, w in enumerate(["coffee", "does", "not", "work"])]
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=6.,
        words=words, language="en")
    option = _option(text="coffee doesn't work",
                     evidence=[{"basis": "speech",
                                "quote": "coffee does not work"}])
    accepted, rejected = op.validate_opening_option(option, packet=p)
    assert rejected is None, rejected
    assert accepted["grounded"] is True


def test_novel_negation_still_rejected():
    accepted, rejected = op.validate_opening_option(
        _option(text="coffee never works"), packet=_packet())
    assert accepted is None
    assert rejected["code"] == "text_unsupported"


# ---------------------------------------------------------------------------
# Promoted probe 5: novel sensory action is not automatically a paraphrase
# ---------------------------------------------------------------------------

def test_novel_sensory_action_is_not_automatically_paraphrase():
    event = {"kind": "visual", "label": "red ball rolls",
             "start": 1., "end": 2., "event_id": "v1"}
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=6.,
        words=[], av_evidence=[event], language="en")
    option = _option(text="red ball explodes", evidence_basis="visual",
                     evidence=[{"basis": "visual", "label": event["label"],
                                "start": 1., "end": 2., "event_id": "v1"}])
    accepted, rejected = op.validate_opening_option(option, packet=p)
    assert accepted is None, \
        "unlisted novel sensory verb treated as supported paraphrase"
    assert rejected["code"] == "text_unsupported"


# ---------------------------------------------------------------------------
# New: supported manner paraphrase still passes
# ---------------------------------------------------------------------------

def test_supported_manner_paraphrase_still_passes():
    event = {"kind": "visual", "label": "crowd cheers",
             "start": 11., "end": 13., "event_id": "v1"}
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=20.,
        words=[], av_evidence=[event], language="en")
    option = _option(text="the crowd erupts", evidence_basis="visual",
                     evidence=[{"basis": "visual", "label": "crowd cheers",
                                "start": 11., "end": 13., "event_id": "v1"}])
    accepted, rejected = op.validate_opening_option(option, packet=p)
    assert rejected is None, rejected
    assert accepted["grounded"] is True


# ---------------------------------------------------------------------------
# New: fail-then-restore mutations for the review-03 gates
# ---------------------------------------------------------------------------

def test_mutation_manner_verb_gate_is_load_bearing():
    event = {"kind": "visual", "label": "crowd cheers",
             "start": 11., "end": 13., "event_id": "v1"}
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=20.,
        words=[], av_evidence=[event], language="en")
    good = _option(text="the crowd erupts", evidence_basis="visual",
                   evidence=[{"basis": "visual", "label": "crowd cheers",
                              "start": 11., "end": 13., "event_id": "v1"}])
    accepted, _ = op.validate_opening_option(good, packet=p)
    assert accepted is not None
    # MUTATION: drop the manner-paraphrase mapping -> "erupts" becomes
    # an unverifiable novel action.
    real = _mutate("_SENSORY_MANNER_VERBS", {})
    try:
        accepted, _ = op.validate_opening_option(good, packet=p)
        assert accepted is None, "mutation did not break the gate"
    finally:
        _mutate("_SENSORY_MANNER_VERBS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(good, packet=p)
    assert accepted is not None


def test_mutation_polarity_gate_is_load_bearing():
    words = [{"word": w, "start": 1 + i * 0.3, "end": 1 + (i + 1) * 0.3}
             for i, w in enumerate(["coffee", "does", "not", "work"])]
    p = op.build_opening_packet(
        source_fingerprint="source", final_start=0., final_end=6.,
        words=words, language="en")
    bad = _option(text="coffee does work",
                  evidence=[{"basis": "speech",
                             "quote": "coffee does not work"}])
    accepted, _ = op.validate_opening_option(bad, packet=p)
    assert accepted is None
    # MUTATION: "not" stops being a negation marker -> the polarity
    # flip passes.
    real = _mutate("_NEGATION_TOKENS", op._NEGATION_TOKENS - {"not"})
    try:
        accepted, _ = op.validate_opening_option(bad, packet=p)
        assert accepted is not None, "mutation did not break the gate"
    finally:
        _mutate("_NEGATION_TOKENS", real)
    # RESTORE.
    accepted, _ = op.validate_opening_option(bad, packet=p)
    assert accepted is None


# ---------------------------------------------------------------------------
# Promoted probe 3: sub-millisecond remainder never starts an SDK call
# ---------------------------------------------------------------------------

def test_submillisecond_remainder_never_starts_sdk_call():
    calls, closes = [], []

    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            return type("Response", (), {"text": "[]"})()

    class Client:
        models = Models()

        def close(self):
            closes.append(1)

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="injected")
    with pytest.raises(op.OpeningBudgetExhausted):
        provider.request_options({"timeout_s": 0.0004, "prompt": "offline"})
    assert calls == [], "SDK must not be called with timeout=0"
    assert closes == [1], "client still closes exactly once"


# ---------------------------------------------------------------------------
# New: remaining-budget callback failure fails closed without a stale timeout
# ---------------------------------------------------------------------------

def test_remaining_callback_failure_fails_closed():
    calls, closes = [], []

    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            return type("Response", (), {"text": "[]"})()

    class Client:
        models = Models()

        def close(self):
            closes.append(1)

    def broken_remaining():
        raise RuntimeError("injected clock failure")

    provider = op.GeminiOpeningProvider(client_factory=Client,
                                       model="injected")
    with pytest.raises(op.OpeningBudgetExhausted):
        provider.request_options({"timeout_s": 5.0, "prompt": "test",
                                  "remaining_s": broken_remaining})
    assert calls == [], "no SDK call on unverifiable budget"
    assert closes == [1]


# ---------------------------------------------------------------------------
# New: representable timeouts still reach the SDK, floored
# ---------------------------------------------------------------------------

def test_representable_small_timeout_calls_sdk_floored():
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
    provider.request_options({"timeout_s": 0.0027, "prompt": "test",
                              "retries_disabled": True})
    assert len(captured) == 1
    assert captured[0]["config"]["http_options"]["timeout"] == 2, \
        "floored to 2ms, never rounded up"


def test_fractional_timeout_never_rounded_up():
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
    assert captured[0]["config"]["http_options"]["timeout"] == 1999
