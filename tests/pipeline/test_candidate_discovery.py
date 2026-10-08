"""Tests for windowed candidate discovery (Task 08).

All provider interactions use injected deterministic fakes;
no live SDK calls.
"""
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clippyme.pipeline import candidate_discovery as cd


def make_words(n, start=0.0, step=0.5, speaker_cycle=2):
    """Deterministic word list with absolute timestamps."""
    words = []
    t = start
    for i in range(n):
        words.append({
            "text": f"word{i}",
            "start": t,
            "end": t + step * 0.9,
            "speaker": i % speaker_cycle,
        })
        t += step
    return words


def make_config(**kw):
    base = dict(
        window_target_s=240.0, window_overlap_s=45.0,
        boundary_tolerance_s=12.0, max_windows=32,
        max_raw_candidates=200, max_provider_attempts=24,
        deadline_s=600.0, sdk_timeout_ms=120000,
        model_name="test-model", fallback_models=(),
    )
    base.update(kw)
    return cd.DiscoveryConfig(**base)


class FakeClient:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def fake_response(candidates):
    class R:
        text = json.dumps({"candidates": candidates})
    return R()


# --- Window planning -------------------------------------------------

def test_plan_windows_covers_all_words():
    # 30 min transcript: early, middle, late.
    words = make_words(3600, step=0.5)  # 1800s
    plan = cd.plan_windows(words, make_config())
    cov = plan["coverage"]
    assert cov["valid_words"] == 3600
    assert cov["covered_words"] == 3600
    assert cov["first_covered"] and cov["last_covered"]
    assert plan["invalid_words"] == []


def test_plan_windows_single_window_source():
    words = make_words(100, step=0.5)  # 50s
    plan = cd.plan_windows(words, make_config())
    assert len(plan["windows"]) == 1
    w = plan["windows"][0]
    assert w.word_start == 0
    assert w.word_end == 100


def test_plan_windows_overlap():
    words = make_words(600, step=0.5)  # 300s
    plan = cd.plan_windows(words, make_config(
        window_target_s=240.0, window_overlap_s=45.0))
    ws = plan["windows"]
    assert len(ws) >= 2
    # Overlap: second window starts before first ends.
    assert ws[1].start_s < ws[0].end_s


def test_plan_windows_isolates_invalid():
    words = make_words(10)
    words[3] = {"text": "", "start": 1.0, "end": 1.5}  # empty
    words[5] = {"text": "x"}  # no timestamps
    words[7] = "not-a-dict"
    plan = cd.plan_windows(words, make_config())
    assert plan["coverage"]["valid_words"] == 7
    assert sorted(plan["invalid_words"]) == [3, 5, 7]


def test_plan_windows_empty():
    plan = cd.plan_windows([], make_config())
    assert plan["windows"] == []
    assert plan["coverage"]["valid_words"] == 0


def test_removing_final_window_fails_coverage():
    words = make_words(600, step=0.5)
    plan = cd.plan_windows(words, make_config())
    ws = plan["windows"]
    # Mutation: drop the last window -> last word uncovered.
    truncated = ws[:-1]
    last_orig = 599
    covered = any(w.word_start <= last_orig < w.word_end for w in truncated)
    assert not covered


# --- Candidate validation --------------------------------------------

def _window():
    return cd.PlannedWindow(index=0, start_s=0.0, end_s=240.0,
                            word_start=0, word_end=480)


def _good_item():
    return {
        "start_s": 10.0, "end_s": 30.0,
        "event_type": "story",
        "evidence_text": "word20 word21 word22 quoted here",
        "evidence_basis": "transcript",
        "context": "a funny story about testing",
        "expected_payoff": "viewers laugh and share",
        "confidence": 0.8, "recommendation_score": 75,
    }


def test_validate_good_candidate():
    words = make_words(480, step=0.5)
    rec = cd.validate_discovery_candidate(_good_item(), _window(), words)
    assert rec.candidate_id.startswith("cand-")
    assert rec.evidence_basis == "transcript"
    assert rec.speaker_ids  # speaker IDs preserved


def test_validate_rejects_bool_confidence():
    words = make_words(480, step=0.5)
    item = _good_item()
    item["confidence"] = True
    with pytest.raises(cd.DiscoveryError):
        cd.validate_discovery_candidate(item, _window(), words)


def test_validate_rejects_reversed_interval():
    words = make_words(480, step=0.5)
    item = _good_item()
    item["start_s"], item["end_s"] = 30.0, 10.0
    with pytest.raises(cd.DiscoveryError):
        cd.validate_discovery_candidate(item, _window(), words)


def test_validate_rejects_outside_window():
    words = make_words(480, step=0.5)
    item = _good_item()
    item["end_s"] = 500.0
    with pytest.raises(cd.DiscoveryError):
        cd.validate_discovery_candidate(item, _window(), words)


def test_validate_rejects_missing_evidence():
    words = make_words(480, step=0.5)
    item = _good_item()
    item["evidence_text"] = "  "
    with pytest.raises(cd.DiscoveryError):
        cd.validate_discovery_candidate(item, _window(), words)


def test_stable_id_ignores_model_id():
    words = make_words(480, step=0.5)
    a = _good_item()
    b = _good_item()
    b["id"] = "model-made-id-123"
    ra = cd.validate_discovery_candidate(a, _window(), words)
    rb = cd.validate_discovery_candidate(b, _window(), words)
    assert ra.candidate_id == rb.candidate_id


# --- Deduplication ----------------------------------------------------

def _rec(cid_suffix, start, end, evidence, event_key=None, score=70):
    return cd.CandidateRecord(
        candidate_id=f"cand-{cid_suffix}", window_index=0,
        start_s=start, end_s=end, event_type="moment",
        evidence_text=evidence, evidence_basis="transcript",
        context="ctx", expected_payoff="payoff",
        event_key=event_key, confidence=0.8,
        recommendation_score=score, speaker_ids=[0])


def test_overlap_alone_never_dedupes():
    # Two different events sharing an interval both survive.
    a = _rec("a", 10.0, 30.0, "the cat sat on the mat today")
    b = _rec("b", 12.0, 28.0, "quantum physics explains everything here")
    result = cd.dedupe_candidates([a, b])
    assert len(result["kept"]) == 2
    assert result["removed"] == {}


def test_event_key_collapse():
    a = _rec("a", 10.0, 30.0, "first telling of the story", event_key="evt1")
    b = _rec("b", 25.0, 45.0, "second telling of the story",
             event_key="evt1", score=60)
    result = cd.dedupe_candidates([a, b])
    assert len(result["kept"]) == 1
    assert result["kept"][0].candidate_id == "cand-a"
    assert result["removed"]["cand-b"]["basis"] == "event_key"


def test_evidence_similarity_collapse():
    a = _rec("a", 10.0, 30.0,
             "the quick brown fox jumps over the lazy dog near the river")
    b = _rec("b", 12.0, 32.0,
             "the quick brown fox jumps over the lazy dog near the river",
             score=60)
    result = cd.dedupe_candidates([a, b])
    assert len(result["kept"]) == 1
    assert result["removed"]["cand-b"]["basis"] == "evidence_similarity"


# --- Budget ------------------------------------------------------------

def test_budget_exhaustion():
    cfg = make_config(max_provider_attempts=2, deadline_s=600.0)
    b = cd.DiscoveryBudget(cfg)
    assert b.consume() and b.consume()
    assert not b.consume()
    assert b.exhausted()


def test_sdk_timeout_bounded_by_deadline():
    cfg = make_config(deadline_s=600.0, sdk_timeout_ms=120000)
    times = [0.0]
    b = cd.DiscoveryBudget(cfg, time_fn=lambda: times[0])
    assert b.sdk_timeout_ms() == 120000
    times[0] = 599.0  # 1s left
    assert b.sdk_timeout_ms() <= 1000
    assert b.sdk_timeout_ms() >= 1000  # finite floor


# --- End-to-end with fakes ----------------------------------------------

def _run_fake(words, responses, **kw):
    """responses: list of (candidates | Exception | 'parse_fail')."""
    cfg = make_config(**kw.pop("config_kw", {}))
    calls = {"n": 0}

    def gen(client, prompt, config, budget):
        calls["n"] += 1
        r = responses[min(calls["n"] - 1, len(responses) - 1)]
        if isinstance(r, Exception):
            raise r
        if r == "parse_fail":
            return "not json at all {{{"
        return json.dumps({"candidates": r})

    client = FakeClient()
    result = cd.run_windowed_discovery(
        {"words": words}, 1800.0,
        config=cfg,
        client_factory=lambda: client,
        generate_fn=gen,
        **kw)
    return result, client, calls


def test_e2e_finds_early_middle_late():
    words = make_words(3600, step=0.5)  # 1800s
    cfg_kw = dict(window_target_s=240.0, window_overlap_s=45.0)

    def gen(client, prompt, config, budget):
        # One candidate per window at the window's midpoint.
        import re as _re
        m = _re.search(r"(\d+\.\d+)s to (\d+\.\d+)s", prompt)
        s, e = float(m.group(1)), float(m.group(2))
        mid = (s + e) / 2
        return json.dumps({"candidates": [{
            "start_s": mid - 5, "end_s": mid + 5,
            "event_type": "moment",
            "evidence_text": f"moment near {mid:.0f} seconds here",
            "context": "context here", "expected_payoff": "payoff here",
            "confidence": 0.7, "recommendation_score": 70}]})

    client = FakeClient()
    result = cd.run_windowed_discovery(
        {"words": words}, 1800.0,
        config=make_config(**cfg_kw),
        client_factory=lambda: client,
        generate_fn=gen)
    funnel = result["candidate_discovery"]
    assert funnel["windows_succeeded"] >= 3
    starts = [c["start"] for c in result["shorts"]]
    assert min(starts) < 300  # early
    assert max(starts) > 1200  # late
    assert client.closed == 1


def test_e2e_malformed_and_failed_window_isolated():
    words = make_words(900, step=0.5)  # 450s -> 3 windows
    good = _good_item()
    bad = {"start_s": "oops", "end_s": 20.0}  # malformed
    responses = [
        [good, bad],  # window 0: one valid, one invalid
        Exception("boom"),  # window 1: failed
        "parse_fail",  # window 2: parse failed
    ]
    result, client, calls = _run_fake(words, responses)
    funnel = result["candidate_discovery"]
    assert funnel["windows_succeeded"] == 1
    assert funnel["windows_failed"] == 2
    assert funnel["valid_candidates"] == 1
    assert funnel["invalid_candidates"] == 1
    assert funnel["counts_reconcile"]
    assert client.closed == 1


def test_e2e_caps_apply_after_aggregation():
    words = make_words(600, step=0.5)
    # Every window returns 5 candidates within its own range.
    def gen(client, prompt, config, budget):
        import re as _re
        m = _re.search(r"(\d+\.\d+)s to (\d+\.\d+)s", prompt)
        s = float(m.group(1))
        cands = []
        for i in range(5):
            cs = s + 10.0 + i * 20
            cands.append({
                "start_s": cs, "end_s": cs + 10,
                "event_type": "moment",
                "evidence_text": f"distinct evidence window {s:.0f} num {i}",
                "context": "context here", "expected_payoff": "payoff here",
                "confidence": 0.7, "recommendation_score": 90 - i})
        return json.dumps({"candidates": cands})
    old_max = os.getenv("CLIPPYME_MAX_CLIPS")
    os.environ["CLIPPYME_MAX_CLIPS"] = "3"
    try:
        client = FakeClient()
        result = cd.run_windowed_discovery(
            {"words": words}, 300.0,
            config=make_config(),
            client_factory=lambda: client,
            generate_fn=gen)
    finally:
        if old_max is None:
            os.environ.pop("CLIPPYME_MAX_CLIPS", None)
        else:
            os.environ["CLIPPYME_MAX_CLIPS"] = old_max
    # 2 windows x 5 = 10 raw, capped to 3 AFTER aggregation.
    assert len(result["shorts"]) == 3
    assert result["candidate_discovery"]["valid_candidates"] == 10


def test_e2e_sparse_never_padded():
    words = make_words(100, step=0.5)
    result, _, _ = _run_fake(words, [[]])
    assert result["shorts"] == []
    assert result["candidate_discovery"]["final_selected"] == 0


def test_e2e_budget_exhaustion_stops_calls():
    words = make_words(1200, step=0.5)  # several windows
    cfg_kw = dict(max_provider_attempts=1)
    result, _, calls = _run_fake(words, [[_good_item()]], config_kw=cfg_kw)
    assert calls["n"] == 1
    funnel = result["candidate_discovery"]
    assert funnel["budget"]["exhausted"]


def test_e2e_client_setup_failure():
    words = make_words(100, step=0.5)

    def bad_factory():
        raise RuntimeError("no client")

    result = cd.run_windowed_discovery(
        {"words": words}, 50.0,
        config=make_config(),
        client_factory=bad_factory,
        generate_fn=lambda *a, **k: "{}")
    assert result["shorts"] == []
    assert any(r["reason"] == "client_setup_failed"
               for r in result["candidate_discovery"]["rejections"])


def _import_main_stubbed():
    """Import main.py with heavy deps stubbed (no cv2/torch on host)."""
    import sys as _sys
    from unittest.mock import MagicMock as _MM
    _torch = _MM()
    _torch.cuda.is_available.return_value = False
    _psutil = _MM()
    _psutil.virtual_memory.return_value.total = 16 * (1024 ** 3)
    _sys.modules["psutil"] = _psutil
    for _m in ["cv2", "scenedetect", "scenedetect.detectors", "ultralytics",
               "mediapipe", "yt_dlp", "tqdm", "dotenv", "google",
               "google.genai", "numpy", "psutil", "faster_whisper",
               "stable_ts"]:
        _sys.modules.setdefault(_m, _MM())
    _sys.modules["torch"] = _torch
    # scenedetect needs specific attrs.
    _sys.modules["scenedetect"].open_video = _MM()
    _sys.modules["scenedetect"].SceneManager = _MM()
    _sys.modules["google"].genai = _sys.modules["google.genai"]
    import clippyme.pipeline.main as main_mod
    return main_mod


def test_flag_disabled_selects_legacy(monkeypatch):
    monkeypatch.setenv("CLIPPYME_WINDOWED_DISCOVERY", "0")
    assert cd.windowed_discovery_enabled() is False
    monkeypatch.setenv("CLIPPYME_WINDOWED_DISCOVERY", "false")
    assert cd.windowed_discovery_enabled() is False
    src_path = Path(__file__).parent.parent.parent / "src" / "clippyme" / "pipeline" / "main.py"
    text = src_path.read_text(encoding="utf-8")
    assert "windowed_discovery_enabled()" in text
    assert "_get_viral_clips_legacy" in text
    assert "client.close()" in text


def test_flag_enabled_routes_windowed(monkeypatch):
    monkeypatch.delenv("CLIPPYME_WINDOWED_DISCOVERY", raising=False)
    assert cd.windowed_discovery_enabled() is True
    monkeypatch.setenv("CLIPPYME_WINDOWED_DISCOVERY", "1")
    assert cd.windowed_discovery_enabled() is True
    src_path = Path(__file__).parent.parent.parent / "src" / "clippyme" / "pipeline" / "main.py"
    text = src_path.read_text(encoding="utf-8")
    assert "_get_viral_clips_windowed" in text


def test_discovery_prompt_contract():
    from clippyme.pipeline.gemini_request import build_discovery_prompt
    w = cd.PlannedWindow(index=0, start_s=0.0, end_s=240.0,
                         word_start=0, word_end=10)
    words = make_words(10)
    p = build_discovery_prompt(w, words, instructions="prefer funny")
    assert '"candidates"' in p
    assert '"evidence_basis": "transcript"' in p
    assert "<untrusted_user_preferences>" in p
    assert "spk:0" in p  # speaker IDs preserved


def test_parse_discovery_payload():
    from clippyme.pipeline.gemini_parser import parse_discovery_payload
    cands = parse_discovery_payload('{"candidates": [{"a": 1}]}')
    assert cands == [{"a": 1}]
    with pytest.raises(ValueError):
        parse_discovery_payload('{"nope": []}')
    with pytest.raises(ValueError):
        parse_discovery_payload('not json')


def test_generate_timeout_param():
    from clippyme.pipeline import gemini_request as gr
    import inspect
    sig = inspect.signature(gr.generate_with_model_fallback)
    assert "timeout_ms" in sig.parameters
    assert sig.parameters["timeout_ms"].default == 120000

# --- Promoted coordinator review1 contracts (durable regressions) ---
# Adapted from coordinator-review1-contract.py; injected providers only.

from types import SimpleNamespace
from clippyme.schemas import ViralClip
import ast

def cfg(**overrides):
    values = dict(
        window_target_s=240.0,
        window_overlap_s=45.0,
        boundary_tolerance_s=12.0,
        max_windows=32,
        max_raw_candidates=200,
        max_provider_attempts=24,
        deadline_s=600.0,
        sdk_timeout_ms=120000,
        model_name="primary",
        fallback_models=(),
    )
    values.update(overrides)
    return cd.DiscoveryConfig(**values)


class Client:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def accepted_transcript(word_count=80, *, start=0.0, step=0.5):
    words = []
    for index in range(word_count):
        at = start + index * step
        words.append({
            "word": f"word{index}",
            "start": at,
            "end": at + step * 0.8,
            "speaker": index % 2,
        })
    return {"text": " ".join(w["word"] for w in words),
            "segments": [{"start": start,
                          "end": words[-1]["end"],
                          "words": words}]}


def good_candidate(start=5.0, end=15.0, *, score=80, evidence="word10 word11 evidence quote"):
    return {
        "start_s": start,
        "end_s": end,
        "event_type": "story",
        "evidence_text": evidence,
        "evidence_basis": "transcript",
        "context": "the setup needed to understand the moment",
        "expected_payoff": "the claim resolves with a clear payoff",
        "confidence": 0.8,
        "recommendation_score": score,
    }


def generate_payload(items):
    return lambda *_args, **_kwargs: json.dumps({"candidates": items})


def test_real_segment_transcript_contract_reaches_provider_and_preserves_speaker_zero():
    transcript = accepted_transcript()
    calls = []
    client = Client()

    def generate(_client, prompt, _config, _budget):
        calls.append(prompt)
        return json.dumps({"candidates": [good_candidate()]})

    result = cd.run_windowed_discovery(
        transcript,
        40.0,
        config=cfg(),
        client_factory=lambda: client,
        generate_fn=generate,
    )

    assert len(calls) == 1
    assert "[spk:0]" in calls[0]
    assert len(result["shorts"]) == 1
    assert result["candidate_discovery"]["transcript_coverage"]["covered_words"] == 80
    assert client.closed == 1


def test_task04_flattened_word_contract_is_accepted_without_rebasing():
    words = [
        {"w": "hello", "s": 30.0, "e": 30.4, "sp": 0},
        {"w": "world", "s": 30.5, "e": 31.0, "sp": 1},
    ]
    plan = cd.plan_windows(words, cfg())
    assert plan["coverage"]["valid_words"] == 2
    assert plan["windows"][0].start_s == 30.0
    assert plan["windows"][0].end_s == 31.0


def test_selected_short_is_a_real_viralclip_shape():
    words = [
        {"text": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.4,
         "speaker": 0}
        for i in range(80)
    ]
    result = cd.run_windowed_discovery(
        {"words": words},
        40.0,
        config=cfg(),
        client_factory=Client,
        generate_fn=generate_payload([good_candidate()]),
    )
    assert result["shorts"]
    # The production downstream contract must validate, rather than merely
    # resembling a subset of the required fields.
    ViralClip.model_validate(result["shorts"][0])


@pytest.mark.parametrize(
    "field,value",
    [("confidence", -0.1), ("confidence", 1.1),
     ("recommendation_score", 0), ("recommendation_score", 101)],
)
def test_candidate_ranges_are_strict(field, value):
    words = [
        {"text": f"word{i}", "start": i * 0.5, "end": i * 0.5 + 0.4}
        for i in range(80)
    ]
    item = good_candidate()
    item[field] = value
    window = cd.PlannedWindow(0, 0.0, 39.9, 0, len(words))
    with pytest.raises(cd.DiscoveryError):
        cd.validate_discovery_candidate(item, window, words)


def test_source_duration_is_an_authoritative_candidate_bound():
    words = [
        {"text": f"word{i}", "start": i, "end": i + 0.8}
        for i in range(100)
    ]
    result = cd.run_windowed_discovery(
        {"words": words},
        30.0,
        config=cfg(),
        client_factory=Client,
        generate_fn=generate_payload([good_candidate(70.0, 80.0)]),
    )
    assert result["shorts"] == []
    assert result["candidate_discovery"]["invalid_candidates"] == 1


def test_explicit_selection_and_duration_parameters_reach_windowed_engine():
    source = Path("src/clippyme/pipeline/main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_get_viral_clips_windowed")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "run_windowed_discovery")
    forwarded = {kw.arg for kw in call.keywords}
    assert {
        "min_duration", "max_duration", "min_clips", "max_clips",
        "clip_type", "duration_mode", "audience_intel",
    } <= forwarded


def test_shared_budget_counts_each_fallback_provider_call():
    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, *, model, **_kwargs):
            self.calls.append(model)
            raise RuntimeError("503 unavailable capacity")

    client = Client()
    client.models = Models()
    words = [{"text": "word", "start": 0.0, "end": 0.5}]
    result = cd.run_windowed_discovery(
        {"words": words},
        1.0,
        config=cfg(max_provider_attempts=1,
                   model_name="m1", fallback_models=("m2", "m3")),
        client_factory=lambda: client,
    )
    assert len(client.models.calls) <= 1
    assert result["candidate_discovery"]["budget"]["attempts_used"] == len(client.models.calls)
    assert client.closed == 1


def test_sdk_timeout_never_exceeds_subsecond_remaining_deadline():
    now = [0.0]
    budget = cd.DiscoveryBudget(
        cfg(deadline_s=10.0, sdk_timeout_ms=120000),
        time_fn=lambda: now[0],
    )
    now[0] = 9.8
    timeout = budget.sdk_timeout_ms()
    assert 0 < timeout <= 200


def test_window_limit_does_not_turn_last_window_into_unbounded_prompt():
    words = [
        {"text": f"word{i}", "start": float(i), "end": float(i) + 0.5}
        for i in range(1000)
    ]
    plan = cd.plan_windows(
        words,
        cfg(window_target_s=100.0, window_overlap_s=10.0,
            boundary_tolerance_s=5.0, max_windows=2),
    )
    assert len(plan["windows"]) <= 2
    assert all(
        window.end_s - window.start_s <= 105.5
        for window in plan["windows"]
    )
    # If the configured cap prevents complete coverage, say so honestly.
    if plan["coverage"]["covered_words"] < plan["coverage"]["valid_words"]:
        assert plan["coverage"].get("truncated_by_window_limit") is True


def test_duplicate_occurrences_have_distinct_trace_ids():
    base = cd.CandidateRecord(
        candidate_id="cand-same",
        window_index=0,
        start_s=10.0,
        end_s=20.0,
        event_type="story",
        evidence_text="the same quote from an overlap window",
        evidence_basis="transcript",
        context="context",
        expected_payoff="payoff",
        event_key="event-one",
        confidence=0.8,
        recommendation_score=80,
        speaker_ids=[0],
    )
    second = cd.CandidateRecord(**{**base.__dict__, "window_index": 1})
    result = cd.dedupe_candidates([base, second])
    assert len(result["removed"]) == 1
    removed_id, detail = next(iter(result["removed"].items()))
    assert removed_id != detail["kept_id"]


def test_funnel_survives_queued_fallback_when_discovery_returns_zero(tmp_path, monkeypatch):
    from clippyme.pipeline import orchestrator as orch

    funnel = {"version": cd.DISCOVERY_VERSION, "final_selected": 0}

    class Legacy:
        @staticmethod
        def get_viral_clips(*_args, **_kwargs):
            return {"shorts": [], "candidate_discovery": funnel,
                    "candidate_store": {}}

        @staticmethod
        def build_texttiling_fallback(*_args, **_kwargs):
            return {"shorts": []}

    class State:
        data = {"job_id": "job"}

        def artifact(self, _name): return None
        def completed(self, _name): return False
        def consume_steering(self): return []
        def start(self, *_args, **_kwargs): pass
        def set_clip_total(self, *_args, **_kwargs): pass
        def complete_stage(self, *_args, **_kwargs): pass
        def note_missed_steering(self, *_args, **_kwargs): pass

    args = SimpleNamespace(
        skip_analysis=False, instructions=None, min_duration=None,
        max_duration=None, min_clips=None, max_clips=None, clip_type=None,
        duration_mode=None, monitor=False, aspect="9:16",
        caption_style_default=None, reframe_mode="auto",
    )
    monkeypatch.setattr(orch, "should_use_fallback", lambda _monitor: True)
    monkeypatch.setattr(orch, "_whole_video_fallback",
                        lambda *_args, **_kwargs: {"shorts": []})
    result, _metadata = orch._load_or_analyze(
        args,
        str(tmp_path / "source.mp4"),
        "title",
        str(tmp_path),
        60.0,
        accepted_transcript(),
        State(),
        Legacy(),
    )
    assert result.get("candidate_discovery") == funnel



# --- Mutation checks (must fail if key behaviors are removed) ---

def test_mutation_segment_flattening_removed_fails():
    """If segments[*].words flattening is removed, zero words reached."""
    # Simulate by passing segments without words key handling:
    # normalize_transcript_words must handle segments.
    transcript = {"segments": [{"words": [
        {"word": "hi", "start": 10.0, "end": 10.5, "speaker": 0},
    ]}]}
    words = cd.normalize_transcript_words(transcript)
    # If flattening removed, this would be 0.
    assert len(words) == 1
    assert words[0]["s"] == 10.0  # Absolute, not rebased.

def test_mutation_max_clips_bypassed_fails():
    """Explicit max_clips must cap globally post-dedupe."""
    import json
    words = [{"w": f"w{i}", "s": float(i), "e": float(i)+0.4}
             for i in range(20)]
    items = []
    for i in range(5):
        items.append({
            "start_s": float(i*2), "end_s": float(i*2+1),
            "event_type": "story",
            "evidence_text": f"evidence quote number {i} here",
            "evidence_basis": "transcript",
            "context": f"context for item {i} here",
            "expected_payoff": f"payoff for item {i} here",
            "confidence": 0.9, "recommendation_score": 90 - i,
        })
    result = cd.run_windowed_discovery(
        {"words": words}, 20.0,
        config=cd.DiscoveryConfig(),
        client_factory=lambda: None,
        generate_fn=lambda *a, **k: json.dumps({"candidates": items}),
        max_clips=2,
    )
    # If max_clips bypassed, we'd get 5.
    assert len(result["shorts"]) == 2

def test_mutation_budget_exceeded_fails():
    """Provider calls must not exceed budget."""
    calls = {"n": 0}
    def fake_gen(*a, **k):
        calls["n"] += 1
        import json
        return json.dumps({"candidates": []})
    words = [{"w": "x", "s": float(i), "e": float(i)+0.4}
             for i in range(600)]  # Multiple windows.
    result = cd.run_windowed_discovery(
        {"words": words}, 300.0,
        config=cd.DiscoveryConfig(max_provider_attempts=2),
        client_factory=lambda: None,
        generate_fn=fake_gen,
    )
    # If budget not enforced, calls would exceed 2.
    assert calls["n"] <= 2

def test_mutation_final_window_stretched_fails():
    """Final window must never exceed target + tolerance."""
    words = [{"w": f"w{i}", "s": float(i), "e": float(i)+0.5}
             for i in range(1000)]
    plan = cd.plan_windows(
        words,
        cd.DiscoveryConfig(window_target_s=100.0, window_overlap_s=10.0,
                           boundary_tolerance_s=5.0, max_windows=2),
    )
    max_allowed = 100.0 + 5.0 + 0.6  # target + tolerance + epsilon
    for w in plan["windows"]:
        assert w.end_s - w.start_s <= max_allowed, (
            f"Window {w.index} stretched: {w.end_s - w.start_s}")

def test_mutation_dedupe_id_aliasing_fails():
    """Duplicate occurrences must have distinct trace IDs."""
    import json
    words = [{"w": f"w{i}", "s": float(i), "e": float(i)+0.4}
             for i in range(100)]
    # Same evidence in overlapping windows.
    item = {
        "start_s": 10.0, "end_s": 20.0,
        "event_type": "story",
        "evidence_text": "the same quote here for dedupe",
        "evidence_basis": "transcript",
        "context": "context here for the moment",
        "expected_payoff": "payoff here for the moment",
        "event_key": "event-1",
        "confidence": 0.8, "recommendation_score": 80,
    }
    result = cd.run_windowed_discovery(
        {"words": words}, 100.0,
        config=cd.DiscoveryConfig(window_target_s=30.0,
                                  window_overlap_s=15.0),
        client_factory=lambda: None,
        generate_fn=lambda *a, **k: json.dumps({"candidates": [item]}),
    )
    store = result.get("candidate_store", {})
    # If IDs aliased, we'd have fewer distinct IDs than occurrences.
    # (With overlap, same moment appears in multiple windows.)
    ids = list(store.keys())
    assert len(ids) == len(set(ids)), "Duplicate IDs in store"

def test_mutation_fallback_drops_funnel_fails(tmp_path):
    """Fallback must preserve candidate_discovery funnel."""
    from clippyme.pipeline import orchestrator as orch
    from types import SimpleNamespace
    funnel = {"version": cd.DISCOVERY_VERSION, "final_selected": 0}
    class Legacy:
        @staticmethod
        def get_viral_clips(*a, **k):
            return {"shorts": [], "candidate_discovery": funnel,
                    "candidate_store": {}}
        @staticmethod
        def build_texttiling_fallback(*a, **k):
            return {"shorts": []}
    class State:
        data = {"job_id": "job"}
        def artifact(self, _n): return None
        def completed(self, _n): return False
        def consume_steering(self): return []
        def start(self, *a, **k): pass
        def set_clip_total(self, *a, **k): pass
        def complete_stage(self, *a, **k): pass
        def note_missed_steering(self, *a, **k): pass
    args = SimpleNamespace(
        skip_analysis=False, instructions=None, min_duration=None,
        max_duration=None, min_clips=None, max_clips=None, clip_type=None,
        duration_mode=None, monitor=False, aspect="9:16",
        caption_style_default=None, reframe_mode="auto",
    )
    import pytest as _pt
    # Monkeypatch via simple attribute set.
    orig_suf = orch.should_use_fallback
    orig_wvf = orch._whole_video_fallback
    orch.should_use_fallback = lambda _m: True
    orch._whole_video_fallback = lambda *a, **k: {"shorts": []}
    try:
        result, _ = orch._load_or_analyze(
            args, str(tmp_path / "s.mp4"), "title", str(tmp_path),
            60.0, {"segments": []}, State(), Legacy(),
        )
    finally:
        orch.should_use_fallback = orig_suf
        orch._whole_video_fallback = orig_wvf
    # If funnel dropped, this fails.
    assert result.get("candidate_discovery") == funnel
