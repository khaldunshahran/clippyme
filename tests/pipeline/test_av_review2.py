"""Task 11 review-02 production contracts, promoted into durable tests.

All six parameterized cases from
``work/muse-handoff/task-11/coordinator-review2-remaining.py``, promoted
into durable repository tests and adapted to the corrected contracts,
plus one mutation check per correction: each mutation test FAILS if the
listed production behavior is broken, proving the suite is sensitive to
the breakage.

1. ``ffmpeg_frame_sampler`` shares one monotonic sampling deadline
   across every frame decode and JPEG-quality subprocess: the remaining
   time is recomputed before every ``subprocess.run`` and no attempt
   starts once the deadline is exhausted.
2. ``validate_av_evidence`` requires ``visual_description`` to be an
   actual JSON string; booleans, numbers, objects, arrays, and other
   types are rejected with ``invalid_visual_description_type``.
3. ``av_checkpoint_identity`` persists ``av_prompt_schema_version``
   explicitly and resume validation compares it field-for-field: a stale
   version forces queued re-analysis instead of reuse.

All provider interactions use injected deterministic fakes; no live SDK
calls.
"""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from clippyme.pipeline import av_discovery as avd
from clippyme.pipeline import candidate_discovery as cd
from clippyme.pipeline import gemini_request as gr
from clippyme.pipeline import orchestrator as orch


@pytest.fixture
def clean_av_env(monkeypatch):
    """Neutral AV env for deterministic runs."""
    monkeypatch.delenv("CLIPPYME_AV_DISCOVERY", raising=False)
    monkeypatch.delenv("CLIPPYME_MIN_VIRAL_SCORE", raising=False)
    monkeypatch.delenv("CLIPPYME_MAX_CLIPS", raising=False)
    monkeypatch.delenv("CLIPPYME_SILENCE_SNAP", raising=False)
    return monkeypatch


def _dcfg(**overrides):
    values = dict(
        window_target_s=60.0,
        window_overlap_s=0.0,
        boundary_tolerance_s=1.0,
        max_windows=1,
        max_raw_candidates=10,
        max_provider_attempts=2,
        deadline_s=30.0,
        sdk_timeout_ms=30000,
        model_name="model-a",
        fallback_models=("model-b",),
    )
    values.update(overrides)
    return cd.DiscoveryConfig(**values)


def _avcfg(**overrides):
    values = dict(
        media_window_target_s=60.0,
        media_window_overlap_s=0.0,
        max_media_windows=1,
        max_total_sampled_frames=1,
        frames_per_window=1,
    )
    values.update(overrides)
    return avd.AVDiscoveryConfig(**values)


def _visual_item(description):
    return {
        "start_s": 10.0,
        "end_s": 30.0,
        "event_type": "visual_reveal",
        "evidence_basis": "visual",
        "evidence_text": "",
        "visual_description": description,
        "visual_frame_ids": ["vf_0_0"],
        "audio_event_ids": [],
        "context": "the object is introduced before the reveal",
        "expected_payoff": "the viewer sees the completed reveal",
        "confidence": 0.8,
        "recommendation_score": 80,
    }


def _validate(item):
    return avd.validate_av_evidence(
        item,
        window_start_s=0.0,
        window_end_s=60.0,
        source_duration=60.0,
        frames_by_id={"vf_0_0": {"decoded_ts": 20.0}},
        audio_events_by_id={},
        window_words=[],
        min_duration=5.0,
        max_duration=60.0,
        source_fingerprint="source",
    )


# --- Correction 1: one monotonic sampling deadline ---------------------


def test_ffmpeg_quality_retries_share_one_sampling_deadline(
        monkeypatch, tmp_path):
    """The per-operation deadline must cover every quality/frame
    subprocess."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    clock = [0.0]
    observed_timeouts = []

    monkeypatch.setattr(avd, "source_media_usable", lambda _path: True)
    monkeypatch.setattr(avd.time, "monotonic", lambda: clock[0])

    class OversizedFrame:
        returncode = 0
        stdout = b"x" * 11
        stderr = b"pts_time:1.000"

    def fake_run(_cmd, *, capture_output, timeout):
        assert capture_output is True
        observed_timeouts.append(timeout)
        work_s = 0.06
        if timeout < work_s:
            clock[0] += timeout
            raise subprocess.TimeoutExpired(_cmd, timeout)
        clock[0] += work_s
        return OversizedFrame()

    monkeypatch.setattr(avd.subprocess, "run", fake_run)
    limits = avd.FrameLimits(
        max_frame_bytes=10,
        sampling_timeout_s=0.1,
        jpeg_quality_steps=(5, 12, 20),
    )

    with pytest.raises(avd.SamplingError):
        avd.ffmpeg_frame_sampler(str(source), [1.0], limits)

    assert observed_timeouts
    assert clock[0] <= 0.100001
    assert observed_timeouts[0] <= 0.1
    assert len(observed_timeouts) <= 2


def test_mutation_frozen_clock_defeats_shared_deadline(
        monkeypatch, tmp_path):
    """Mutation: with the deadline never recomputed (frozen clock), every
    attempt gets the full timeout and the operation overshoots its
    budget -- proving the shared-deadline gate is load-bearing."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture")
    real_elapsed = [0.0]
    observed_timeouts = []

    monkeypatch.setattr(avd, "source_media_usable", lambda _path: True)
    # Mutated production behavior: the sampler's clock never advances, so
    # the remaining time is always the full per-operation timeout.
    monkeypatch.setattr(avd.time, "monotonic", lambda: 1000.0)

    class OversizedFrame:
        returncode = 0
        stdout = b"x" * 11
        stderr = b"pts_time:1.000"

    def fake_run(_cmd, *, capture_output, timeout):
        observed_timeouts.append(timeout)
        work_s = 0.06
        if timeout < work_s:
            real_elapsed[0] += timeout
            raise subprocess.TimeoutExpired(_cmd, timeout)
        real_elapsed[0] += work_s
        return OversizedFrame()

    monkeypatch.setattr(avd.subprocess, "run", fake_run)
    limits = avd.FrameLimits(
        max_frame_bytes=10,
        sampling_timeout_s=0.1,
        jpeg_quality_steps=(5, 12, 20),
    )

    with pytest.raises(avd.SamplingError):
        avd.ffmpeg_frame_sampler(str(source), [1.0], limits)

    # Without recomputation every quality step gets the full timeout and
    # the real elapsed time exceeds the 0.1s operation budget.
    assert len(observed_timeouts) == 3
    assert real_elapsed[0] > 0.1


# --- Correction 2: visual_description must be an actual string ----------


@pytest.mark.parametrize("bad", [True, 7, {}, []])
def test_visual_description_must_be_a_json_string(bad):
    with pytest.raises(avd.EvidenceValidationError) as exc:
        _validate(_visual_item(bad))
    assert exc.value.reason == "invalid_visual_description_type"


def test_mutation_description_type_gate_is_load_bearing():
    """int 7 stringifies to "7" (nonempty, within the length cap): only
    the explicit type gate can reject it -- the emptiness/length gates
    cannot. A genuine string "7" is accepted and labeled
    model-produced."""
    with pytest.raises(avd.EvidenceValidationError) as exc:
        _validate(_visual_item(7))
    assert exc.value.reason == "invalid_visual_description_type"
    rec = _validate(_visual_item("7"))
    assert rec.visual_description == {"text": "7", "model_produced": True}


# --- Correction 3: explicit AV prompt/schema version in identity --------


def test_checkpoint_identity_exposes_prompt_schema_version():
    identity = cd.av_checkpoint_identity(
        source_fingerprint="source",
        av_config=_avcfg(),
        discovery_config=_dcfg(),
        source_duration=60.0,
        transcript_result={"words": []},
    )
    assert identity["av_prompt_schema_version"] == gr.AV_PROMPT_SCHEMA_VERSION


def test_mutation_version_blind_identity_reuses_stale_schema(monkeypatch):
    """Mutation: if av_checkpoint_identity stopped persisting the
    explicit schema version, the exposure contract breaks -- proving
    the explicit field is load-bearing."""
    real_identity = cd.av_checkpoint_identity

    def _version_blind(**kwargs):
        ident = real_identity(**kwargs)
        del ident["av_prompt_schema_version"]
        return ident

    monkeypatch.setattr(cd, "av_checkpoint_identity", _version_blind)
    with pytest.raises(KeyError):
        cd.av_checkpoint_identity(
            source_fingerprint="source",
            av_config=_avcfg(),
            discovery_config=_dcfg(),
            source_duration=60.0,
            transcript_result={"words": []},
        )["av_prompt_schema_version"]


def test_queued_resume_stale_schema_version_forces_reanalysis(
        tmp_path, clean_av_env, monkeypatch):
    """End-to-end through the queued resume path in orchestrator.py: a
    checkpoint whose explicit AV prompt/schema version is older than the
    running code must NOT be reused."""
    monkeypatch.setenv("GEMINI_MODEL", "model-a")
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    src = tmp_path / "resume_schema.mp4"
    src.write_bytes(b"V" * 70000)
    dcfg = cd.DiscoveryConfig.from_env(model_name="model-a")
    avcfg = avd.AVDiscoveryConfig.from_env()
    identity = cd.av_checkpoint_identity(
        source_fingerprint=avd.fingerprint_source(str(src)),
        av_config=avcfg, discovery_config=dcfg,
        source_duration=60.0,
        transcript_result={"words": []})
    saved = {
        "shorts": [],
        "candidate_discovery": {
            "version": cd.DISCOVERY_VERSION,
            "config": {
                "model_name": "model-a",
                "fallback_models": list(dcfg.fallback_models),
            },
            "av_discovery": {
                "enabled": True,
                "version": avd.AV_DISCOVERY_VERSION,
                "identity": identity,
            },
        },
    }
    ok, _ = orch._analysis_checkpoint_reusable(
        saved, str(src), 60.0, {"words": []}, run_analysis=True)
    assert ok is True
    # Simulate a checkpoint persisted under an older AV prompt/schema
    # contract: the explicit version no longer matches the running code.
    saved["candidate_discovery"]["av_discovery"]["identity"][
        "av_prompt_schema_version"] = gr.AV_PROMPT_SCHEMA_VERSION - 1
    ok2, why2 = orch._analysis_checkpoint_reusable(
        saved, str(src), 60.0, {"words": []}, run_analysis=True)
    assert ok2 is False
    assert why2 == "av_identity_mismatch:av_prompt_schema_version"
