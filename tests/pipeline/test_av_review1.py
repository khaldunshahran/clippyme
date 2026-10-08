"""Task 11 review-01 regression tests (promoted coordinator probes).

All 8 independent production-contract probes from
``work/muse-handoff/task-11/coordinator-review1-remaining.py``, promoted
into durable repository tests and adapted to the corrected contracts
(review-01 half-open frame boundary, strict sampling budget, shared
deadline, provider-contract resume identity), plus one mutation check
per correction: each mutation test FAILS if the listed production
behavior is broken, proving the suite is sensitive to the breakage.

All provider interactions use injected deterministic fakes; no live SDK
calls.
"""
import json
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
        window_target_s=240.0,
        window_overlap_s=45.0,
        boundary_tolerance_s=12.0,
        max_windows=32,
        max_raw_candidates=200,
        max_provider_attempts=24,
        deadline_s=120.0,
        sdk_timeout_ms=120000,
        model_name="primary",
        fallback_models=("fallback",),
    )
    values.update(overrides)
    return cd.DiscoveryConfig(**values)


def _visual_item(frame_id="vf_0_0", *, description="a red card appears"):
    item = {
        "start_s": 10.0,
        "end_s": 30.0,
        "event_type": "visual_reveal",
        "evidence_basis": "visual",
        "evidence_text": "",
        "visual_frame_ids": [frame_id],
        "audio_event_ids": [],
        "context": "the object is introduced before the reveal",
        "expected_payoff": "the viewer sees the completed reveal",
        "confidence": 0.8,
        "recommendation_score": 80,
    }
    if description is not None:
        item["visual_description"] = description
    return item


def _validate_visual(item, *, window_index=0, frame_id="vf_0_0",
                     decoded_ts=20.0):
    return avd.validate_av_evidence(
        item,
        window_start_s=0.0,
        window_end_s=60.0,
        window_index=window_index,
        source_duration=60.0,
        frames_by_id={frame_id: {
            "frame_id": frame_id,
            "requested_ts": decoded_ts,
            "decoded_ts": decoded_ts,
        }},
        audio_events_by_id={},
        window_words=[],
        min_duration=5.0,
        max_duration=60.0,
        source_fingerprint="src-one",
    )


# ---------------------------------------------------------------------------
# The 8 promoted production-contract probes
# ---------------------------------------------------------------------------

class TestReview1Probes:
    def test_live_discovery_prompt_has_one_noncontradictory_av_schema(
            self):
        class Window:
            index = 0
            start_s = 0.0
            end_s = 60.0

        packet = {"manifest": {
            "frames": [{"frame_id": "vf_0_0", "timestamp_s": 20.0}],
            "audio_events": [],
        }}
        prompt = gr.build_discovery_prompt(
            Window(), [], evidence_packet=packet,
            duration_min_s=5.0, duration_max_s=60.0)
        schema = prompt.split("Respond with STRICT JSON only", 1)[1]
        assert '"visual_frame_ids"' in schema
        assert '"audio_event_ids"' in schema
        assert '"visual_description"' in schema
        assert "Every candidate MUST say evidence_basis is transcript" \
            not in prompt
        assert "multimodal moment DISCOVERY engine" in prompt

    def test_discovery_prompt_without_evidence_stays_transcript_only(
            self):
        # The no-evidence path keeps the exact accepted transcript-only
        # prompt: no AV schema fields, transcript-only mandate intact.
        class Window:
            index = 0
            start_s = 0.0
            end_s = 60.0

        prompt = gr.build_discovery_prompt(
            Window(), [], duration_min_s=5.0, duration_max_s=60.0)
        assert '"visual_frame_ids"' not in prompt
        assert '"audio_event_ids"' not in prompt
        assert '"visual_description"' not in prompt
        assert "AUDIOVISUAL EVIDENCE" not in prompt
        assert "Every candidate MUST say evidence_basis is transcript" \
            in prompt

    def test_real_elevenlabs_audio_event_text_schema_is_normalized(
            self):
        events, issues = avd.normalize_audio_events([
            {"text": "(laughter)", "start": 1.0, "end": 2.0},
        ], 10.0)
        assert issues == []
        assert len(events) == 1
        assert events[0]["label"] == "(laughter)"
        assert events[0]["start_s"] == 1.0
        assert events[0]["end_s"] == 2.0

    def test_visual_frame_must_fall_inside_candidate_interval(self):
        with pytest.raises(avd.EvidenceValidationError) as exc:
            _validate_visual(_visual_item(), decoded_ts=50.0)
        assert exc.value.reason == "visual_frame_outside_candidate:vf_0_0"

    def test_visual_candidate_requires_auditable_model_description(self):
        with pytest.raises(avd.EvidenceValidationError) as exc:
            _validate_visual(_visual_item(description=None))
        assert exc.value.reason == "missing_visual_description"

    def test_moment_fingerprint_is_stable_across_overlapping_window_local_ids(
            self):
        first = _validate_visual(
            _visual_item("vf_0_0"), window_index=0,
            frame_id="vf_0_0", decoded_ts=20.0)
        second = _validate_visual(
            _visual_item("vf_1_0"), window_index=1,
            frame_id="vf_1_0", decoded_ts=20.0)
        assert first.candidate_id != second.candidate_id
        assert first.moment_fingerprint == second.moment_fingerprint

        def _record(v):
            return cd.CandidateRecord(
                candidate_id=v.candidate_id,
                window_index=v.window_index,
                start_s=v.start_s,
                end_s=v.end_s,
                event_type=v.event_type,
                evidence_text=v.evidence_text,
                evidence_basis=v.evidence_basis,
                context=v.context,
                expected_payoff=v.expected_payoff,
                event_key=v.event_key,
                confidence=v.confidence,
                recommendation_score=v.recommendation_score,
                speaker_ids=v.speaker_ids,
                moment_fingerprint=v.moment_fingerprint,
                raw={"visual_frame_ids": list(v.visual_frame_ids)},
            )

        dedup = cd.dedupe_candidates([_record(first), _record(second)])
        assert len(dedup["kept"]) == 1
        assert len(dedup["removed"]) == 1

    def test_sampling_budget_never_overshoots_invocation_caps(self):
        cfg = avd.AVDiscoveryConfig(
            max_total_sampled_frames=3,
            max_invocation_bytes=100,
        )
        budget = avd.SamplingBudget(cfg, time_fn=lambda: 0.0)
        assert budget.charge(1, 80) is True
        assert budget.charge(1, 30) is False
        assert budget.frames_used == 1
        assert budget.bytes_used == 80
        assert budget.charge(3, 1) is False
        assert budget.frames_used == 1

    def test_sampling_timeout_is_capped_by_shared_discovery_deadline(
            self, tmp_path, clean_av_env):
        source = tmp_path / "source.mp4"
        source.write_bytes(b"not decoded by injected sampler")
        observed = []

        def sampler(_source, timestamps, limits):
            observed.append(limits.sampling_timeout_s)
            return [avd.FrameSample(t, t, b"JPEG") for t in timestamps]

        def av_generate(_client, _prompt, _parts, _cfg, _budget):
            # Compliant candidate: the single planned frame decodes at
            # 30.0s, inside the [20, 40) interval (half-open boundary).
            item = _visual_item()
            item["start_s"] = 20.0
            item["end_s"] = 40.0
            return json.dumps({"candidates": [item]})

        result = cd.run_windowed_discovery(
            {"words": []}, 60.0,
            config=_dcfg(deadline_s=0.25, max_provider_attempts=2),
            source_path=str(source),
            frame_sampler=sampler,
            av_generate_fn=av_generate,
            av_config=avd.AVDiscoveryConfig(
                media_window_target_s=60.0,
                media_window_overlap_s=0.0,
                max_media_windows=1,
                max_total_sampled_frames=1,
                frames_per_window=1,
                sampling_timeout_s=30.0,
            ),
            min_duration=5.0,
            max_duration=60.0,
            run_editorial=False,
        )
        assert observed
        assert observed[0] <= 0.25
        assert result["candidate_discovery"]["counts_reconcile"] is True

    def test_resume_identity_changes_when_provider_model_changes(self):
        avcfg = avd.AVDiscoveryConfig(
            media_window_target_s=60.0,
            media_window_overlap_s=0.0,
            max_media_windows=2,
            max_total_sampled_frames=2,
            frames_per_window=1,
        )
        one = cd.av_checkpoint_identity(
            source_fingerprint="src-one",
            av_config=avcfg,
            discovery_config=_dcfg(model_name="model-a"),
            source_duration=60.0,
            transcript_result={"words": []},
        )
        two = cd.av_checkpoint_identity(
            source_fingerprint="src-one",
            av_config=avcfg,
            discovery_config=_dcfg(model_name="model-b"),
            source_duration=60.0,
            transcript_result={"words": []},
        )
        assert one != two
        assert (one["provider_contract_digest"]
                != two["provider_contract_digest"])


# ---------------------------------------------------------------------------
# Mutation checks: each FAILS if the listed production behavior is broken
# ---------------------------------------------------------------------------

class TestReview1Mutations:
    def test_mutation_live_prompt_contradiction_detected(
            self, monkeypatch):
        # Correct: with evidence the production prompt has AV schema
        # fields and NO transcript-only mandate.
        def _assert_noncontradictory(prompt):
            schema = prompt.split("Respond with STRICT JSON only", 1)[1]
            assert '"visual_frame_ids"' in schema
            assert '"visual_description"' in schema
            assert "Every candidate MUST say evidence_basis is " \
                "transcript" not in prompt

        class Window:
            index = 0
            start_s = 0.0
            end_s = 60.0

        packet = {"manifest": {
            "frames": [{"frame_id": "vf_0_0", "timestamp_s": 20.0}],
            "audio_events": [],
        }}
        _assert_noncontradictory(gr.build_discovery_prompt(
            Window(), [], evidence_packet=packet,
            duration_min_s=5.0, duration_max_s=60.0))

        # Mutation: the old defect — AV paragraph appended but the
        # transcript-only schema and final mandate kept.
        real_build = gr.build_discovery_prompt

        def _contradictory_build(*args, **kwargs):
            prompt = real_build(*args, **kwargs)
            return prompt.replace(
                "evidence_basis MUST be one of transcript, visual, "
                "audio_event, multimodal",
                'evidence_basis": "transcript') + \
                "\nEvery candidate MUST say evidence_basis is transcript."

        monkeypatch.setattr(gr, "build_discovery_prompt",
                            _contradictory_build)
        with pytest.raises(AssertionError):
            _assert_noncontradictory(gr.build_discovery_prompt(
                Window(), [], evidence_packet=packet,
                duration_min_s=5.0, duration_max_s=60.0))

    def test_mutation_frame_outside_clip_acceptance_detected(
            self, monkeypatch):
        # Correct: a frame decoded outside the candidate interval is
        # rejected with the exact reason.
        with pytest.raises(avd.EvidenceValidationError) as exc:
            _validate_visual(_visual_item(), decoded_ts=50.0)
        assert (exc.value.reason
                == "visual_frame_outside_candidate:vf_0_0")

        # Mutation: a validator that silently accepts out-of-interval
        # frames (the old defect) lets the bad candidate through, so
        # the rejection assertion fails.
        real_validate = avd.validate_av_evidence

        def _accepting_validate(item, **kwargs):
            frames = dict(kwargs.get("frames_by_id") or {})
            frames = {
                fid: dict(f, decoded_ts=item["start_s"])
                for fid, f in frames.items()
            }
            kwargs["frames_by_id"] = frames
            return real_validate(item, **kwargs)

        monkeypatch.setattr(avd, "validate_av_evidence",
                            _accepting_validate)
        # Under the mutant the bad candidate is ACCEPTED, so the
        # rejection assertion below must fail with AssertionError.
        with pytest.raises(AssertionError):
            try:
                _validate_visual(_visual_item(), decoded_ts=50.0)
            except avd.EvidenceValidationError as exc:
                assert (exc.reason
                        == "visual_frame_outside_candidate:vf_0_0")
            else:
                raise AssertionError(
                    "mutation accepted the out-of-interval frame")

    def test_mutation_local_id_fingerprinting_is_unstable(self):
        # The corrected contract: canonical absolute evidence identity
        # is stable across overlapping windows.
        new0 = avd.make_av_moment_fingerprint(
            "src-one", 10.0, 30.0, "visual",
            [("frame", "20.000")], "a red card appears")
        new1 = avd.make_av_moment_fingerprint(
            "src-one", 10.0, 30.0, "visual",
            [("frame", "20.000")], "a red card appears")
        assert new0 == new1
        # Mutation (the old defect): hashing window-local frame IDs
        # makes the same absolute evidence unstable across windows.
        old0 = avd.make_av_moment_fingerprint(
            "src-one", 10.0, 30.0, "visual",
            [("vf", "vf_0_0")], "a red card appears")
        old1 = avd.make_av_moment_fingerprint(
            "src-one", 10.0, 30.0, "visual",
            [("vf", "vf_1_0")], "a red card appears")
        assert old0 != old1, \
            "local-ID hashing must be cross-window unstable (the defect)"

    def test_mutation_fingerprint_includes_model_observation(self):
        # Different visual observations sharing time stay distinct.
        base = dict(source_fingerprint="src-one", start_s=10.0,
                    end_s=30.0, evidence_basis="visual",
                    evidence_refs=[("frame", "20.000")])
        f1 = avd.make_av_moment_fingerprint(
            **base, evidence_text="a red card appears")
        f2 = avd.make_av_moment_fingerprint(
            **base, evidence_text="a blue card appears")
        assert f1 != f2

    def test_mutation_budget_overshoot_detected(self):
        cfg = avd.AVDiscoveryConfig(
            max_total_sampled_frames=3,
            max_invocation_bytes=100,
        )
        # Correct: strict decline before exceeding the caps.
        budget = avd.SamplingBudget(cfg, time_fn=lambda: 0.0)
        assert budget.charge(1, 80) is True
        assert budget.charge(1, 30) is False
        assert (budget.frames_used, budget.bytes_used) == (1, 80)
        # Mutation: the old charge-then-check overshoot.
        real_charge = avd.SamplingBudget.charge

        def _overshoot(self, frames, byte_count):
            if self.exhausted():
                return False
            self.frames_used += frames
            self.bytes_used += byte_count
            return True

        avd.SamplingBudget.charge = _overshoot
        try:
            mutated = avd.SamplingBudget(cfg, time_fn=lambda: 0.0)
            assert mutated.charge(1, 80) is True
            with pytest.raises(AssertionError):
                assert mutated.charge(1, 30) is False
        finally:
            avd.SamplingBudget.charge = real_charge

    def test_mutation_model_insensitive_resume_detected(
            self, monkeypatch):
        def _identity(model):
            return cd.av_checkpoint_identity(
                source_fingerprint="src-one",
                av_config=avd.AVDiscoveryConfig(
                    media_window_target_s=60.0,
                    media_window_overlap_s=0.0,
                    max_media_windows=2,
                    max_total_sampled_frames=2,
                    frames_per_window=1,
                ),
                discovery_config=_dcfg(model_name=model),
                source_duration=60.0,
                transcript_result={"words": []},
            )

        # Correct: a model change changes the persisted identity.
        assert _identity("model-a") != _identity("model-b")
        # Mutation: an identity blind to the provider contract reuses
        # the checkpoint across models.
        real_identity = cd.av_checkpoint_identity

        def _contract_blind(**kwargs):
            ident = real_identity(**kwargs)
            ident["provider_contract_digest"] = "0" * 32
            return ident

        monkeypatch.setattr(cd, "av_checkpoint_identity",
                            _contract_blind)
        with pytest.raises(AssertionError):
            assert _identity("model-a") != _identity("model-b")

    def test_mutation_queued_resume_reuses_stale_model_checkpoint(
            self, tmp_path, clean_av_env, monkeypatch):
        # End-to-end through the queued resume path in orchestrator.py:
        # a checkpoint saved under model-a must NOT be reused under
        # model-b. A model-blind resume check would return reusable.
        from clippyme.pipeline import candidate_discovery as _cd

        monkeypatch.setenv("GEMINI_MODEL", "model-a")
        monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
        src = tmp_path / "resume_mut.mp4"
        src.write_bytes(b"V" * 70000)
        dcfg = _cd.DiscoveryConfig.from_env(model_name="model-a")
        # The queued path recomputes the AV config from the environment
        # too, so the saved checkpoint uses from_env() as well.
        avcfg = avd.AVDiscoveryConfig.from_env()
        saved = {
            "shorts": [],
            "candidate_discovery": {
                "version": _cd.DISCOVERY_VERSION,
                "config": {
                    "model_name": "model-a",
                    "fallback_models": list(dcfg.fallback_models),
                },
                "av_discovery": {
                    "enabled": True,
                    "version": avd.AV_DISCOVERY_VERSION,
                    "identity": _cd.av_checkpoint_identity(
                        source_fingerprint=avd.fingerprint_source(
                            str(src)),
                        av_config=avcfg, discovery_config=dcfg,
                        source_duration=60.0,
                        transcript_result={"words": []}),
                },
            },
        }
        ok, _ = orch._analysis_checkpoint_reusable(
            saved, str(src), 60.0, {"words": []}, run_analysis=True)
        assert ok is True
        monkeypatch.setenv("GEMINI_MODEL", "model-b")
        ok2, why2 = orch._analysis_checkpoint_reusable(
            saved, str(src), 60.0, {"words": []}, run_analysis=True)
        assert ok2 is False
        assert why2 == "av_identity_mismatch:provider_contract_digest"
