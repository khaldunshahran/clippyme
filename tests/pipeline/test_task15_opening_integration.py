"""Task 15: grounded openings — integration, emission, export, mutations.

Covers the orchestrator integration helper (flag OFF no-op, direct vs
queued routes reach the packet with final cutter bounds), project
emission (flag ON: selected/user-authored/none; flag OFF: legacy),
project save/load/version/undo, a real render_project export with a
selected delayed hook (decode before/during/at-exclusive-end), and
fail-then-restore mutations proving each gate is load-bearing.
Injected providers only; synthetic AV fixtures; no live providers.
"""
import asyncio
import json
import os
import subprocess

import pytest

from clippyme.pipeline import opening_plan as op
from clippyme.pipeline.project_emit import _build_project
from clippyme.domain.clip_project import (
    validate_project, project_to_json, project_from_json)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == 0, r.stderr[-1000:]
    return r.stdout


@pytest.fixture
def av_job(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    js = str(job)
    # 6s source slice, 320x180, blue bg, 880Hz tone.
    src = os.path.join(js, "source_clip0.mp4")
    _run(["ffmpeg", "-y", "-v", "error",
          "-f", "lavfi", "-i", "color=blue:s=320x180:r=30:d=6",
          "-f", "lavfi", "-i",
          "sine=frequency=880:sample_rate=48000:duration=6",
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-shortest", src])
    # Final (reframed) mp4: same geometry for the test.
    final = os.path.join(js, "clip0.mp4")
    _run(["ffmpeg", "-y", "-v", "error", "-i", src,
          "-c:v", "libx264", "-pix_fmt", "yuv420p",
          "-c:a", "aac", final])
    return js, src, final


def _words_transcript():
    return {"segments": [{"words": [
        {"word": "hello", "start": 0.5, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 1.5},
        {"word": "this", "start": 2.0, "end": 2.3},
        {"word": "is", "start": 2.3, "end": 2.4},
        {"word": "amazing", "start": 2.4, "end": 3.0},
    ]}]}


class _FakeProvider:
    def __init__(self, options):
        self._options = options
        self.calls = 0

    def request_options(self, request):
        self.calls += 1
        return self._options


def _good_option(**kw):
    base = {
        "id": "op-1", "text": "hello world", "language": "en",
        "rationale": "opens on the exact first spoken words",
        "evidence_basis": "speech",
        "evidence": [{"quote": "hello world", "basis": "speech"}],
        "display": {"start": 2.0, "end": 4.0},
    }
    base.update(kw)
    return base


def _planned_clip(start=0.0, end=6.0, provider_options=None,
                  budget=None, transcript=None, **clip_kw):
    clip = {"start": start, "end": end, "candidate_id": "cand-1",
            "viral_hook_text": "You need to see this"}
    clip.update(clip_kw)
    clips_data = {"transcript": transcript or _words_transcript()}
    return clip, clips_data


def _plan(clip, clips_data, src, provider_options, budget=None):
    budget = budget or op.OpeningBudget(deadline_s=60, max_requests=4,
                                       request_timeout_s=5)
    return op.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=src, index=0,
        budget=budget,
        provider=_FakeProvider(provider_options)
        if provider_options is not None else None)


@pytest.fixture
def clean_state(monkeypatch):
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    op.set_opening_provider(None)
    op.set_opening_fallback_provider(None)
    op.reset_run_budget()
    yield
    op.set_opening_provider(None)
    op.set_opening_fallback_provider(None)
    op.reset_run_budget()


# ---------------------------------------------------------------------------
# Orchestrator integration helper: flag OFF no-op, routes reach packets
# ---------------------------------------------------------------------------

def test_flag_off_is_noop_zero_provider_calls(tmp_path, clean_state,
                                              monkeypatch):
    from clippyme.pipeline.orchestrator import _maybe_plan_opening
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip, clips_data = _planned_clip()
    meta = str(tmp_path / "meta.json")
    prov = _FakeProvider([_good_option()])
    op.set_opening_provider(prov)
    result = _maybe_plan_opening(clip, clips_data, str(src), 0, meta)
    assert result is None
    assert prov.calls == 0
    assert "opening_plan" not in clip
    assert "opening_funnel" not in clips_data


def test_direct_and_queued_routes_reach_packet(tmp_path, clean_state,
                                              monkeypatch):
    from clippyme.pipeline.orchestrator import _maybe_plan_opening
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    plans = {}
    for route in ("direct", "queued"):
        # Mirror production: the clip lives in clips_data["shorts"], so
        # the atomic metadata save persists its opening_plan.
        # Clip [0, 6) keeps the fixture words inside the final interval.
        clip, clips_data = _planned_clip(start=0.0, end=6.0)
        clips_data["shorts"] = [clip]
        meta = str(tmp_path / f"meta_{route}.json")
        op.reset_run_budget()
        op.set_opening_provider(_FakeProvider([_good_option()]))
        # Simulate the post-repair cutter bounds per route.
        plan = _maybe_plan_opening(clip, clips_data, str(src), 0, meta,
                                   route=route)
        assert plan is not None
        plans[route] = plan
        # Atomic metadata persistence carries the plan + funnel.
        saved = json.load(open(meta, encoding="utf-8"))
        assert saved["opening_funnel"]["totals"]["clips"] == 1
        assert saved["shorts"][0]["opening_plan"]["packet_id"] == \
            plan["packet_id"]
    d, q = plans["direct"], plans["queued"]
    # FINAL cutter bounds reach the packet identically on both routes.
    assert d["packet_identity"]["final_start"] == 0.0
    assert q["packet_identity"]["final_start"] == 0.0
    assert d["packet_identity"]["final_end"] == 6.0
    assert q["packet_identity"]["final_end"] == 6.0
    assert (d["packet_identity"]["words_sha256"] ==
            q["packet_identity"]["words_sha256"])
    assert d["route"] == "direct" and q["route"] == "queued"


# ---------------------------------------------------------------------------
# Editorial intentional-empty stays empty
# ---------------------------------------------------------------------------

def test_intentional_editorial_empty_stays_empty(tmp_path, clean_state,
                                                 monkeypatch):
    from clippyme.pipeline.editorial_application import (
        is_intentional_editorial_empty, APPLICATION_VERSION)
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    prov = _FakeProvider([_good_option()])
    op.set_opening_provider(prov)
    # Genuine intentional-empty record: editorial ran, rejected every
    # original selection, zero shorts retained.
    clips_data = {
        "shorts": [],
        "editorial_application": {
            "version": APPLICATION_VERSION,
            "enabled": True,
            "counts_reconcile": True,
            "original_selected_count": 3,
            "final_selected_count": 0,
            "accepts_retained": 0,
            "adjustments_applied": 0,
            "rejects_removed": 3,
            "unrefined_retained": 0,
            "legacy_no_id_retained": 0,
            "groups_applied": 0,
            "application_skips": {},
            "skipped": [],
        },
    }
    assert is_intentional_editorial_empty(clips_data) is True
    # No clips -> the stage never runs -> no provider calls, no funnel,
    # and no resurrection via fallback.
    for _ in clips_data["shorts"]:
        raise AssertionError("unreachable")
    assert prov.calls == 0
    assert "opening_funnel" not in clips_data


# ---------------------------------------------------------------------------
# Project emission: flag ON (selected / user-authored / none)
# ---------------------------------------------------------------------------

def _emit(clip, clips_data, js, src, final, monkeypatch):
    monkeypatch.setenv("CLIPPYME_GROUNDED_OPENINGS", "1")
    return _build_project(
        job_id="t15", index=0, start=clip["start"], end=clip["end"],
        clip=clip, clips_data=clips_data, clip_source=src,
        clip_final=final, job_dir=js)


def test_emission_uses_selected_option(av_job, clean_state, monkeypatch):
    js, src, final = av_job
    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, [_good_option()])
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert len(hooks) == 1
    assert hooks[0].text == "hello world"
    # Explicit Task 06 timing from the selected option, not legacy [0,3].
    assert (hooks[0].start, hooks[0].end) == (2.0, 4.0)
    opening = project.opening
    assert opening.status == "selected"
    assert opening.option_id == "op-1"
    assert opening.display_start == 2.0 and opening.display_end == 4.0
    assert opening.packet_id == clip["opening_plan"]["packet_id"]
    assert opening.evidence[0].basis == "speech"
    # Old automatic hook kept only as unverified legacy metadata.
    assert opening.legacy_unverified_hook == "You need to see this"


def test_emission_no_selection_no_automatic_hook(av_job, clean_state,
                                                monkeypatch):
    js, src, final = av_job
    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, None)  # no provider -> honest no-selection
    assert clip["opening_plan"]["selected"] is None
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    assert [o for o in project.overlays if o.type == "hook"] == []
    assert project.opening.status == "none"
    assert project.opening.no_selection_reason == "no_provider_configured"
    assert project.opening.legacy_unverified_hook == "You need to see this"


def test_emission_user_override_wins(av_job, clean_state, monkeypatch):
    js, src, final = av_job
    clip, clips_data = _planned_clip(hook="My hand-written hook")
    _plan(clip, clips_data, src, [_good_option()])
    assert clip["opening_plan"]["user_authored"] is True
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert len(hooks) == 1
    assert hooks[0].text == "My hand-written hook"  # no provider rewrite
    assert project.opening.status == "user_authored"


def test_emission_flag_off_keeps_legacy(av_job, clean_state, monkeypatch):
    js, src, final = av_job
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    clip, clips_data = _planned_clip()
    data = _build_project(
        job_id="t15", index=0, start=clip["start"], end=clip["end"],
        clip=clip, clips_data=clips_data, clip_source=src,
        clip_final=final, job_dir=js)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert len(hooks) == 1
    assert hooks[0].text == "You need to see this"  # legacy backfill text
    assert (hooks[0].start, hooks[0].end) == (0.0, 3.0)  # legacy timing
    assert project.opening is None  # no Task 15 field when flag OFF


# ---------------------------------------------------------------------------
# Project save/load/version/undo with the opening field
# ---------------------------------------------------------------------------

def test_project_roundtrip_version_undo(av_job, clean_state, monkeypatch):
    js, src, final = av_job
    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, [_good_option()])
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    payload = project_to_json(project)
    assert '"opening"' in payload
    restored = project_from_json(payload)
    assert restored.opening.status == "selected"
    assert restored.opening.option_id == "op-1"
    assert (restored.opening.display_start,
            restored.opening.display_end) == (2.0, 4.0)
    # Version bump + undo-shaped restore keep the opening identity.
    project.version = 2
    assert project_from_json(project_to_json(project)).opening.text == \
        "hello world"
    project.version = 1  # undo
    assert project_from_json(project_to_json(project)).opening.status == \
        "selected"


def test_old_project_without_opening_still_valid(av_job, clean_state,
                                                 monkeypatch):
    js, src, final = av_job
    monkeypatch.delenv("CLIPPYME_GROUNDED_OPENINGS", raising=False)
    clip, clips_data = _planned_clip()
    data = _build_project(
        job_id="t15", index=0, start=clip["start"], end=clip["end"],
        clip=clip, clips_data=clips_data, clip_source=src,
        clip_final=final, job_dir=js)
    project = validate_project(data)
    assert project.opening is None
    restored = project_from_json(project_to_json(project))
    assert restored.opening is None


# ---------------------------------------------------------------------------
# Real render_project export with the selected delayed hook
# ---------------------------------------------------------------------------

def _drive(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    else:
        coro.close()
        raise AssertionError("unexpected suspension")


def _hook_visible(path, t, w=320, h=180):
    import numpy as np
    rgb = _run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path,
                "-frames:v", "1", "-pix_fmt", "rgb24",
                "-f", "rawvideo", "-"])
    arr = np.frombuffer(rgb, dtype=np.uint8).reshape(h, w, 3)
    region = arr[30:120, 80:240]
    is_blue = (region[:, :, 2] > 150) & (region[:, :, 0] < 100)
    return int((~is_blue).sum()) > 300


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_real_export_selected_delayed_hook(av_job, clean_state, monkeypatch):
    import asyncio
    from clippyme.domain import project_render as pr
    from clippyme.pipeline.media_qa import probe_media
    monkeypatch.setenv("CLIPPYME_VIDEO_ENCODER", "libx264")
    js, src, final = av_job
    before_hash = _sha(src)

    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, [_good_option()])  # display [2.0, 4.0)
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)

    async def inline_thread(func, *args, **kwargs):
        return func(*args, **kwargs)
    monkeypatch.setattr(asyncio, "to_thread", inline_thread)

    result = _drive(pr.render_project(
        job_id="t15", clip_index=0, project=project,
        job_dir=js, metadata={}, clip_info={}))
    out = result.output_path
    assert os.path.isfile(out)

    # Decode before/during/at-exclusive-end: delayed hook honored.
    assert not _hook_visible(out, 1.0), "hook visible before start=2.0"
    assert _hook_visible(out, 3.0), "hook not visible inside [2.0, 4.0)"
    assert _hook_visible(out, 3.9), "hook not visible just before end"
    assert not _hook_visible(out, 4.5), "hook visible after end=4.0"

    # Media integrity: geometry, FPS, frames, duration, 880Hz audio.
    media = probe_media(out)
    assert media["width"] == 320 and media["height"] == 180
    assert abs(media["fps"] - 30.0) < 0.5
    n_frames = int(_run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-count_frames", "-show_entries",
                         "stream=nb_read_frames", "-of", "csv=p=0",
                         out]).decode().strip())
    assert abs(n_frames - 180) <= 3
    assert abs(media["duration"] - 6.0) < 0.2
    assert media["has_audio"] is True
    tone = _run(["ffmpeg", "-v", "error", "-i", out, "-map", "a",
                 "-f", "f32le", "-ac", "1", "-ar", "8000", "-"])
    import numpy as np, struct
    samples = np.frombuffer(tone, dtype=np.float32)
    assert len(samples) > 8000
    # 880Hz tone check via zero crossings on a stable slice.
    seg = samples[8000:16000]
    crossings = int(((seg[:-1] < 0) != (seg[1:] < 0)).sum())
    hz = crossings / 2 / (len(seg) / 8000)
    assert abs(hz - 880) < 60, f"audio tone {hz}Hz != 880Hz"
    # Source untouched; retained reframed base intact.
    assert _sha(src) == before_hash


# ---------------------------------------------------------------------------
# Fail-then-restore mutations (each gate is load-bearing)
# ---------------------------------------------------------------------------

def _validate_opt(option, packet):
    norm, rej = op.validate_opening_option(option, packet=packet)
    return norm, rej


def test_mutation_removed_evidence_rejection_is_load_bearing(
        clean_state, monkeypatch):
    from clippyme.pipeline import opening_plan as opm
    pkt = opm.build_opening_packet(
        source_fingerprint="fp", final_start=0.0, final_end=6.0,
        words=[{"word": "hello", "start": 0.5, "end": 1.0}],
        language="en")
    bad = {"id": "x", "text": "goodbye moon", "language": "en",
           "rationale": "r", "evidence_basis": "speech",
           "evidence": [{"quote": "goodbye moon", "basis": "speech"}],
           "display": {"start": 0.0, "end": 2.0}}
    _, rej = _validate_opt(bad, pkt)
    assert rej["code"] == "quote_not_grounded"
    # MUTATION: skip evidence validation -> removed evidence accepted.
    monkeypatch.setattr(opm, "_validate_speech_evidence",
                        lambda items, stream: None)
    norm, rej = _validate_opt(bad, pkt)
    assert norm is not None, "mutation did not break the gate"
    # RESTORE.
    monkeypatch.undo()
    _, rej = _validate_opt(bad, pkt)
    assert rej["code"] == "quote_not_grounded"


def test_mutation_full_token_grounding_is_load_bearing(
        clean_state, monkeypatch):
    from clippyme.pipeline import opening_plan as opm
    pkt = opm.build_opening_packet(
        source_fingerprint="fp", final_start=0.0, final_end=6.0,
        words=[{"word": "hello", "start": 0.5, "end": 1.0}],
        language="en")
    bad = {"id": "x", "text": "hell", "language": "en",
           "rationale": "r", "evidence_basis": "speech",
           "evidence": [{"quote": "hell", "basis": "speech"}],
           "display": {"start": 0.0, "end": 2.0}}
    _, rej = _validate_opt(bad, pkt)
    assert rej["code"] == "quote_not_grounded"
    # MUTATION: substring match instead of complete token slices.
    monkeypatch.setattr(
        opm, "_quote_is_token_slice",
        lambda qt, stream: any(" ".join(qt) in " ".join(stream[i:i + len(qt)])
                               for i in range(len(stream))) if qt else False)
    norm, rej = _validate_opt(bad, pkt)
    assert norm is not None, "mutation did not break the gate"
    # RESTORE.
    monkeypatch.undo()
    _, rej = _validate_opt(bad, pkt)
    assert rej["code"] == "quote_not_grounded"


def test_mutation_final_bound_forwarding_is_load_bearing(
        tmp_path, clean_state, monkeypatch):
    from clippyme.pipeline import opening_plan as opm
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)
    clip = {"start": 4.0, "end": 9.0, "candidate_id": "c"}
    clips_data = {"transcript": {"segments": [{"words": [
        {"word": "hello", "start": 4.5, "end": 5.0}]}]}}
    budget = opm.OpeningBudget(deadline_s=60, max_requests=4,
                               request_timeout_s=5)
    real = opm.build_opening_packet
    # MUTATION: shift the forwarded bounds (simulates pre-repair bounds).
    monkeypatch.setattr(
        opm, "build_opening_packet",
        lambda **kw: real(**{**kw, "final_start": kw["final_start"] - 1.0}))
    plan = opm.plan_clip_opening(
        clip=clip, clips_data=clips_data, input_video=str(src), index=0,
        budget=budget, provider=_FakeProvider([]))
    assert plan["packet_identity"]["final_start"] == 3.0, \
        "mutation did not break bound forwarding"
    # RESTORE.
    monkeypatch.undo()
    opm.reset_run_budget()
    clip2 = {"start": 4.0, "end": 9.0, "candidate_id": "c"}
    clips_data2 = {"transcript": {"segments": [{"words": [
        {"word": "hello", "start": 4.5, "end": 5.0}]}]}}
    plan2 = opm.plan_clip_opening(
        clip=clip2, clips_data=clips_data2, input_video=str(src), index=0,
        budget=opm.OpeningBudget(deadline_s=60, max_requests=4,
                                 request_timeout_s=5),
        provider=_FakeProvider([]))
    assert plan2["packet_identity"]["final_start"] == 4.0
    assert plan2["packet_identity"]["final_end"] == 9.0


def test_mutation_resume_identity_is_load_bearing(
        tmp_path, clean_state, monkeypatch):
    from clippyme.pipeline import opening_plan as opm
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 512)

    def _ctx(words):
        clip = {"start": 0.0, "end": 6.0}
        clips_data = {"transcript": {"segments": [{"words": words}]}}
        return clip, clips_data

    words = [{"word": "hello", "start": 0.5, "end": 1.0}]
    clip, clips_data = _ctx(words)
    budget = opm.OpeningBudget(deadline_s=60, max_requests=8,
                               request_timeout_s=5)
    opt = {"id": "op-1", "text": "hello", "language": "en",
           "rationale": "r", "evidence_basis": "speech",
           "evidence": [{"quote": "hello", "basis": "speech"}],
           "display": {"start": 0.0, "end": 2.0}}
    opm.plan_clip_opening(clip=clip, clips_data=clips_data,
                          input_video=str(src), index=0, budget=budget,
                          provider=_FakeProvider([opt]))
    # Words changed -> must NOT resume.
    clip2, clips_data2 = _ctx([{"word": "goodbye", "start": 0.5, "end": 1.0}])
    clips_data2["shorts"] = [clip2]
    clip2["opening_plan"] = clip["opening_plan"]
    # MUTATION: resume always allowed.
    monkeypatch.setattr(opm, "opening_plan_resumable",
                        lambda saved, packet: (True, "mutated"))
    plan = opm.plan_clip_opening(
        clip=clip2, clips_data=clips_data2, input_video=str(src), index=0,
        budget=budget, provider=None)
    assert plan["resumed"] is True, "mutation did not break the gate"
    # RESTORE.
    monkeypatch.undo()
    clip3, clips_data3 = _ctx([{"word": "goodbye", "start": 0.5, "end": 1.0}])
    clip3["opening_plan"] = clip["opening_plan"]
    plan3 = opm.plan_clip_opening(
        clip=clip3, clips_data=clips_data3, input_video=str(src), index=0,
        budget=opm.OpeningBudget(deadline_s=60, max_requests=8,
                                 request_timeout_s=5),
        provider=_FakeProvider([opt]))
    assert plan3["resumed"] is False


def test_mutation_selected_option_emission_is_load_bearing(
        av_job, clean_state, monkeypatch):
    from clippyme.pipeline import project_emit as pem
    js, src, final = av_job
    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, [_good_option()])
    # MUTATION: emission falls back to the legacy hook text.
    monkeypatch.setattr(
        pem, "_grounded_hook_overlay",
        lambda clip, duration: (
            [{"type": "hook", "text": (clip.get("viral_hook_text") or ""),
              "style": "viral", "start": 0.0, "end": 3.0,
              "position": "top"}],
            {"status": "selected"}))
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert hooks[0].text == "You need to see this", \
        "mutation did not break selected-option emission"
    # RESTORE.
    monkeypatch.undo()
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert hooks[0].text == "hello world"
    assert (hooks[0].start, hooks[0].end) == (2.0, 4.0)


def test_mutation_explicit_timing_emission_is_load_bearing(
        av_job, clean_state, monkeypatch):
    from clippyme.pipeline import project_emit as pem
    js, src, final = av_job
    clip, clips_data = _planned_clip()
    _plan(clip, clips_data, src, [_good_option()])  # display [2.0, 4.0)
    real = pem._grounded_hook_overlay

    def _legacy_timing(clip, duration):
        overlays, opening = real(clip, duration)
        for o in overlays:
            o["start"] = 0.0
            o["end"] = round(min(3.0, duration), 3)
        return overlays, opening

    # MUTATION: drop the explicit display interval for legacy timing.
    monkeypatch.setattr(pem, "_grounded_hook_overlay", _legacy_timing)
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert (hooks[0].start, hooks[0].end) == (0.0, 3.0), \
        "mutation did not break explicit timing"
    # RESTORE: the delayed/exclusive interval is carried again.
    monkeypatch.undo()
    data = _emit(clip, clips_data, js, src, final, monkeypatch)
    project = validate_project(data)
    hooks = [o for o in project.overlays if o.type == "hook"]
    assert (hooks[0].start, hooks[0].end) == (2.0, 4.0)
