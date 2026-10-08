"""Task 11 acceptance tests: bounded audiovisual discovery.

These tests exercise PRODUCTION entry points with injected deterministic
providers only (no live SDK calls, no network):

- ``candidate_discovery.run_windowed_discovery`` (with ``source_path``,
  ``frame_sampler``, ``av_generate_fn`` / ``av_parse_fn`` seams),
- ``main.get_viral_clips`` (the real direct entry point; heavy CV/ML
  imports are stubbed at module load so the real ``main.py`` code runs),
- ``orchestrator._load_or_analyze`` (the real queued analysis function),
- ``candidate_discovery.av_resume_reusable`` /
  ``orchestrator._analysis_checkpoint_reusable`` (checkpoint/resume gates).

Every acceptance bullet from ``work/muse-handoff/task-11/TASK.md`` is
covered, plus one mutation-sensitive gate per listed mutation. A gate
test either asserts the correct behavior directly (which a breaking
mutation would fail) or additionally demonstrates sensitivity by
monkeypatching the production seam to the broken behavior and showing
the acceptance assertion fails.

Conventions: ``SRC`` helpers build deterministic fixtures; the fake
frame sampler never touches real media; the fake Gemini client parses
the REAL production prompt text (window range, manifest frame IDs,
transcript word lines) so provider items stay consistent with the
evidence the production code actually sampled.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# ---------------------------------------------------------------------------
# Import-time stubs for main.py's heavy CV/ML dependencies.
#
# The laptop's host python lacks ultralytics/torch/mediapipe/psutil and the
# google-genai SDK (those ship in the Docker backend image). The stubs below
# let the REAL main.py module import and run its pure entry points
# (get_viral_clips -> _get_viral_clips_windowed); nothing in the exercised
# path touches the stubbed libraries at runtime. The google namespace
# package itself is left intact -- only a ``google.genai`` submodule stub
# is registered, exposing an injectable ``Client`` and ``types.Part``.
# ---------------------------------------------------------------------------

def _install_heavy_stubs():
    if getattr(_install_heavy_stubs, "done", False):
        return
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        mod = types.ModuleType("ultralytics")
        mod.YOLO = type("YOLO", (), {})
        sys.modules["ultralytics"] = mod
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_mod = types.ModuleType("torch")
        cuda_mod = types.ModuleType("torch.cuda")
        cuda_mod.is_available = lambda: False
        torch_mod.cuda = cuda_mod
        torch_mod.device = lambda x: x
        sys.modules["torch"] = torch_mod
        sys.modules["torch.cuda"] = cuda_mod
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        sys.modules["mediapipe"] = types.ModuleType("mediapipe")
    try:
        import psutil  # noqa: F401
    except ImportError:
        psutil_mod = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil_mod.virtual_memory = lambda: _VM()
        psutil_mod.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil_mod
    # google.genai stub: keep the real `google` namespace package, only
    # register the submodule. `Client` raises until a test monkeypatches
    # it with a deterministic fake; `types.Part.from_bytes` builds plain
    # data (the fakes never inspect SDK parts).
    import google  # noqa: F401  (real namespace package, left intact)
    if "google.genai" not in sys.modules:
        genai_mod = types.ModuleType("google.genai")

        class _PartFactory:
            @classmethod
            def from_bytes(cls, data=None, mime_type=None):
                return {"data": data, "mime_type": mime_type}

        types_mod = types.ModuleType("google.genai.types")
        types_mod.Part = _PartFactory
        genai_mod.types = types_mod

        class _UnpatchedClient:
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "google.genai.Client must be monkeypatched in tests")

        genai_mod.Client = _UnpatchedClient
        sys.modules["google.genai"] = genai_mod
        sys.modules["google.genai.types"] = types_mod
        google.genai = genai_mod
    _install_heavy_stubs.done = True


_install_heavy_stubs()

from clippyme.pipeline import av_discovery as avd  # noqa: E402
from clippyme.pipeline import candidate_discovery as cd  # noqa: E402
from clippyme.pipeline import editorial_application as ea  # noqa: E402
from clippyme.pipeline import editorial_refinement as er  # noqa: E402
from clippyme.pipeline import gemini_parser as gp  # noqa: E402
from clippyme.pipeline import gemini_request as gr  # noqa: E402
from clippyme.pipeline import orchestrator as orch  # noqa: E402
from clippyme.domain.runtime_state import RuntimeState  # noqa: E402
import clippyme.pipeline.main as main_module  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / fakes
# ---------------------------------------------------------------------------

_WINDOW_RANGE_RE = re.compile(
    r"Window absolute time range: ([\d.]+)s to ([\d.]+)s")
_MANIFEST_FRAME_RE = re.compile(r"^\s+frame\s+(vf_\d+_\d+):", re.MULTILINE)
_WORD_LINE_RE = re.compile(r"\[([\d.]+)\]\[spk:(-?\d+)\]\s+(\S+)")


def prompt_window_range(prompt):
    m = _WINDOW_RANGE_RE.search(prompt)
    assert m, "production prompt must state the absolute window range"
    return float(m.group(1)), float(m.group(2))


def prompt_frame_ids(prompt):
    """Frame IDs from the prompt's manifest block, in manifest order."""
    return _MANIFEST_FRAME_RE.findall(prompt)


def prompt_words(prompt):
    """(timestamp, text) pairs from the prompt's transcript word lines."""
    return [(float(m.group(1)), m.group(3))
            for m in _WORD_LINE_RE.finditer(prompt)]


def prompt_speakers(prompt):
    return {m.group(2) for m in _WORD_LINE_RE.finditer(prompt)}


def quote_for(prompt, start_s, end_s):
    return " ".join(t for (ts, t) in prompt_words(prompt)
                    if start_s <= ts < end_s)


def make_av_config(**kw):
    base = dict(
        enabled=True,
        media_window_target_s=120.0,
        media_window_overlap_s=30.0,
        max_media_windows=8,
        max_total_sampled_frames=64,
        frames_per_window=3,
        frame_max_width=320,
        frame_max_pixels=102400,
        max_frame_bytes=65536,
        max_window_bytes=262144,
        max_invocation_bytes=4 * 1024 * 1024,
        sampling_timeout_s=30.0,
        fingerprint_probe_bytes=65536,
        max_audio_event_label_chars=64,
        max_visual_description_chars=500,
        max_manifest_frames=12,
        max_manifest_events=20,
    )
    base.update(kw)
    return avd.AVDiscoveryConfig(**base)


def make_discovery_config(**kw):
    base = dict(
        window_target_s=120.0,
        window_overlap_s=30.0,
        boundary_tolerance_s=5.0,
        max_windows=8,
        max_raw_candidates=50,
        max_provider_attempts=20,
        deadline_s=120.0,
        sdk_timeout_ms=5000,
        model_name="fake-model",
        fallback_models=(),
    )
    base.update(kw)
    return cd.DiscoveryConfig(**base)


def make_source_file(tmp_path, name="source.mp4", nbytes=70000, seed=b"V"):
    p = tmp_path / name
    p.write_bytes((seed * ((nbytes // len(seed)) + 1))[:nbytes])
    return str(p)


def make_words(n, start=0.0, step=1.0):
    """Normalized words {"w","s","e","sp"} with speaker 0."""
    return [{"w": f"word{i}", "s": start + i * step,
             "e": start + i * step + step * 0.9, "sp": 0}
            for i in range(n)]


def clustered_transcript():
    """Speech clusters [10,70) and [300,360); 600s source."""
    return {"words": make_words(60, start=10.0) + make_words(60, start=300.0)}


class FakeSampler:
    """Deterministic injectable frame sampler.

    ``fail_ranges``: list of (start, end); any timestamp inside fails the
    whole window with RuntimeError. ``write_temp_dir``: when set, each
    call writes temp files there and removes them before returning
    (seam contract: a custom sampler cleans up after itself); the paths
    are recorded for the cleanup assertion.
    """

    def __init__(self, fail_ranges=(), byte_payload=64, write_temp_dir=None):
        self.calls = []
        self.fail_ranges = list(fail_ranges)
        self.byte_payload = byte_payload
        self.write_temp_dir = write_temp_dir
        self.temp_paths_written = []

    def __call__(self, source_path, timestamps, limits=None):
        self.calls.append((source_path, [float(t) for t in timestamps]))
        for (fs, fe) in self.fail_ranges:
            if any(fs <= t < fe for t in timestamps):
                raise RuntimeError("decode exploded")
        tmp_paths = []
        try:
            if self.write_temp_dir:
                os.makedirs(self.write_temp_dir, exist_ok=True)
                for i, t in enumerate(timestamps):
                    p = os.path.join(
                        self.write_temp_dir, f"frame_{t:.3f}_{i}.tmp")
                    with open(p, "wb") as f:
                        f.write(b"TMP" * 16)
                    tmp_paths.append(p)
                    self.temp_paths_written.append(p)
            return [avd.FrameSample(
                requested_ts=float(t), decoded_ts=float(t),
                frame_bytes=b"JPEG" + bytes([int(t) % 256])
                * self.byte_payload)
                for t in timestamps]
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass


class FakeGenaiClient:
    """Deterministic stand-in for genai.Client (no network).

    ``produce`` is a class-level hook: contents -> response text. The
    instance records generate/close calls so tests can assert the
    exactly-once client lifecycle.
    """
    instances = []
    produce = staticmethod(lambda contents: json.dumps({"candidates": []}))

    def __init__(self, *args, **kwargs):
        self.close_calls = 0
        self.generate_calls = []
        FakeGenaiClient.instances.append(self)

    @property
    def models(self):
        return self

    def generate_content(self, model=None, contents=None, config=None):
        self.generate_calls.append(
            {"model": model, "contents": contents, "config": config})
        return SimpleNamespace(text=FakeGenaiClient.produce(contents))

    def close(self):
        self.close_calls += 1


# -- provider item builders (raw provider shapes; the strict production
# -- parser assigns local candidate IDs / moment fingerprints) --------------

def visual_candidate(prompt, fids, *, score=80.0,
                     desc="a person raises both arms in celebration",
                     n_refs=2, start_off=2.0, dur=20.0, basis="visual",
                     start_s=None, end_s=None, extra=None):
    ws, we = prompt_window_range(prompt)
    s = round(ws + start_off if start_s is None else start_s, 3)
    e = round(min(s + dur, we - 0.5) if end_s is None else end_s, 3)
    item = {
        "start_s": s,
        "end_s": e,
        "event_type": "moment",
        "evidence_basis": basis,
        "evidence_text": "",
        "visual_frame_ids": list(fids[:n_refs]),
        "visual_description": desc,
        "context": "a clear visual moment with a beginning and an end",
        "expected_payoff": "the visual payoff lands inside the interval",
        "confidence": 0.8,
        "recommendation_score": score,
    }
    if extra:
        item.update(extra)
    return item


def av_transcript_candidate(prompt, start_s, end_s, *, score=80.0,
                            basis="transcript"):
    return {
        "start_s": start_s,
        "end_s": end_s,
        "event_type": "story",
        "evidence_basis": basis,
        "evidence_text": quote_for(prompt, start_s, end_s),
        "context": "a coherent spoken passage with a clear setup",
        "expected_payoff": "the spoken payoff lands before the interval ends",
        "confidence": 0.8,
        "recommendation_score": score,
    }


def multimodal_candidate(prompt, fids, start_s, end_s, *, score=80.0,
                         desc="a speaker gestures at a whiteboard"):
    item = av_transcript_candidate(prompt, start_s, end_s, score=score)
    item["evidence_basis"] = "multimodal"
    item["visual_frame_ids"] = list(fids[:2])
    item["visual_description"] = desc
    return item


def t08_transcript_candidate(prompt, start_s, end_s, *, score=80.0):
    """Task-08 transcript-path provider item (for the disabled AV path)."""
    return av_transcript_candidate(prompt, start_s, end_s, score=score)


def av_generate_from_plan(plan_fn):
    """Adapt plan_fn(prompt, frame_ids) -> items into the av_generate_fn seam."""
    def _gen(client, prompt, image_parts, cfg, budget):
        fids = [p["frame_id"] for p in (image_parts or [])]
        return json.dumps({"candidates": plan_fn(prompt, fids)})
    return _gen


def one_visual_per_window(prompt, fids):
    if not fids:
        return []
    return [visual_candidate(prompt, fids)]


def run_av(transcript, duration, source_path, sampler, item_plan, *,
           av_config="default", discovery_config=None,
           min_duration=10.0, max_duration=60.0,
           run_editorial=None, client_factory=None, **kw):
    """Run the production entry point with injected fakes.

    ``av_config="default"`` uses the small deterministic test config;
    ``av_config=None`` passes through to the production
    ``AVDiscoveryConfig.from_env()`` (what the real direct/queued entry
    points use).
    """
    call_kw = dict(
        config=discovery_config or make_discovery_config(),
        client_factory=client_factory,
        av_generate_fn=av_generate_from_plan(item_plan),
        source_path=source_path,
        frame_sampler=sampler,
        av_config=(make_av_config() if av_config == "default"
                   else av_config),
        min_duration=min_duration,
        max_duration=max_duration,
    )
    if run_editorial is not None:
        call_kw["run_editorial"] = run_editorial
    call_kw.update(kw)
    return cd.run_windowed_discovery(transcript, duration, **call_kw)


def av_funnel(result):
    return result["candidate_discovery"]["av_discovery"]


def store_by_basis(result, basis):
    return [e for e in result["candidate_store"].values()
            if e.get("evidence_basis") == basis]


@pytest.fixture
def clean_av_env(monkeypatch):
    """Neutral AV/editorial env for deterministic runs."""
    monkeypatch.delenv("CLIPPYME_AV_DISCOVERY", raising=False)
    monkeypatch.delenv("CLIPPYME_MIN_VIRAL_SCORE", raising=False)
    monkeypatch.delenv("CLIPPYME_MAX_CLIPS", raising=False)
    monkeypatch.delenv("CLIPPYME_SILENCE_SNAP", raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# 1. Wordless source -> bounded media windows covering [0, duration) and one
#    visual candidate grounded in real supplied frame IDs.
# ---------------------------------------------------------------------------

class TestWordlessSourceMediaWindows:
    def test_wordless_source_yields_visual_candidate(self, tmp_path,
                                                     clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        result = run_av({"words": []}, 600.0, src, sampler,
                        one_visual_per_window)
        funnel = result["candidate_discovery"]
        avf = funnel["av_discovery"]

        assert avf["enabled"] is True
        assert avf["media_windows_planned"] >= 1
        # Bounded media windows covering [0, source_duration).
        wins = sorted(avf["window_results"], key=lambda r: r["start_s"])
        assert wins[0]["start_s"] == 0.0
        assert wins[-1]["end_s"] == 600.0
        cursor = 0.0
        for w in wins:
            assert w["start_s"] <= cursor + 1e-6, "gap in media coverage"
            cursor = max(cursor, w["end_s"])
        assert cursor >= 600.0 - 1e-6
        # The sampler was actually used (bounded media windows attempted).
        assert sampler.calls, "no media sampling happened"
        assert avf["media_windows_attempted"] == len(wins)
        assert avf["frames_sampled_total"] > 0
        # Exact sampled-frame timestamps recorded, no bytes persisted:
        # frame metadata carries IDs + timestamps + sizes only.
        for w in wins:
            assert w["frame_count"] == 3
            for f in w["frames"]:
                assert set(f.keys()) == {"frame_id", "requested_ts",
                                         "decoded_ts", "byte_size"}
                assert w["start_s"] <= f["decoded_ts"] <= w["end_s"]
        # Window-level "frame_bytes" is a byte TOTAL (allowed metadata);
        # per-frame payloads never persist (key set checked above).
        # One visual candidate per window, grounded in supplied frame IDs.
        visuals = store_by_basis(result, "visual")
        assert len(visuals) == avf["media_windows_succeeded"] >= 1
        sampled_ids = {f["frame_id"]
                       for w in wins for f in w["frames"]}
        for v in visuals:
            assert v["candidate_id"].startswith("avcand-")
            assert v["visual_frame_ids"], "visual candidate without frames"
            assert set(v["visual_frame_ids"]) <= sampled_ids
            assert v["source_fingerprint"] == avf["source_fingerprint"]
            assert v["wordless"] is True
        assert funnel["counts_reconcile"] is True
        assert avf["counts_reconcile"] is True
        # Selected shorts carry the visual candidates.
        short_ids = {s["candidate_id"] for s in result["shorts"]}
        assert {v["candidate_id"] for v in visuals} <= short_ids


# ---------------------------------------------------------------------------
# 2. Speech-light gap planned for media coverage; first/last intervals and
#    all Task 08 words remain covered.
# ---------------------------------------------------------------------------

class TestSpeechLightGapPlanning:
    def test_gap_planned_words_covered(self, tmp_path, clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        transcript = clustered_transcript()
        result = run_av(transcript, 600.0, src, sampler,
                        lambda prompt, fids: [])
        funnel = result["candidate_discovery"]
        avf = funnel["av_discovery"]

        # All Task 08 words remain covered by transcript windows.
        cov = funnel["transcript_coverage"]
        assert cov["valid_words"] == 120
        assert cov["covered_words"] == 120
        assert cov["first_covered"] is True
        assert cov["last_covered"] is True
        assert cov["uncovered_word_ranges"] == []
        # First and last source intervals are covered by media windows.
        wins = sorted(avf["window_results"], key=lambda r: r["start_s"])
        assert wins[0]["start_s"] == 0.0
        assert wins[-1]["end_s"] == 600.0
        cursor = 0.0
        for w in wins:
            assert w["start_s"] <= cursor + 1e-6
            cursor = max(cursor, w["end_s"])
        assert cursor >= 600.0 - 1e-6
        # The speech-light gap (70, 300) is planned with media-only windows.
        gap = [w for w in wins
               if w["start_s"] < 300.0 and w["end_s"] > 70.0
               and w["window_kind"] == "media"]
        assert gap, "speech-light gap has no media-only windows"
        # Transcript windows coalesced where speech exists.
        kinds = {w["window_kind"] for w in wins}
        assert "transcript" in kinds
        assert funnel["counts_reconcile"] is True


# ---------------------------------------------------------------------------
# 3. Synthetic audiovisual fixture sampled by the PRODUCTION sampler at
#    in-range absolute timestamps; frame bytes capped; decoded colors
#    distinguishable; source hash unchanged.
# ---------------------------------------------------------------------------

FFMPEG = shutil.which("ffmpeg")


def _build_tricolor_fixture(path):
    """9s mp4: red 0-3s, green 3-6s, blue 6-9s; all-intra for exact seeks."""
    segs = []
    for color in ("red", "green", "blue"):
        segs += ["-f", "lavfi", "-i",
                 f"color=c={color}:size=64x64:rate=10:duration=3"]
    for vcodec in ("libx264", "mpeg4"):
        cmd = ([FFMPEG, "-y"] + segs +
               ["-filter_complex",
                "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p",
                "-g", "1", "-c:v", vcodec, str(path)])
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode == 0 and Path(path).stat().st_size > 0:
            return
    raise RuntimeError("could not build synthetic fixture")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")
class TestProductionSamplerOnSyntheticFixture:
    def test_absolute_timestamps_colors_distinguishable_hash_stable(
            self, tmp_path):
        PIL_Image = pytest.importorskip("PIL.Image")
        from io import BytesIO
        src = tmp_path / "fixture.mp4"
        _build_tricolor_fixture(src)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        lim = avd.FrameLimits()
        # Production sampler, injectable via the seam, at absolute
        # in-range timestamps (one per color segment).
        samples = avd.ffmpeg_frame_sampler(str(src), [1.5, 4.5, 7.5], lim)
        assert len(samples) == 3
        # Source opened read-only: never modified.
        assert hashlib.sha256(src.read_bytes()).hexdigest() == before
        prev_ts = -1.0
        for s, want_ts, want_dom in zip(samples, (1.5, 4.5, 7.5),
                                        ("r", "g", "b")):
            assert s.requested_ts == pytest.approx(want_ts)
            # Absolute, in-range decoded timestamp (never rebased).
            assert 0.0 <= s.decoded_ts < 9.0
            assert abs(s.decoded_ts - want_ts) < 0.2
            assert s.decoded_ts > prev_ts
            prev_ts = s.decoded_ts
            # Per-frame byte cap enforced.
            assert 0 < len(s.frame_bytes) <= lim.max_frame_bytes
            img = PIL_Image.open(BytesIO(s.frame_bytes)).convert("RGB")
            px = list(img.getdata())
            n = len(px)
            mean = [sum(c[i] for c in px) / n for i in range(3)]
            dom = "rgb"[mean.index(max(mean))]
            assert dom == want_dom, f"ts={want_ts}: means={mean}"
            others = [m for i, m in enumerate(mean) if "rgb"[i] != dom]
            assert max(mean) - max(others) > 40  # distinguishable


# ---------------------------------------------------------------------------
# 4. Transcript evidence keeps exact token-slice grounding and speaker ID 0;
#    a mixed (multimodal) candidate references both a real quote and
#    supplied frames.
# ---------------------------------------------------------------------------

class TestTranscriptGroundingAndSpeakerZero:
    def test_token_slice_speaker_zero_and_multimodal(self, tmp_path,
                                                     clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        transcript = {"words": make_words(90, start=10.0)}  # [10,100)

        def plan(prompt, fids):
            ws, we = prompt_window_range(prompt)
            assert (ws, we) == (0.0, 120.0)
            items = [av_transcript_candidate(prompt, 15.0, 35.0, score=82.0)]
            if fids:
                items.append(multimodal_candidate(prompt, fids, 50.0, 70.0,
                                                  score=88.0))
            return items

        result = run_av(transcript, 120.0, src, sampler, plan)
        store = result["candidate_store"]
        transcripts = store_by_basis(result, "transcript")
        multis = store_by_basis(result, "multimodal")
        assert len(transcripts) == 1
        assert len(multis) == 1

        t = transcripts[0]
        # Exact token-slice grounding: the evidence text is a contiguous
        # slice of the candidate interval's words.
        words = transcript["words"]
        interval_toks = [w["w"] for w in words
                         if t["start_s"] <= w["s"] < t["end_s"]]
        ev_toks = t["evidence_text"].split()
        assert ev_toks, "evidence text must not be empty"
        assert any(interval_toks[i:i + len(ev_toks)] == ev_toks
                   for i in range(len(interval_toks) - len(ev_toks) + 1))
        # Speaker ID 0 preserved exactly.
        assert t["speaker_ids"] == [0]

        m = multis[0]
        # Mixed candidate: real quote AND supplied frames.
        assert m["evidence_text"].split(), "multimodal needs the quote"
        interval_toks = [w["w"] for w in words
                         if m["start_s"] <= w["s"] < m["end_s"]]
        ev_toks = m["evidence_text"].split()
        assert any(interval_toks[i:i + len(ev_toks)] == ev_toks
                   for i in range(len(interval_toks) - len(ev_toks) + 1))
        assert m["speaker_ids"] == [0]
        # Frame IDs referenced are the actually sampled ones.
        wins = av_funnel(result)["window_results"]
        sampled = {f["frame_id"] for w in wins for f in w["frames"]}
        assert set(m["visual_frame_ids"]) <= sampled
        assert m["visual_frame_ids"], "multimodal needs frame refs"
        # Visual description labeled as model-produced evidence.
        assert m["visual_description"]["model_produced"] is True
        # Both reach the selected shorts.
        short_ids = {s["candidate_id"] for s in result["shorts"]}
        assert t["candidate_id"] in short_ids
        assert m["candidate_id"] in short_ids
        assert result["candidate_discovery"]["counts_reconcile"] is True


# ---------------------------------------------------------------------------
# 5. Foreign frame/event IDs, audio events with invalid types/times,
#    invented transcript phrases, wordless candidates without media
#    evidence, out-of-window intervals, and duration violations are
#    isolated with exact reasons.
# ---------------------------------------------------------------------------

class TestRejectionReasonsIsolated:
    def test_invalid_items_isolated_with_exact_reasons(self, tmp_path,
                                                       clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        # Words in [10,70); one valid + several invalid audio events.
        transcript = {
            "words": make_words(60, start=10.0),
            "audio_events": [
                {"label": "applause", "start_s": 20.0, "end_s": 25.0},
                {"label": "reversed", "start_s": 30.0, "end_s": 20.0},
                {"label": "", "start_s": 40.0, "end_s": 41.0},
                {"label": "x" * 65, "start_s": 60.0, "end_s": 61.0},
                # Pre-filtered by the overlap gate (never normalized):
                {"label": "bad-bool", "start_s": True, "end_s": 5.0},
                {"label": "too-late", "start_s": 500.0, "end_s": 510.0},
            ],
        }

        def plan(prompt, fids):
            ws, we = prompt_window_range(prompt)
            items = []
            if fids:
                widx = fids[0].split("_")[1]
                # Valid controls: one visual + one audio_event candidate.
                items.append(visual_candidate(prompt, fids, score=70.0))
                items.append({
                    "start_s": 18.0, "end_s": 28.0,
                    "event_type": "moment", "evidence_basis": "audio_event",
                    "evidence_text": "",
                    "audio_event_ids": [f"ae_w{widx}_0"],
                    "context": "crowd applause swells then fades",
                    "expected_payoff": "the applause peak lands in interval",
                    "confidence": 0.8, "recommendation_score": 72.0,
                })
            # Invalid: foreign frame ID.
            items.append(visual_candidate(
                prompt, fids, score=95.0,
                extra={"visual_frame_ids": ["vf_99_9"]}))
            # Invalid: unknown audio event ID.
            items.append({
                "start_s": 18.0, "end_s": 28.0,
                "event_type": "moment", "evidence_basis": "audio_event",
                "evidence_text": "",
                "audio_event_ids": ["ae_99"],
                "context": "some context with enough detail here",
                "expected_payoff": "some payoff with enough detail here",
                "confidence": 0.8, "recommendation_score": 60.0,
            })
            # Invalid: invented transcript phrase.
            items.append(av_transcript_candidate(
                prompt, 15.0, 35.0, score=61.0) | {
                    "evidence_text": "flibbertigibbet never spoken here"})
            # Invalid: wordless transcript candidate without media evidence
            # (interval [95,115] overlaps no words).
            items.append({
                "start_s": 95.0, "end_s": 115.0,
                "event_type": "story", "evidence_basis": "transcript",
                "evidence_text": "",
                "context": "some context with enough detail here",
                "expected_payoff": "some payoff with enough detail here",
                "confidence": 0.8, "recommendation_score": 62.0,
            })
            # Invalid: out-of-window interval.
            items.append(av_transcript_candidate(
                prompt, we - 5.0, we + 15.0, score=63.0))
            # Invalid: duration violations (policy 10..60 here).
            items.append(av_transcript_candidate(
                prompt, 30.0, 33.0, score=64.0))
            long_q = quote_for(prompt, 30.0, 100.0)
            items.append({
                "start_s": 30.0, "end_s": 100.0,
                "event_type": "story", "evidence_basis": "transcript",
                "evidence_text": long_q,
                "context": "some context with enough detail here",
                "expected_payoff": "some payoff with enough detail here",
                "confidence": 0.8, "recommendation_score": 65.0,
            })
            # Invalid: transcript basis carrying frame refs.
            items.append(av_transcript_candidate(
                prompt, 15.0, 35.0, score=66.0) | {
                    "visual_frame_ids": fids[:1]})
            return items

        result = run_av(transcript, 240.0, src, sampler, plan)
        funnel = result["candidate_discovery"]
        avf = funnel["av_discovery"]
        reasons = funnel["invalid_reasons"]

        # Exact reason strings, each isolated (one bad item never blocks
        # the others).
        assert reasons.get("unknown_frame_id:vf_99_9", 0) >= 1
        assert reasons.get("unknown_audio_event_id:ae_99", 0) >= 1
        assert reasons.get("evidence_not_grounded", 0) >= 1
        assert reasons.get("wordless_without_evidence", 0) >= 1
        assert reasons.get("interval_out_of_window", 0) >= 1
        assert reasons.get("duration_below_minimum", 0) >= 1
        assert reasons.get("duration_above_maximum", 0) >= 1
        assert reasons.get("unexpected_visual_reference", 0) >= 1
        # The valid controls survived alongside the rejections.
        assert len(store_by_basis(result, "visual")) >= 1
        assert len(store_by_basis(result, "audio_event")) >= 1
        # Audio events: 1 valid normalized; the 3 invalid events that
        # overlap a window are isolated (boolean-time / out-of-source
        # events are pre-filtered by the overlap gate, never normalized).
        assert avf["audio_events_normalized"] >= 1
        assert avf["audio_event_issues"] == 3
        # Invalid candidates never reach the store or the shorts: every
        # store entry is a validated candidate (here: all AV).
        for entry in result["candidate_store"].values():
            assert entry["candidate_id"].startswith("avcand-")
        short_ids = {s["candidate_id"] for s in result["shorts"]}
        assert set(result["candidate_store"]) == short_ids
        assert funnel["invalid_candidates"] >= 8
        assert funnel["counts_reconcile"] is True
        assert avf["counts_reconcile"] is True

    def test_audio_event_invalid_reasons_exact(self):
        events, issues = avd.normalize_audio_events([
            {"label": "ok", "start_s": 1.0, "end_s": 2.0},
            "not-a-dict",
            {"label": "ok", "start_s": True, "end_s": 2.0},
            {"label": "ok", "start_s": 5.0, "end_s": 4.0},
            {"label": "ok", "start_s": -1.0, "end_s": 2.0},
        ], 600.0)
        assert [e["event_id"] for e in events] == ["ae_0"]
        assert [i["reason"] for i in issues] == [
            "event_not_object",
            "boolean_event_time",
            "event_time_reversed",
            "event_out_of_source_bounds",
        ]


# ---------------------------------------------------------------------------
# 6. Two different visual moments sharing time survive; a genuine repeated
#    moment with matching evidence identity dedupes; overlap alone never
#    removes one.
# ---------------------------------------------------------------------------

class TestDedupeOverlapNeverDedupes:
    def _run(self, tmp_path, item_plan):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        return run_av({"words": []}, 240.0, src, sampler, item_plan)

    def test_distinct_moments_survive_repeat_dedupes(self, tmp_path,
                                                     clean_av_env):
        def plan(prompt, fids):
            if not fids:
                return []
            ws, we = prompt_window_range(prompt)
            a = visual_candidate(prompt, fids, score=80.0,
                                 desc="a person raises both arms",
                                 n_refs=1, start_off=2.0, dur=20.0)
            # Different visual moment sharing time: different frame,
            # different description -> different evidence identity.
            b = visual_candidate(prompt, fids[1:], score=78.0,
                                 desc="the crowd waves flags wildly",
                                 n_refs=1, start_off=10.0, dur=20.0)
            # Genuine repeat of A: same interval, same refs, same text.
            a_repeat = dict(a, recommendation_score=75.0)
            return [a, b, a_repeat]

        result = self._run(tmp_path, plan)
        funnel = result["candidate_discovery"]
        avf = funnel["av_discovery"]
        # The genuine repeat deduped on the moment fingerprint...
        assert funnel["duplicates_removed"] >= 1
        assert avf["duplicates_removed_av"] >= 1
        assert any(d["basis"] == "av_moment_fingerprint"
                   for d in funnel["duplicate_details"])
        # ...while the two different overlapping moments both survive
        # selection (post-dedupe shorts, not just the store).
        short_ids = {s["candidate_id"] for s in result["shorts"]}
        descs = [e["visual_description"]["text"]
                 for e in store_by_basis(result, "visual")
                 if e["candidate_id"] in short_ids]
        assert "a person raises both arms" in descs
        assert "the crowd waves flags wildly" in descs
        kept = [e for e in store_by_basis(result, "visual")
                if e["candidate_id"] in short_ids]
        assert len(kept) >= 2
        assert funnel["counts_reconcile"] is True

    def test_overlap_alone_never_dedupes_av(self, tmp_path, clean_av_env):
        # Direct production dedupe entry point: two AV records with
        # heavily overlapping intervals but different evidence identity.
        recs = []
        for i, (desc, fid, score) in enumerate(
                [("red car passes", "vf_0_0", 80.0),
                 ("blue car passes", "vf_0_1", 79.0)]):
            recs.append(cd.CandidateRecord(
                candidate_id=f"avcand-test{i:02d}",
                window_index=0, start_s=10.0, end_s=40.0,
                event_type="moment", evidence_text="",
                evidence_basis="visual", context="c", expected_payoff="p",
                event_key=None, confidence=0.8,
                recommendation_score=score, speaker_ids=[],
                moment_fingerprint=avd.make_av_moment_fingerprint(
                    "src-x", 10.0, 40.0, "visual", [fid], desc),
                raw={"visual_frame_ids": [fid]},
            ))
        dedup = cd.dedupe_candidates(recs)
        assert len(dedup["kept"]) == 2, \
            "overlap alone must never dedupe AV candidates"
        assert dedup["removed"] == {}
        # And a genuine repeat (identical fingerprint) does collapse.
        repeat = cd.CandidateRecord(
            candidate_id="avcand-test02", window_index=0,
            start_s=10.0, end_s=40.0, event_type="moment", evidence_text="",
            evidence_basis="visual", context="c", expected_payoff="p",
            event_key=None, confidence=0.8, recommendation_score=70.0,
            speaker_ids=[],
            moment_fingerprint=recs[0].moment_fingerprint,
            raw={"visual_frame_ids": ["vf_0_0"]},
        )
        dedup2 = cd.dedupe_candidates(recs + [repeat])
        assert len(dedup2["kept"]) == 2
        assert len(dedup2["removed"]) == 1


# ---------------------------------------------------------------------------
# 7. Explicit final caps/floors apply only after global aggregation,
#    including visual candidates; every funnel removal reconciles.
# ---------------------------------------------------------------------------

class TestFinalCapsFloorsAfterAggregation:
    def test_caps_apply_to_av_candidates(self, tmp_path, clean_av_env,
                                         monkeypatch):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        scores_by_window = {0: 90.0, 1: 80.0, 2: 70.0, 3: 60.0}

        def plan(prompt, fids):
            if not fids:
                return []
            widx = int(fids[0].split("_")[1])
            return [visual_candidate(prompt, fids,
                                     score=scores_by_window[widx])]

        monkeypatch.setenv("CLIPPYME_MIN_VIRAL_SCORE", "75")
        result = run_av({"words": []}, 360.0, src, sampler, plan,
                        max_clips=1,
                        av_config=make_av_config(
                            media_window_target_s=120.0,
                            media_window_overlap_s=30.0,
                            max_media_windows=8),
                        discovery_config=make_discovery_config(
                            max_provider_attempts=20))
        funnel = result["candidate_discovery"]
        avf = funnel["av_discovery"]

        # Aggregation first: all 4 candidates post-dedupe...
        assert funnel["post_dedupe_candidates"] == 4
        assert funnel["duplicates_removed"] == 0
        # ...then the floor removes 70 and 60, then the cap removes 80.
        assert funnel["score_floor_removals"] == 2
        assert funnel["final_cap_removals"] == 1
        assert funnel["final_selected"] == 1
        assert result["shorts"][0]["viral_score"] == 90
        # The removals are AV candidates (caps include visual candidates).
        removed_ids = []
        for r in funnel["rejections"]:
            if r.get("reason") == "max_clips_cap":
                removed_ids = r.get("removed_ids", [])
        assert removed_ids and all(str(i).startswith("avcand-")
                                   for i in removed_ids)
        # Every funnel removal reconciles, globally and in the AV section.
        assert funnel["counts_reconcile"] is True
        assert avf["counts_reconcile"] is True
        c4 = (funnel["post_dedupe_candidates"]
              == funnel["score_floor_removals"]
              + funnel["final_cap_removals"] + funnel["final_selected"])
        assert c4
        av_c4 = (sum(avf["valid_by_basis"].values())
                 == avf["final_selected_av"] + avf["duplicates_removed_av"]
                 + funnel["score_floor_removals"]
                 + funnel["final_cap_removals"])
        assert av_c4


# ---------------------------------------------------------------------------
# 8. Disabling via CLIPPYME_AV_DISCOVERY=0/false/no or omitting the source
#    executes the accepted Task 10 discovery/editorial/application behavior
#    with zero sampler calls and byte-equivalent compatible metadata
#    outside the new disabled record.
# ---------------------------------------------------------------------------

class TestDisabledPathMatchesTask10:
    def _run_transcript_only(self, transcript, duration, **kw):
        prompts = []

        def _gen(client, prompt, cfg, budget):
            prompts.append(prompt)
            ws, we = prompt_window_range(prompt)
            # One grounded transcript candidate per window.
            s = round(ws + 2.0, 3)
            e = round(min(s + 20.0, we - 0.5), 3)
            return json.dumps({"candidates": [
                t08_transcript_candidate(prompt, s, e)]})

        def _parse(text):
            return json.loads(text)["candidates"]

        call_kw = dict(config=make_discovery_config(),
                       client_factory=None,
                       generate_fn=_gen, parse_fn=_parse,
                       min_duration=10.0, max_duration=60.0)
        call_kw.update(kw)
        result = cd.run_windowed_discovery(transcript, duration, **call_kw)
        return result, prompts

    def test_flag_values_disable(self, tmp_path, clean_av_env, monkeypatch):
        transcript = {"words": make_words(60, start=10.0)}
        for flag in ("0", "false", "no"):
            monkeypatch.setenv("CLIPPYME_AV_DISCOVERY", flag)
            src = make_source_file(tmp_path, name=f"src_{flag}.mp4")
            sampler = FakeSampler()
            result, prompts = self._run_transcript_only(
                transcript, 240.0, source_path=src,
                frame_sampler=sampler)
            avf = result["candidate_discovery"]["av_discovery"]
            assert avf == {"enabled": False, "reason": "flag_disabled",
                           "version": avd.AV_DISCOVERY_VERSION}, flag
            assert sampler.calls == [], "sampler must never run disabled"
            assert not any("AUDIOVISUAL EVIDENCE" in p for p in prompts)
            assert not any("SPARSE OBSERVATION" in p for p in prompts)
            # Task 10 transcript behavior intact: grounded candidates found.
            assert result["shorts"], "transcript discovery broke"

    def test_no_source_disables(self, tmp_path, clean_av_env):
        transcript = {"words": make_words(60, start=10.0)}
        sampler = FakeSampler()
        result, _ = self._run_transcript_only(
            transcript, 240.0, source_path=None, frame_sampler=sampler)
        avf = result["candidate_discovery"]["av_discovery"]
        assert avf["enabled"] is False
        assert avf["reason"] == "no_source"
        assert sampler.calls == []

    def test_disabled_metadata_byte_equivalent(self, tmp_path, clean_av_env,
                                               monkeypatch):
        transcript = {"words": make_words(60, start=10.0)}
        src = make_source_file(tmp_path)
        monkeypatch.setenv("CLIPPYME_AV_DISCOVERY", "0")
        res_flag, _ = self._run_transcript_only(
            transcript, 240.0, source_path=src,
            frame_sampler=FakeSampler())
        monkeypatch.delenv("CLIPPYME_AV_DISCOVERY")
        res_none, _ = self._run_transcript_only(
            transcript, 240.0, source_path=None,
            frame_sampler=FakeSampler())
        f1 = dict(res_flag["candidate_discovery"])
        f2 = dict(res_none["candidate_discovery"])
        r1, r2 = f1.pop("av_discovery"), f2.pop("av_discovery")
        assert r1["enabled"] is False and r2["enabled"] is False
        assert r1["reason"] == "flag_disabled"
        assert r2["reason"] == "no_source"
        # Everything outside the new disabled record is byte-equivalent.
        assert (json.dumps(f1, sort_keys=True)
                == json.dumps(f2, sort_keys=True))
        assert (json.dumps(res_flag["shorts"], sort_keys=True)
                == json.dumps(res_none["shorts"], sort_keys=True))


# ---------------------------------------------------------------------------
# 9. A valid speech-free candidate reaches both the actual direct and
#    queued render-selection loops as unrefined, while a rejected/invalid
#    one never renders.
#
#    The render-selection loops are `for clip in clips_data['shorts']` in
#    main.py (direct) and `for index, clip in enumerate(clips)` with
#    `clips = clips_data.get("shorts")` in orchestrator.run (queued): both
#    render exactly the shorts list these entry points produce.
# ---------------------------------------------------------------------------

def _direct_av_produce(contents):
    """Fake provider for the direct/queued entry tests.

    One valid visual candidate per media window plus one INVALID item
    with a deliberately higher score (unknown frame ID) -- proving the
    invalid one never reaches the render-selection loops even when
    top-ranked.
    """
    prompt = contents[0] if isinstance(contents, list) else contents
    fids = prompt_frame_ids(prompt)
    items = []
    if fids:
        items.append(visual_candidate(prompt, fids, score=80.0))
        bad = visual_candidate(prompt, fids, score=95.0,
                               desc="an invented spectacular moment")
        bad["visual_frame_ids"] = ["vf_99_9"]
        items.append(bad)
    return json.dumps({"candidates": items})


@pytest.fixture
def direct_av_stack(tmp_path, monkeypatch):
    """Inject deterministic fakes under the real main/orchestrator."""
    monkeypatch.delenv("CLIPPYME_AV_DISCOVERY", raising=False)
    monkeypatch.delenv("CLIPPYME_MIN_VIRAL_SCORE", raising=False)
    monkeypatch.delenv("CLIPPYME_MAX_CLIPS", raising=False)
    monkeypatch.setenv("CLIPPYME_SILENCE_SNAP", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    FakeGenaiClient.instances.clear()
    FakeGenaiClient.produce = staticmethod(_direct_av_produce)
    sampler = FakeSampler()
    monkeypatch.setattr(main_module.genai, "Client", FakeGenaiClient)
    monkeypatch.setattr(avd, "ffmpeg_frame_sampler", sampler)
    src = make_source_file(tmp_path, name="direct_src.mp4")
    return SimpleNamespace(src=src, sampler=sampler)


def _assert_unrefined_reachability(result, *, expect_windows):
    funnel = result["candidate_discovery"]
    avf = funnel["av_discovery"]
    assert avf["enabled"] is True
    # The valid speech-free candidates were selected...
    visuals = store_by_basis(result, "visual")
    assert len(visuals) == expect_windows
    short_ids = {s["candidate_id"] for s in result["shorts"]}
    assert {v["candidate_id"] for v in visuals} <= short_ids
    # ...carry unrefined plans with the exact Task 11 reason...
    plans = {p["candidate_id"]: p for p in result["editorial_plans"]}
    for v in visuals:
        p = plans[v["candidate_id"]]
        assert p["verdict"] == "unrefined"
        assert p["status"] == "unrefined"
        assert p["reason"] == "speech_free_av_no_transcript_evidence"
        # ...keep the discovery interval exactly...
        assert p["start_s"] == v["start_s"]
        assert p["end_s"] == v["end_s"]
    # ...and Task 10 retains them in the rendered shorts.
    app = result["editorial_application"]
    assert app["unrefined_retained"] == expect_windows
    for s in result["shorts"]:
        v = result["candidate_store"][s["candidate_id"]]
        assert (s["start"], s["end"]) == (v["start_s"], v["end_s"])
    # The invalid top-scored item never renders.
    assert funnel["invalid_reasons"].get(
        "unknown_frame_id:vf_99_9", 0) == expect_windows
    for s in result["shorts"]:
        assert "invented spectacular" not in json.dumps(s)
    assert funnel["counts_reconcile"] is True


class TestRenderSelectionReachability:
    def test_direct_entry_unrefined_reaches_shorts(self, direct_av_stack):
        result = main_module.get_viral_clips(
            {"words": []}, 240.0, source_path=direct_av_stack.src,
            min_duration=10.0, max_duration=60.0)
        assert isinstance(result, dict) and result["shorts"]
        wins = av_funnel(result)["window_results"]
        _assert_unrefined_reachability(result, expect_windows=len(wins))
        # Discovery client closed exactly once.
        assert len(FakeGenaiClient.instances) == 1
        assert FakeGenaiClient.instances[0].close_calls == 1
        # Media identity persisted on the direct path.
        ident = av_funnel(result)["identity"]
        assert ident["source_fingerprint"] == avd.fingerprint_source(
            direct_av_stack.src)
        assert ident["av_discovery_version"] == avd.AV_DISCOVERY_VERSION

    def test_queued_entry_unrefined_reaches_shorts(self, tmp_path,
                                                   direct_av_stack):
        out_dir = tmp_path / "queued_out"
        out_dir.mkdir()
        args = SimpleNamespace(
            skip_analysis=False, instructions=None,
            min_duration=10.0, max_duration=60.0,
            min_clips=None, max_clips=None, clip_type=None,
            duration_mode=None, aspect="9:16", monitor=False,
            reframe_mode="auto",
        )
        state = RuntimeState(str(out_dir), job_id="av-accept")
        clips_data, metadata_file = orch._load_or_analyze(
            args, direct_av_stack.src, "test-video", str(out_dir), 240.0,
            {"words": []}, state, main_module)
        assert clips_data["shorts"]
        wins = clips_data["candidate_discovery"][
            "av_discovery"]["window_results"]
        _assert_unrefined_reachability(clips_data,
                                       expect_windows=len(wins))
        # The queued checkpoint persists matching media identity.
        assert os.path.isfile(metadata_file)
        with open(metadata_file, encoding="utf-8") as f:
            saved = json.load(f)
        saved_av = saved["candidate_discovery"]["av_discovery"]
        assert saved_av["identity"]["source_fingerprint"] == \
            avd.fingerprint_source(direct_av_stack.src)
        assert saved_av["enabled"] is True
        # AV evidence refs + modality persisted in the candidate store
        # (no raw images or full responses).
        for cid, entry in saved["candidate_store"].items():
            if cid.startswith("avcand-"):
                assert entry["visual_frame_ids"]
                assert entry["source_fingerprint"]
        for w in saved_av["window_results"]:
            for f in w["frames"]:
                assert set(f.keys()) == {"frame_id", "requested_ts",
                                         "decoded_ts", "byte_size"}
        # Discovery client closed exactly once on the queued path too.
        assert len(FakeGenaiClient.instances) == 1
        assert FakeGenaiClient.instances[0].close_calls == 1

    def test_queued_fallback_preserves_av_metadata(self, tmp_path,
                                                   direct_av_stack,
                                                   monkeypatch):
        """When discovery yields no shorts, the queued whole-video fallback
        preserves the candidate_discovery AV record and candidate store."""
        monkeypatch.setattr(
            FakeGenaiClient, "produce",
            staticmethod(lambda contents: json.dumps({"candidates": []})))
        out_dir = tmp_path / "queued_fb"
        out_dir.mkdir()
        args = SimpleNamespace(
            skip_analysis=False, instructions=None,
            min_duration=10.0, max_duration=60.0,
            min_clips=None, max_clips=None, clip_type=None,
            duration_mode=None, aspect="9:16", monitor=False,
            reframe_mode="auto",
        )
        state = RuntimeState(str(out_dir), job_id="av-accept-fb")
        clips_data, _ = orch._load_or_analyze(
            args, direct_av_stack.src, "test-video", str(out_dir), 240.0,
            {"words": []}, state, main_module)
        cdf = clips_data["candidate_discovery"]
        assert cdf.get("fallback_used") is True
        assert cdf.get("fallback_strategy") == "whole_video"
        # The AV discovery record (identity + window accounting) survives
        # the fallback instead of being dropped.
        assert cdf["av_discovery"]["enabled"] is True
        assert cdf["av_discovery"]["identity"]["source_fingerprint"] == \
            avd.fingerprint_source(direct_av_stack.src)
        assert "candidate_store" in clips_data


# ---------------------------------------------------------------------------
# 10. Direct and queued checkpoint/resume paths persist matching media
#     identity and do NOT reuse stale source/config/evidence records
#     (mismatched fingerprint -> re-run, not silent reuse).
# ---------------------------------------------------------------------------

class TestCheckpointResumeIdentity:
    def _run_and_save(self, tmp_path, src_name="resume_src.mp4"):
        src = make_source_file(tmp_path, name=src_name)
        sampler = FakeSampler()
        # NOTE: av_config=None -> AVDiscoveryConfig.from_env(), exactly
        # what the production direct/queued entry points use, so the
        # orchestrator resume wrapper (which also recomputes from env)
        # sees matching identities.
        dcfg = make_discovery_config()
        transcript = {"words": []}
        result = run_av(transcript, 240.0, src, sampler,
                        one_visual_per_window,
                        av_config=None, discovery_config=dcfg)
        acfg = avd.AVDiscoveryConfig.from_env()
        return result, src, transcript, acfg, dcfg

    def test_direct_identity_matches_recomputed(self, tmp_path,
                                                clean_av_env):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        saved = result["candidate_discovery"]["av_discovery"]["identity"]
        # The persisted identity matches a fresh recomputation: source
        # fingerprint, AV config/version, window plan, evidence manifest.
        fresh = cd.av_checkpoint_identity(
            source_fingerprint=avd.fingerprint_source(src),
            av_config=acfg, discovery_config=dcfg,
            source_duration=240.0, transcript_result=transcript)
        assert saved == fresh
        assert saved["source_fingerprint"] == avd.fingerprint_source(src)

    def test_resume_reusable_when_identities_match(self, tmp_path,
                                                   clean_av_env):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        ok, why = cd.av_resume_reusable(
            result["candidate_discovery"],
            source_path=src, source_duration=240.0,
            transcript_result=transcript,
            av_config=acfg, discovery_config=dcfg)
        assert (ok, why) == (True, "av_identity_match")

    def test_stale_source_fingerprint_forces_rerun(self, tmp_path,
                                                   clean_av_env):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        # A different source file (different fingerprint): must re-run.
        src2 = make_source_file(tmp_path, name="resume_src2.mp4",
                                seed=b"W")
        assert avd.fingerprint_source(src2) != avd.fingerprint_source(src)
        ok, why = cd.av_resume_reusable(
            result["candidate_discovery"],
            source_path=src2, source_duration=240.0,
            transcript_result=transcript,
            av_config=acfg, discovery_config=dcfg)
        assert ok is False
        assert why == "av_identity_mismatch:source_fingerprint"

    def test_stale_config_digest_forces_rerun(self, tmp_path, clean_av_env):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        saved = json.loads(json.dumps(result["candidate_discovery"]))
        saved["av_discovery"]["identity"]["av_config_digest"] = "0" * 32
        ok, why = cd.av_resume_reusable(
            saved, source_path=src, source_duration=240.0,
            transcript_result=transcript,
            av_config=acfg, discovery_config=dcfg)
        assert (ok, why) == (False, "av_identity_mismatch:av_config_digest")

    def test_enablement_change_forces_rerun(self, tmp_path, clean_av_env,
                                            monkeypatch):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        monkeypatch.setenv("CLIPPYME_AV_DISCOVERY", "0")
        ok, why = cd.av_resume_reusable(
            result["candidate_discovery"],
            source_path=src, source_duration=240.0,
            transcript_result=transcript,
            discovery_config=dcfg)
        assert (ok, why) == (False, "av_enablement_changed")

    def test_disabled_both_reusable(self, tmp_path, clean_av_env,
                                    monkeypatch):
        monkeypatch.setenv("CLIPPYME_AV_DISCOVERY", "0")
        saved = {"version": cd.DISCOVERY_VERSION,
                 "av_discovery": {"enabled": False, "reason": "flag_disabled",
                                  "version": avd.AV_DISCOVERY_VERSION}}
        ok, why = cd.av_resume_reusable(
            saved, source_path=None, source_duration=240.0,
            transcript_result={"words": []})
        assert (ok, why) == (True, "av_disabled_both")

    def test_queued_wrapper_rejects_stale_source(self, tmp_path,
                                                 clean_av_env):
        result, src, transcript, acfg, dcfg = self._run_and_save(tmp_path)
        saved_meta = {"shorts": result["shorts"],
                      "candidate_discovery": result["candidate_discovery"]}
        ok, why = orch._analysis_checkpoint_reusable(
            saved_meta, src, 240.0, transcript, run_analysis=True)
        assert (ok, why) == (True, "av_identity_match")
        src2 = make_source_file(tmp_path, name="resume_src3.mp4",
                                seed=b"Z")
        ok2, why2 = orch._analysis_checkpoint_reusable(
            saved_meta, src2, 240.0, transcript, run_analysis=True)
        assert ok2 is False
        assert why2 == "av_identity_mismatch:source_fingerprint"

    def test_migrated_task10_record_never_silently_enabled(
            self, clean_av_env):
        # A Task 10 (v1) record migrates with the AV section disabled;
        # resume then treats it as enablement-changed, never as a match.
        v1 = {"version": 1, "windows_planned": 2}
        migrated = cd.migrate_candidate_discovery_record(v1)
        assert migrated["av_discovery"]["enabled"] is False
        assert migrated["version"] == cd.DISCOVERY_VERSION


# ---------------------------------------------------------------------------
# 11. Provider/sampling budget exhaustion, frame decode failure, client
#     setup/replacement failure, and parse failure isolate work, preserve
#     transcript candidates, clean temporary assets, and close owned
#     resources exactly once.
# ---------------------------------------------------------------------------

class TestFailureIsolationAndLifecycle:
    def test_provider_budget_exhaustion_skips_cleanly(self, tmp_path,
                                                      clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        clients = []

        def _factory():
            c = FakeGenaiClient()
            clients.append(c)
            return c

        result = run_av({"words": []}, 240.0, src, sampler,
                        one_visual_per_window,
                        client_factory=_factory,
                        discovery_config=make_discovery_config(
                            max_provider_attempts=0))
        funnel = result["candidate_discovery"]
        assert funnel["budget"]["exhausted"] is True
        assert funnel["windows_skipped"] == funnel["windows_planned"] > 0
        assert funnel["windows_attempted"] == 0
        assert result["shorts"] == []
        assert sampler.calls == [], "no sampling when budget exhausted"
        # Owned client closed exactly once.
        assert len(clients) == 1
        assert clients[0].close_calls == 1
        assert funnel["counts_reconcile"] is True

    def test_sampling_budget_exhaustion_bounds_frames(self, tmp_path,
                                                      clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        # Frame-count cap enforced at plan time: trailing windows are
        # reported unattempted with exact intervals/reasons.
        result = run_av(
            {"words": []}, 360.0, src, sampler, one_visual_per_window,
            av_config=make_av_config(max_total_sampled_frames=3,
                                     frames_per_window=3),
            discovery_config=make_discovery_config(
                max_provider_attempts=30))
        avf = av_funnel(result)
        assert avf["frames_sampled_total"] <= 3
        assert avf["media_windows_planned"] == 1
        un = avf["media_windows_unattempted"]
        assert len(un) == 3
        assert all(u["reason"] == "frame_budget_cap" for u in un)
        assert all(u["end_s"] > u["start_s"] for u in un)
        assert funnel_reconciles(result)
        # Byte cap enforced at runtime: later windows skip once the
        # sampling budget is exhausted.
        sampler2 = FakeSampler()
        result2 = run_av(
            {"words": []}, 360.0, src, sampler2, one_visual_per_window,
            av_config=make_av_config(max_invocation_bytes=100),
            discovery_config=make_discovery_config(
                max_provider_attempts=30))
        avf2 = av_funnel(result2)
        assert avf2["media_windows_planned"] == 4
        assert avf2["media_windows_skipped"] >= 1
        assert avf2["frame_bytes_total"] <= 100 + 3 * 68
        assert funnel_reconciles(result2)

    def test_decode_failure_isolates_window(self, tmp_path, clean_av_env):
        src = make_source_file(tmp_path)
        # Fail the middle media window's timestamps only.
        sampler = FakeSampler(fail_ranges=[(100.0, 220.0)])
        result = run_av({"words": []}, 360.0, src, sampler,
                        one_visual_per_window)
        avf = av_funnel(result)
        assert avf["transcript_only_fallbacks"] >= 1
        assert any(d["reason"] == "sampler_error"
                   for d in avf["fallback_details"])
        # Other windows still produced visual candidates.
        assert len(store_by_basis(result, "visual")) >= 1
        assert result["shorts"]
        assert funnel_reconciles(result)

    def test_client_setup_failure_skips_all_windows(self, tmp_path,
                                                    clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()

        def _boom():
            raise RuntimeError("no client today")

        result = run_av({"words": []}, 240.0, src, sampler,
                        one_visual_per_window, client_factory=_boom)
        funnel = result["candidate_discovery"]
        assert any(r.get("reason") == "client_setup_failed"
                   for r in funnel["rejections"])
        assert funnel["windows_skipped"] == funnel["windows_planned"] > 0
        assert result["shorts"] == []
        assert sampler.calls == []
        assert funnel["counts_reconcile"] is True

    def test_editorial_client_replacement_failure_keeps_candidates(
            self, tmp_path, clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        transcript = {"words": make_words(60, start=10.0)}

        def plan(prompt, fids):
            ws, we = prompt_window_range(prompt)
            items = []
            if fids:
                widx = int(fids[0].split("_")[1])
                if widx == 0:
                    items.append(visual_candidate(prompt, fids, score=70.0))
            items.append(av_transcript_candidate(prompt, 15.0, 35.0,
                                                 score=75.0))
            return items

        def _boom():
            raise RuntimeError("editorial client exploded")

        result = run_av(transcript, 240.0, src, sampler, plan,
                        editorial_client_factory=_boom)
        # Both candidates overlap speech, so neither is speech-free: the
        # editorial client replacement failure marks each unrefined with
        # the exact reason instead of erasing them.
        reasons = {p["candidate_id"]: p["reason"]
                   for p in result["editorial_plans"]}
        assert reasons, "expected editorial plans"
        assert set(reasons.values()) <= {
            "client_setup_failed", "speech_free_av_no_transcript_evidence"}
        assert any(r == "client_setup_failed" for r in reasons.values())
        assert result["shorts"], "candidates must be retained"
        assert funnel_reconciles(result)

    def test_parse_failure_isolates_window(self, tmp_path, clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        calls = {"n": 0}

        def _parse(raw_text, evidence_context):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("provider returned garbage")
            return gp.parse_av_discovery_payload(
                raw_text, evidence_context=evidence_context)

        result = run_av({"words": []}, 360.0, src, sampler,
                        one_visual_per_window, av_parse_fn=_parse,
                        discovery_config=make_discovery_config(
                            max_provider_attempts=30))
        funnel = result["candidate_discovery"]
        statuses = [w["status"]
                    for w in funnel["av_discovery"]["window_results"]]
        assert "parse_failed" in statuses
        assert "ok" in statuses
        # Candidates from the surviving windows are preserved.
        assert len(store_by_basis(result, "visual")) >= 1
        assert funnel_reconciles(result)

    def test_temp_assets_cleaned_and_client_closed_once(self, tmp_path,
                                                        clean_av_env):
        src = make_source_file(tmp_path)
        tmpd = tmp_path / "sampler_tmp"
        sampler = FakeSampler(write_temp_dir=str(tmpd))
        clients = []

        def _factory():
            c = FakeGenaiClient()
            clients.append(c)
            return c

        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = run_av({"words": []}, 240.0, src, sampler,
                            one_visual_per_window,
                            client_factory=_factory)
        finally:
            os.chdir(cwd)
        assert result["shorts"]
        # The custom sampler cleaned its own temp files (seam contract).
        assert sampler.temp_paths_written, "sampler wrote no temp files"
        leftovers = list(tmpd.iterdir()) if tmpd.exists() else []
        assert leftovers == [], f"temp files leaked: {leftovers}"
        # No stray files from the run in the working directory.
        stray = [p for p in tmp_path.iterdir()
                 if p.is_file() and p.suffix == ".tmp"]
        assert stray == []
        # Owned client closed exactly once.
        assert len(clients) == 1
        assert clients[0].close_calls == 1

    def test_total_sampling_failure_preserves_transcript(self, tmp_path,
                                                         clean_av_env):
        src = make_source_file(tmp_path)
        sampler = FakeSampler(fail_ranges=[(0.0, 10000.0)])
        transcript = {"words": make_words(60, start=10.0)}

        def _gen(client, prompt, cfg, budget):
            ws, we = prompt_window_range(prompt)
            s = round(ws + 2.0, 3)
            e = round(min(s + 20.0, we - 0.5), 3)
            return json.dumps({"candidates": [
                t08_transcript_candidate(prompt, s, e)]})

        def _parse(text):
            return json.loads(text)["candidates"]

        result = cd.run_windowed_discovery(
            transcript, 240.0,
            config=make_discovery_config(),
            client_factory=None,
            generate_fn=_gen, parse_fn=_parse,
            source_path=src, frame_sampler=sampler,
            av_config=make_av_config(),
            min_duration=10.0, max_duration=60.0)
        avf = av_funnel(result)
        # Every attempted window fell back to transcript-only evidence...
        assert avf["transcript_only_fallbacks"] == \
            avf["media_windows_attempted"] > 0
        # ...yet transcript-grounded candidates were preserved.
        assert len(store_by_basis(result, "transcript")) >= 1
        assert result["shorts"]
        assert all(e.get("evidence_basis") == "transcript"
                   for e in result["candidate_store"].values())
        assert funnel_reconciles(result)


def funnel_reconciles(result):
    f = result["candidate_discovery"]
    avf = f["av_discovery"]
    return bool(f["counts_reconcile"] and avf["counts_reconcile"])


# ---------------------------------------------------------------------------
# Mutation-sensitive gates.
#
# Each gate is a test that FAILS if the listed production behavior is
# broken. The primary acceptance tests above are themselves gates; the
# tests here additionally execute the mutation (via monkeypatch on the
# production seam -- src/ stays untouched) and show the acceptance
# assertion failing, proving the suite is sensitive to the breakage.
# ---------------------------------------------------------------------------

class TestMutationGates:
    def _wordless_run(self, tmp_path):
        src = make_source_file(tmp_path)
        sampler = FakeSampler()
        return run_av({"words": []}, 600.0, src, sampler,
                      one_visual_per_window)

    def test_gate_a_removing_media_only_windows(self, tmp_path,
                                                clean_av_env, monkeypatch):
        real_plan = avd.plan_media_windows

        def _no_media_windows(duration, transcript_windows, config=None):
            plan = real_plan(duration, transcript_windows, config)
            plan["windows"] = [w for w in plan["windows"]
                               if w.kind != "media"]
            return plan

        monkeypatch.setattr(avd, "plan_media_windows", _no_media_windows)
        # The acceptance property -- a wordless source is fully covered
        # by bounded media windows -- now fails.
        with pytest.raises(AssertionError):
            result = self._wordless_run(tmp_path)
            wins = sorted(av_funnel(result)["window_results"],
                          key=lambda r: r["start_s"])
            assert wins, "no windows planned at all"
            assert wins[0]["start_s"] == 0.0
            assert wins[-1]["end_s"] == 600.0
        # And without the mutation the property holds.
        monkeypatch.undo()
        result = self._wordless_run(tmp_path)
        wins = av_funnel(result)["window_results"]
        assert wins[0]["start_s"] == 0.0
        assert wins[-1]["end_s"] == 600.0

    def test_gate_b_rebased_frame_timestamps_rejected(self, tmp_path,
                                                       clean_av_env):
        # Unit contract: window-relative decoded timestamps are rejected
        # by the sampler seam...
        w = avd.MediaWindow(index=1, start_s=100.0, end_s=200.0,
                            kind="media")

        def _rebased(source_path, timestamps, limits=None):
            base = min(timestamps)
            return [avd.FrameSample(requested_ts=t,
                                    decoded_ts=t - base,
                                    frame_bytes=b"J" * 32)
                    for t in timestamps]

        res = avd.sample_media_window("/fake", w, _rebased,
                                      avd.FrameLimits())
        assert res.status == "failed"
        assert res.reason == "decoded_timestamp_out_of_window"
        # ...and by the manifest builder (defense in depth).
        with pytest.raises(ValueError, match="outside window"):
            gr.build_av_evidence_manifest(
                w, [{"frame_id": "vf_1_0", "requested_ts": 120.0,
                     "decoded_ts": 20.0}])
        # End to end: windows past the source origin fail closed on
        # rebased timestamps; no visual candidate is grounded in them.
        src = make_source_file(tmp_path)
        result = run_av({"words": []}, 240.0, src, _rebased,
                        one_visual_per_window,
                        av_config=make_av_config(
                            media_window_target_s=120.0,
                            media_window_overlap_s=30.0))
        avf = av_funnel(result)
        for wr in avf["window_results"]:
            if wr["start_s"] > 0:
                assert wr["sampling_status"] == "failed"
            for f in wr["frames"]:
                assert wr["start_s"] <= f["decoded_ts"] <= wr["end_s"]
        assert any(d["reason"] == "decoded_timestamp_out_of_window"
                   for d in avf["fallback_details"])
        # Every accepted visual candidate references frames whose decoded
        # timestamps sit inside the candidate's own window.
        for v in store_by_basis(result, "visual"):
            for fid in v["visual_frame_ids"]:
                holder = next(wr for wr in avf["window_results"]
                              if any(f["frame_id"] == fid
                                     for f in wr["frames"]))
                f = next(f for f in holder["frames"]
                         if f["frame_id"] == fid)
                assert holder["start_s"] <= f["decoded_ts"] \
                    <= holder["end_s"]

    def test_gate_c_unknown_frame_id_rejected(self, tmp_path, clean_av_env,
                                              monkeypatch):
        def plan(prompt, fids):
            if not fids:
                return []
            bad = visual_candidate(prompt, fids, score=95.0)
            bad["visual_frame_ids"] = ["vf_99_9"]
            return [bad]

        def run():
            src = make_source_file(tmp_path, name="gate_c.mp4")
            return run_av({"words": []}, 240.0, src, FakeSampler(), plan)

        # Correct behavior: rejected with the exact reason, never stored.
        result = run()
        funnel = result["candidate_discovery"]
        assert funnel["invalid_reasons"].get(
            "unknown_frame_id:vf_99_9", 0) >= 1
        assert store_by_basis(result, "visual") == []
        assert result["shorts"] == []

        # Mutation: a validator that silently accepts unknown frame IDs.
        real_validate = avd.validate_av_evidence

        def _accepting_unknown(item, **kw):
            item = dict(item)
            refs = [r for r in (item.get("visual_frame_ids") or [])
                    if not r.startswith("vf_99")]
            if not refs and item.get("visual_frame_ids"):
                refs = ["vf_0_0"]  # pretend the foreign ID was supplied
            item["visual_frame_ids"] = refs
            return real_validate(item, **kw)

        monkeypatch.setattr(avd, "validate_av_evidence",
                            _accepting_unknown)
        with pytest.raises(AssertionError):
            mutated = run()
            mfunnel = mutated["candidate_discovery"]
            assert mfunnel["invalid_reasons"].get(
                "unknown_frame_id:vf_99_9", 0) >= 1

    def test_gate_d_overlap_never_dedupes(self, tmp_path, clean_av_env,
                                          monkeypatch):
        def plan(prompt, fids):
            if not fids:
                return []
            a = visual_candidate(prompt, fids, score=80.0,
                                 desc="a person raises both arms",
                                 n_refs=1, start_off=2.0, dur=20.0)
            b = visual_candidate(prompt, fids[1:], score=78.0,
                                 desc="the crowd waves flags wildly",
                                 n_refs=1, start_off=10.0, dur=20.0)
            return [a, b]

        def run_and_assert_both_survive():
            src = make_source_file(tmp_path, name="gate_d.mp4")
            result = run_av({"words": []}, 240.0, src, FakeSampler(),
                            plan)
            # Post-dedupe selection (shorts), not the store: the store
            # keeps every valid candidate including deduped ones.
            short_ids = {s["candidate_id"] for s in result["shorts"]}
            descs = [e["visual_description"]["text"]
                     for e in store_by_basis(result, "visual")
                     if e["candidate_id"] in short_ids]
            assert "a person raises both arms" in descs
            assert "the crowd waves flags wildly" in descs

        run_and_assert_both_survive()

        # Mutation: dedupe keyed on time overlap alone.
        def _overlap_dedupe(candidates):
            kept, removed, groups = [], {}, []
            for cand in sorted(
                    candidates,
                    key=lambda c: (-c.recommendation_score,
                                   c.candidate_id)):
                dup_of = next(
                    (k.candidate_id for k in kept
                     if cand.start_s < k.end_s
                     and k.start_s < cand.end_s), None)
                if dup_of:
                    removed[cand.candidate_id] = {
                        "kept_id": dup_of, "basis": "overlap",
                        "candidate_id": cand.candidate_id}
                else:
                    kept.append(cand)
                    groups.append({"kept_id": cand.candidate_id,
                                   "removed_ids": [], "basis": None})
            return {"kept": kept, "removed": removed, "groups": groups}

        monkeypatch.setattr(cd, "dedupe_candidates", _overlap_dedupe)
        with pytest.raises(AssertionError):
            run_and_assert_both_survive()

    def test_gate_e_wordless_candidate_survives_refinement(
            self, tmp_path, clean_av_env, monkeypatch):
        def run():
            src = make_source_file(tmp_path, name="gate_e.mp4")
            # Editorial enabled (default): the speech-free candidate must
            # be classified up front, before any provider/client use.
            return run_av({"words": []}, 240.0, src, FakeSampler(),
                          one_visual_per_window)

        result = run()
        plans = {p["candidate_id"]: p
                 for p in result["editorial_plans"]}
        assert plans, "expected editorial plans"
        for p in plans.values():
            assert p["verdict"] == "unrefined"
            assert p["reason"] == "speech_free_av_no_transcript_evidence"

        # Mutation: Task 09 no longer recognizes speech-free AV
        # candidates (drops them into provider refinement, which has no
        # client here) -- the exact unrefined reason disappears.
        monkeypatch.setattr(er, "_is_speech_free_av_candidate",
                            lambda cand, words: False)
        with pytest.raises(AssertionError):
            mutated = run()
            mplans = {p["candidate_id"]: p
                      for p in mutated["editorial_plans"]}
            assert mplans
            for p in mplans.values():
                assert (p["verdict"] == "unrefined"
                        and p["reason"]
                        == "speech_free_av_no_transcript_evidence")

    def test_gate_f_caps_apply_to_av_candidates(self, tmp_path,
                                                clean_av_env):
        # Bypassing final caps would keep every candidate: the cap test
        # asserts exact removal counts, the removed IDs, and full funnel
        # reconciliation -- any bypass fails it.
        src = make_source_file(tmp_path)
        scores = {0: 90.0, 1: 80.0, 2: 70.0, 3: 60.0}

        def plan(prompt, fids):
            if not fids:
                return []
            widx = int(fids[0].split("_")[1])
            return [visual_candidate(prompt, fids,
                                     score=scores[widx])]

        result = run_av({"words": []}, 360.0, src, FakeSampler(), plan,
                        max_clips=2)
        funnel = result["candidate_discovery"]
        assert funnel["post_dedupe_candidates"] == 4
        assert funnel["final_cap_removals"] == 2
        assert funnel["final_selected"] == 2
        kept_scores = sorted((s["viral_score"] for s in result["shorts"]),
                             reverse=True)
        assert kept_scores == [90, 80]
        assert funnel["counts_reconcile"] is True
        assert av_funnel(result)["counts_reconcile"] is True

    def test_gate_g_stale_source_fingerprint_forces_rerun(
            self, tmp_path, clean_av_env, monkeypatch):
        src = make_source_file(tmp_path, name="gate_g.mp4")
        sampler = FakeSampler()
        acfg, dcfg = make_av_config(), make_discovery_config()
        transcript = {"words": []}
        result = run_av(transcript, 240.0, src, sampler,
                        one_visual_per_window,
                        av_config=acfg, discovery_config=dcfg)
        src2 = make_source_file(tmp_path, name="gate_g2.mp4", seed=b"W")

        def check(path):
            return cd.av_resume_reusable(
                result["candidate_discovery"], source_path=path,
                source_duration=240.0, transcript_result=transcript,
                av_config=acfg, discovery_config=dcfg)

        # Correct behavior: stale fingerprint -> re-run, never reuse.
        assert check(src2) == (
            False, "av_identity_mismatch:source_fingerprint")
        # Mutation: the fingerprint always reports the stale value, so
        # the stale record is silently reused.
        stale = avd.fingerprint_source(src)
        monkeypatch.setattr(avd, "fingerprint_source",
                            lambda p, **kw: stale)
        ok, why = check(src2)
        assert ok is True, \
            f"mutation not demonstrated: got {(ok, why)}"

    def test_gate_h_av_metadata_survives_fallback(self, tmp_path,
                                                  clean_av_env,
                                                  monkeypatch):
        # A fallback that lost the AV metadata would fail these: the
        # queued fallback path must persist media identity, window
        # accounting, and the candidate store.
        monkeypatch.setattr(
            FakeGenaiClient, "produce",
            staticmethod(lambda contents: json.dumps({"candidates": []})))
        monkeypatch.setattr(main_module.genai, "Client", FakeGenaiClient)
        sampler = FakeSampler()
        monkeypatch.setattr(avd, "ffmpeg_frame_sampler", sampler)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("CLIPPYME_SILENCE_SNAP", "0")
        src = make_source_file(tmp_path, name="gate_h.mp4")
        out_dir = tmp_path / "gate_h_out"
        out_dir.mkdir()
        args = SimpleNamespace(
            skip_analysis=False, instructions=None,
            min_duration=10.0, max_duration=60.0,
            min_clips=None, max_clips=None, clip_type=None,
            duration_mode=None, aspect="9:16", monitor=False,
            reframe_mode="auto")
        state = RuntimeState(str(out_dir), job_id="gate-h")
        clips_data, _ = orch._load_or_analyze(
            args, src, "v", str(out_dir), 240.0, {"words": []}, state,
            main_module)
        cdf = clips_data["candidate_discovery"]
        assert cdf["av_discovery"]["enabled"] is True
        assert cdf["av_discovery"]["identity"]["source_fingerprint"] == \
            avd.fingerprint_source(src)
        assert "candidate_store" in clips_data
