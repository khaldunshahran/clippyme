"""Tests for editorial benchmark foundation (Task 07, rev3).

Covers review2 five groups.
"""
import json
import subprocess
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from clippyme.pipeline import editorial_benchmark as eb


@pytest.fixture
def fx(tmp_path):
    d = tmp_path / "eb"
    d.mkdir()
    assets = d / "assets"
    assets.mkdir()
    for name in ("source.mp4", "clip.mp4"):
        p = assets / name
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=blue:s=160x90:r=10:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
            check=True)
    manifest = {
        "version": 1, "asset_root": "assets",
        "sources": [
            {"id": "s", "file": "source.mp4", "duration": 60.0,
             "genre": eb.GENRES[0], "synthetic": True},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc", "settings": {},
             "source_ids": ["s"],
             "clips": [
                 {"clip_id": "c", "source_id": "s",
                  "start": 0.0, "end": 10.0,
                  "file": "clip.mp4", "moment_id": "m"},
             ]},
        ],
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def _load(d):
    return eb.load_manifest(d / "manifest.json")


# Group 1: Media QA.
def test_media_qa_real_signature(fx):
    m = _load(fx)
    report = eb.run_benchmark(m, {}, check_media=True)
    errors = report["media_checks"]["source:s"]["errors"]
    assert not any("required keyword-only" in x for x in errors), errors


# Group 2: Output collisions.
def test_output_cannot_replace_reviews(fx):
    m = _load(fx)
    reviews = fx / "reviews.json"
    reviews.write_text('{"reviews":[]}')
    before = reviews.read_bytes()
    rc = eb.main(["--manifest", str(fx / "manifest.json"),
                  "--reviews", str(reviews),
                  "--json-out", str(reviews)])
    assert rc == 1
    assert reviews.read_bytes() == before


# Group 3: Frozen identities.
def test_run_fingerprint_mismatch_rejected(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["runs"][0]["source_fingerprints"] = {"s": "sha256:old"}
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_clip_fingerprint_preserved(fx):
    m = _load(fx)
    stats = eb.run_benchmark(m, {})
    assert m["runs"][0]["clips"][0]["fingerprint"] in eb._strict_json(stats)


# Group 4: Denominators.
def test_zero_candidate_source_counts(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"].append({**data["sources"][0], "id": "s2"})
    data["runs"][0]["source_ids"].append("s2")
    p.write_text(json.dumps(data))
    stats = eb.run_benchmark(_load(fx), {"c": {"accepted": True}})
    # 2 sources x 60s = 120s = 0.033h; 1 accepted -> 30/hour.
    assert stats["run_stats"][0]["accepted_per_source_hour"] == 30


def test_partial_labels_genre_unknown(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["runs"][0]["clips"].append(
        {**data["runs"][0]["clips"][0], "clip_id": "c2", "moment_id": None})
    p.write_text(json.dumps(data))
    stats = eb.run_benchmark(
        _load(fx), {"c": {"accepted": True}, "c2": {"accepted": True}})
    assert stats["genre_stats"][eb.GENRES[0]][
        "distinct_accepted_moments"] is None


def test_obstruction_framing_coverage(fx):
    stats = eb.run_benchmark(
        _load(fx),
        {"c": {"accepted": None, "hook_obstruction": True,
               "caption_obstruction": False}})
    j = stats["run_stats"][0]["judgments"]
    assert "hook_obstruction" in j and "caption_obstruction" in j
    assert "framing_known" in j


# Group 5: Strict validation.
def test_sources_none_clear_error(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"] = None
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_oversized_number_clear_error(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"][0]["duration"] = 10 ** 1000
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_json_boolean_string_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text('{"reviews":[{"clip_id":"c","accepted":"true"}]}')
    with pytest.raises(eb.BenchmarkError):
        eb.load_reviews(p)


def test_synthetic_string_rejected(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"][0]["synthetic"] = "false"
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)
