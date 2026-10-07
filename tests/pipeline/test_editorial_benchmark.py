"""Tests for editorial benchmark foundation (Task 07)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from clippyme.pipeline import editorial_benchmark as eb


@pytest.fixture
def fixture_dir(tmp_path):
    d = tmp_path / "eb"
    d.mkdir()
    # Tiny synthetic videos.
    for name in ("syn1.mp4", "syn2.mp4"):
        p = d / name
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=blue:s=160x90:r=10:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
            check=True)
    # Manifest.
    manifest = {
        "version": 1, "asset_root": ".",
        "sources": [
            {"id": "syn1", "file": "syn1.mp4", "duration": 60.0,
             "genre": "interview/podcast", "synthetic": True},
            {"id": "syn2", "file": "syn2.mp4", "duration": 60.0,
             "genre": "reaction/comedy", "synthetic": True},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc",
             "clips": [
                 {"clip_id": "c1", "source_id": "syn1",
                  "start": 10.0, "end": 25.0, "moment_id": "m1"},
                 {"clip_id": "c2", "source_id": "syn1",
                  "start": 30.0, "end": 45.0, "moment_id": "m1"},
             ]},
        ],
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    # Reviews.
    reviews = {"reviews": [
        {"clip_id": "c1", "accepted": True},
        {"clip_id": "c2", "accepted": False},
    ]}
    (d / "reviews.json").write_text(json.dumps(reviews))
    return d


def test_load_manifest_ok(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    assert m["version"] == 1
    assert len(m["sources"]) == 2
    assert len(m["runs"]) == 1


def test_manifest_rejects_traversal(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "version": 1, "asset_root": ".",
        "sources": [{"id": "s", "file": "../evil.mp4",
                     "duration": 10.0, "genre": "interview/podcast"}],
        "runs": []}))
    with pytest.raises(eb.BenchmarkError, match="traversal"):
        eb.load_manifest(d / "manifest.json")


def test_manifest_rejects_missing_asset(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "version": 1, "asset_root": ".",
        "sources": [{"id": "s", "file": "nope.mp4",
                     "duration": 10.0, "genre": "interview/podcast"}],
        "runs": []}))
    with pytest.raises(eb.BenchmarkError, match="missing"):
        eb.load_manifest(d / "manifest.json")


def test_manifest_rejects_duplicate_clip(fixture_dir):
    m_path = fixture_dir / "manifest.json"
    data = json.loads(m_path.read_text())
    data["runs"][0]["clips"].append(
        {"clip_id": "c1", "source_id": "syn1", "start": 0, "end": 5})
    m_path.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError, match="duplicate"):
        eb.load_manifest(m_path)


def test_reviews_duplicate_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"reviews": [
        {"clip_id": "c1", "accepted": True},
        {"clip_id": "c1", "accepted": False},
    ]}))
    with pytest.raises(eb.BenchmarkError, match="duplicate"):
        eb.load_reviews(p)


def test_unknown_stays_unknown(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    # No reviews for c2? Actually c2 has review. Let's test unreviewed.
    reviews = {"c1": {"accepted": True}}  # c2 unreviewed
    stats = eb.run_benchmark(m, reviews)
    rs = stats["run_stats"][0]
    # c1 reviewed/accepted, c2 unknown -> reviewed=1, accepted=1
    assert rs["reviewed_count"] == 1
    assert rs["accepted_count"] == 1


def test_distinct_moments_only_with_labels(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    reviews = {
        "c1": {"accepted": True},
        "c2": {"accepted": True},  # same moment m1
    }
    stats = eb.run_benchmark(m, reviews)
    rs = stats["run_stats"][0]
    # Both accepted, same moment m1 -> distinct=1
    assert rs["distinct_accepted_moments"] == 1


def test_denominator_correctness(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    reviews = eb.load_reviews(fixture_dir / "reviews.json")
    stats = eb.run_benchmark(m, reviews)
    rs = stats["run_stats"][0]
    # 2 candidates, 2 reviewed, 1 accepted -> rate 50%
    assert rs["candidate_count"] == 2
    assert rs["reviewed_count"] == 2
    assert rs["accepted_count"] == 1
    assert rs["acceptance_rate"] == 0.5


def test_strict_json_rejects_nan():
    import math
    with pytest.raises(eb.BenchmarkError, match="non-finite"):
        eb._strict_json({"x": math.nan})


def test_cli_end_to_end(fixture_dir, tmp_path):
    out_json = tmp_path / "out.json"
    out_txt = tmp_path / "out.txt"
    rc = eb.main([
        "--manifest", str(fixture_dir / "manifest.json"),
        "--reviews", str(fixture_dir / "reviews.json"),
        "--json-out", str(out_json),
        "--text-out", str(out_txt),
    ])
    assert rc == 0
    data = json.loads(out_json.read_text())
    assert "run_stats" in data
    assert out_txt.read_text().startswith("Editorial Benchmark Report")


def test_cli_malformed_fails(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid json")
    rc = eb.main([
        "--manifest", str(bad),
        "--reviews", str(bad),
    ])
    assert rc == 1
