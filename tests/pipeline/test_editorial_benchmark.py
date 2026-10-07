"""Tests for editorial benchmark foundation (Task 07, rev2)."""
import json
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
    for name in ("syn1.mp4", "syn2.mp4", "c1.mp4", "c2.mp4"):
        p = d / name
        subprocess.run([
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "color=blue:s=160x90:r=10:d=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)],
            check=True)
    manifest = {
        "version": 1, "asset_root": ".",
        "sources": [
            {"id": "syn1", "file": "syn1.mp4", "duration": 60.0,
             "genre": "interview/podcast", "synthetic": True},
            {"id": "syn2", "file": "syn2.mp4", "duration": 60.0,
             "genre": "reaction/comedy", "synthetic": True},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc", "settings": {},
             "clips": [
                 {"clip_id": "c1", "source_id": "syn1",
                  "start": 10.0, "end": 25.0, "moment_id": "m1",
                  "file": "c1.mp4"},
                 {"clip_id": "c2", "source_id": "syn1",
                  "start": 30.0, "end": 45.0, "moment_id": "m1",
                  "file": "c2.mp4"},
             ]},
        ],
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    reviews = {"reviews": [
        {"clip_id": "c1", "accepted": True},
        {"clip_id": "c2", "accepted": False},
    ]}
    (d / "reviews.json").write_text(json.dumps(reviews))
    return d


def test_load_manifest_ok(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    assert m["version"] == 1
    assert "sha256:" in m["sources"]["syn1"]["fingerprint"]


def test_sibling_prefix_escape_rejected(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    sibling = tmp_path / "assets_other"
    sibling.mkdir()
    (sibling / "x.mp4").write_bytes(b"outside")
    with pytest.raises(eb.BenchmarkError):
        eb._resolve_asset(root, "../assets_other/x.mp4")


def test_duplicate_run_ids_rejected(fixture_dir):
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    data["runs"].append({"run_id": "baseline", "clips": []})
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_bool_duration_rejected(fixture_dir):
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"][0]["duration"] = True
    data["runs"][0]["clips"] = []
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_missing_clip_asset_rejected(fixture_dir):
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    data["runs"][0]["clips"][0]["file"] = "missing.mp4"
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_malformed_root_rejected(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("[]")
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_invalid_boolean_rejected(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(
        {"reviews": [{"clip_id": "c", "accepted": "definitely"}]}))
    with pytest.raises(eb.BenchmarkError):
        eb.load_reviews(p)


def test_orphan_review_rejected(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    with pytest.raises(eb.BenchmarkError):
        eb.run_benchmark(m, {"typo": {"accepted": True}})


def test_partial_labels_no_distinct_claim(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    # c1 has moment m1, c2 has moment m1 but we remove c2's label.
    # Actually both have m1. Let's make c2 unlabeled.
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    del data["runs"][0]["clips"][1]["moment_id"]
    p.write_text(json.dumps(data))
    m = eb.load_manifest(p)
    stats = eb.run_benchmark(
        m, {"c1": {"accepted": True}, "c2": {"accepted": True}})
    # c1 labeled, c2 not -> partial -> None
    assert stats["run_stats"][0]["distinct_accepted_moments"] is None


def test_moment_ids_source_scoped(fixture_dir):
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    # Add second source with same moment ID.
    data["sources"].append({
        "id": "syn3", "file": "syn2.mp4", "duration": 60.0,
        "genre": "interview/podcast", "synthetic": True})
    data["runs"][0]["clips"].append({
        "clip_id": "c3", "source_id": "syn3",
        "start": 0.0, "end": 10.0, "moment_id": "m1", "file": "c1.mp4"})
    p.write_text(json.dumps(data))
    m = eb.load_manifest(p)
    stats = eb.run_benchmark(m, {
        "c1": {"accepted": True}, "c2": {"accepted": False},
        "c3": {"accepted": True}})
    # (syn1,m1) and (syn3,m1) are distinct -> 2
    assert stats["run_stats"][0]["distinct_accepted_moments"] == 2


def test_all_genres_visible(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    stats = eb.run_benchmark(m, {})
    assert set(stats["genre_stats"]) == set(eb.GENRES)


def test_per_run_hour_denominators(fixture_dir):
    p = fixture_dir / "manifest.json"
    data = json.loads(p.read_text())
    # Add candidate run with same source.
    data["runs"].append({
        "run_id": "candidate", "commit": "def",
        "clips": [{"clip_id": "c3", "source_id": "syn1",
                   "start": 0.0, "end": 10.0, "file": "c1.mp4"}]})
    p.write_text(json.dumps(data))
    m = eb.load_manifest(p)
    stats = eb.run_benchmark(m, {
        "c1": {"accepted": True}, "c2": {"accepted": False},
        "c3": {"accepted": True}})
    # Each run: 1 accepted, 60s source = 0.0167h -> 60/hour
    for rs in stats["run_stats"]:
        assert rs["accepted_per_source_hour"] == 60


def test_fingerprints_preserved(fixture_dir):
    m = eb.load_manifest(fixture_dir / "manifest.json")
    stats = eb.run_benchmark(m, {})
    assert "sha256:" in eb._strict_json(stats)


def test_strict_json_rejects_nan():
    import math
    with pytest.raises(eb.BenchmarkError):
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
    assert "sha256:" in out_json.read_text()


def test_cli_malformed_fails(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{invalid")
    assert eb.main(["--manifest", str(bad), "--reviews", str(bad)]) == 1


def test_output_collision_rejected(fixture_dir, tmp_path):
    # Output inside asset root should be rejected.
    out = fixture_dir / "report.json"
    rc = eb.main([
        "--manifest", str(fixture_dir / "manifest.json"),
        "--reviews", str(fixture_dir / "reviews.json"),
        "--json-out", str(out),
    ])
    assert rc == 1
