"""Tests for editorial benchmark (Task 07, rev4)."""
import json
import subprocess
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from clippyme.pipeline import editorial_benchmark as eb


def _fp(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


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
    sfp = _fp(assets / "source.mp4")
    manifest = {
        "version": 1, "asset_root": "assets",
        "sources": [
            {"id": "s", "file": "source.mp4", "duration": 60.0,
             "genre": eb.GENRES[0], "synthetic": True},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc", "settings": {},
             "source_ids": ["s"],
             "source_fingerprints": {"s": sfp},
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


def test_media_qa_real_signature(fx):
    m = _load(fx)
    report = eb.run_benchmark(m, {}, check_media=True)
    errors = report["media_checks"]["source:s"]["errors"]
    assert not any("required keyword-only" in x for x in errors), errors


def test_output_cannot_replace_reviews(fx):
    _load(fx)
    reviews = fx / "reviews.json"
    reviews.write_text('{"reviews":[]}')
    before = reviews.read_bytes()
    rc = eb.main(["--manifest", str(fx / "manifest.json"),
                  "--reviews", str(reviews),
                  "--json-out", str(reviews)])
    assert rc == 1
    assert reviews.read_bytes() == before


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


def test_zero_candidate_source_counts(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["sources"].append({**data["sources"][0], "id": "s2"})
    s2fp = _fp(fx / "assets" / "source.mp4")
    data["runs"][0]["source_ids"].append("s2")
    data["runs"][0]["source_fingerprints"]["s2"] = s2fp
    p.write_text(json.dumps(data))
    stats = eb.run_benchmark(_load(fx), {"c": {"accepted": True}})
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


# Review3 contracts.
def test_media_result_uses_real_schema(monkeypatch):
    from clippyme.pipeline import media_qa
    monkeypatch.setattr(
        media_qa, "inspect_clip",
        lambda *a, **k: {"ok": True, "critical": False, "issues": [],
                         "warnings": [],
                         "metrics": {"duration": 10., "width": 90,
                                     "height": 160}})
    result = eb._media_check("x.mp4", 10., 9/16)
    assert result["ok"] is True
    assert result["critical"] is False
    assert result["errors"] == []


def test_missing_frozen_identity_rejected(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    del data["runs"][0]["source_fingerprints"]
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)


def test_duplicate_source_membership_rejected(fx):
    p = fx / "manifest.json"
    data = json.loads(p.read_text())
    data["runs"][0]["source_ids"] = ["s", "s"]
    sfp = data["runs"][0]["source_fingerprints"]["s"]
    data["runs"][0]["source_fingerprints"] = {"s": sfp}
    p.write_text(json.dumps(data))
    with pytest.raises(eb.BenchmarkError):
        eb.load_manifest(p)

def test_generate_demo_clean_checkout(tmp_path):
    """Clean-checkout: generator in isolated dir, verify, run CLI."""
    import shutil
    import subprocess
    src_gen = (Path(__file__).parent.parent.parent / "tests" / "fixtures"
               / "editorial_benchmark" / "generate_demo.py")
    iso = tmp_path / "iso"
    iso.mkdir()
    shutil.copy(src_gen, iso / "generate_demo.py")
    subprocess.run(
        [sys.executable, str(iso / "generate_demo.py"), str(iso)],
        check=True, cwd=str(tmp_path))
    manifest_path = iso / "manifest.json"
    assert manifest_path.exists()
    m = eb.load_manifest(manifest_path)
    for sid, s in m["sources"].items():
        result = eb._media_check(s["_path"], s["duration"], None)
        assert result["invocation_error"] is None
    reviews_path = iso / "reviews.json"
    json_out = tmp_path / "out.json"
    text_out = tmp_path / "out.txt"
    rc = eb.main([
        "--manifest", str(manifest_path),
        "--reviews", str(reviews_path),
        "--check-media",
        "--json-out", str(json_out),
        "--text-out", str(text_out),
    ])
    assert rc == 0
    data = json.loads(json_out.read_text())
    assert data["version"] == eb.MANIFEST_VERSION
    assert "media_checks" in data
    assert any(v["ok"] for v in data["media_checks"].values())
    text = text_out.read_text()
    assert "baseline" in text and "candidate" in text
    # Rerun deterministically.
    subprocess.run(
        [sys.executable, str(iso / "generate_demo.py"), str(iso)],
        check=True, cwd=str(tmp_path))
    m2 = eb.load_manifest(manifest_path)
    assert (m2["sources"]["syn1"]["fingerprint"]
            == m["sources"]["syn1"]["fingerprint"])
