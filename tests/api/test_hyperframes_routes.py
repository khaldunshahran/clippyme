"""Task 18 — hyperframes API gates (host-runnable, no heavy deps).

Covers contracts 6/7/8 at the route level with the router mounted
standalone (auth dependency overridden):
- strict input validation (template/params/version),
- stale version -> 409, renderer-off -> honest 400,
- asset containment (traversal/symlink/unknown asset rejected),
- render record binds job/clip/version/digest (contract 8),
- cache key binds every input; no reuse across changed inputs,
- worker-busy maps to a failed state, not a silent success.
"""
import json
import os
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

OUTPUT_DIR = None  # set per-test via env before router import


def _make_client(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    import importlib
    import clippyme.api.hyperframes_routes as hr
    importlib.reload(hr)
    from clippyme.api.auth import AuthUser, get_current_user
    app = FastAPI()
    app.include_router(hr.router)
    app.dependency_overrides[get_current_user] = lambda: AuthUser(
        id="admin", email="a@x", role="admin", is_admin=True)
    return TestClient(app, raise_server_exceptions=False), hr, str(out)


def _job_id():
    return str(uuid.uuid4())


def _write_job(out, job_id, project):
    job_dir = os.path.join(out, job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, ".clippyme_runtime.json"), "w") as f:
        json.dump({"user_id": "someone"}, f)
    # versioned project files as written by save_project_version
    with open(os.path.join(job_dir, "clip-project-0.json"), "w") as f:
        json.dump(project, f)
    with open(os.path.join(
            job_dir, f"clip-project-0.v{project['version']}.json"), "w") as f:
        json.dump(project, f)
    with open(os.path.join(job_dir, "project-versions.json"), "w") as f:
        json.dump({"0": {"latest": project["version"],
                         "versions": {str(project["version"]): {
                             "version": project["version"],
                             "origin": "user",
                             "project_file": f"clip-project-0.v{project['version']}.json"}}}}, f)
    return job_dir


def _project(finishing="hyperframes"):
    return {
        "schema": "nugget.clip-project/1",
        "version": 2,
        "origin": "user",
        "job_id": "x",
        "clip_index": 0,
        "idempotency_key": "k",
        "source": {"file": "source_0.mp4", "width": 608, "height": 1080},
        "segments": [{"id": "seg0", "start": 0, "end": 5,
                      "crop": {"x": 0, "y": 0, "w": 608, "h": 1080}}],
        "captions": {"words": [{"id": "w0", "w": "hi", "start": 0, "end": 1}]},
        "render": {"width": 608, "height": 1080, "fps": 30},
        "finishing": ({"renderer": finishing, "template": "clean-captions"}
                      if finishing else None),
    }


# --- capability ---------------------------------------------------------------

def test_capability_shape(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    r = client.get("/api/hyperframes/capability")
    assert r.status_code == 200
    body = r.json()
    assert body["renderer_default"] == "legacy"
    assert {t["id"] for t in body["templates"]} == {
        "clean-captions", "emphasized-word", "timed-opening"}
    assert body["renderer"]["engine"] == "0.8.145"


def test_invalid_job_id_rejected(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    r = client.post("/api/hyperframes/compile/not-a-uuid/0",
                    json={"template_id": "clean-captions",
                          "params": {}, "expected_version": 1})
    assert r.status_code == 400


# --- compile: version + renderer gates ------------------------------------------

def test_compile_renderer_off_honest_400(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project(finishing=None))
    r = client.post(f"/api/hyperframes/compile/{jid}/0",
                    json={"template_id": "clean-captions",
                          "params": {}, "expected_version": 2})
    assert r.status_code == 400
    assert "not enabled" in r.json()["detail"]


def test_compile_stale_version_409(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project())
    r = client.post(f"/api/hyperframes/compile/{jid}/0",
                    json={"template_id": "clean-captions",
                          "params": {}, "expected_version": 1})
    assert r.status_code == 409
    assert "Stale project version" in r.json()["detail"]


def test_compile_invalid_template_400(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project())
    r = client.post(f"/api/hyperframes/compile/{jid}/0",
                    json={"template_id": "cinematic-zoom",
                          "params": {}, "expected_version": 2})
    assert r.status_code == 400


def test_compile_unknown_param_400(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project())
    r = client.post(f"/api/hyperframes/compile/{jid}/0",
                    json={"template_id": "clean-captions",
                          "params": {"zoom_speed": 9}, "expected_version": 2})
    assert r.status_code == 400


# --- asset containment ------------------------------------------------------------

def test_composition_asset_traversal_rejected(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    os.makedirs(os.path.join(job_dir, ".hf", "abcdef0123456789"), exist_ok=True)
    for bad in ["..%2f..%2fclip_project.py", "..\\..\\x", "manifest.json",
                "nope.mp4", ""]:
        r = client.get(
            f"/api/hyperframes/composition/{jid}/abcdef0123456789/{bad}")
        assert r.status_code in (400, 404), bad


def test_composition_symlink_refused(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    comp = os.path.join(job_dir, ".hf", "abcdef0123456789")
    os.makedirs(comp, exist_ok=True)
    target = tmp_path / "secret.mp4"
    target.write_bytes(b"x")
    link = os.path.join(comp, "base.mp4")
    try:
        os.symlink(str(target), link)
    except OSError:
        pytest.skip("symlinks unavailable")
    r = client.get(
        f"/api/hyperframes/composition/{jid}/abcdef0123456789/base.mp4")
    assert r.status_code == 400


def test_composition_allowlist_blocks_unlisted_real_file(tmp_path, monkeypatch):
    """The asset allowlist is load-bearing (M6): a real file inside the comp
    dir that is NOT allowlisted must be refused with 404, even though the
    path-containment check alone would pass. Without the allowlist,
    manifest.json would be served."""
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    comp = os.path.join(job_dir, ".hf", "abcdef0123456789")
    os.makedirs(comp, exist_ok=True)
    with open(os.path.join(comp, "manifest.json"), "w") as f:
        f.write("{}")
    r = client.get(
        f"/api/hyperframes/composition/{jid}/abcdef0123456789/manifest.json")
    assert r.status_code == 404


# --- cache key binding --------------------------------------------------------------

def test_cache_key_changes_with_every_input(tmp_path, monkeypatch):
    from clippyme.finishing import cache as _cache
    kw = dict(base_sha256="b" * 64, project_digest="p" * 64, project_version=2,
              template_id="clean-captions", template_version=1,
              params={"caption_size": "medium"}, font_shas=["f" * 64],
              renderer={"engine": "0.8.145"}, width=608, height=1080, fps=30)
    k0 = _cache.cache_key(**kw)
    for mutate in [
        {"base_sha256": "c" * 64},
        {"project_digest": "q" * 64},
        {"project_version": 3},
        {"template_id": "emphasized-word"},
        {"params": {"caption_size": "large"}},
        {"font_shas": ["e" * 64]},
        {"renderer": {"engine": "0.8.146"}},
        {"width": 1080},
    ]:
        k = _cache.cache_key(**{**kw, **mutate})
        assert k != k0, mutate
    assert _cache.cache_key(**kw) == k0  # deterministic


def test_cache_lookup_miss_and_store(tmp_path, monkeypatch):
    from clippyme.finishing import cache as _cache
    job_dir = str(tmp_path / "job")
    os.makedirs(job_dir, exist_ok=True)
    assert _cache.lookup(job_dir, "k" * 64) is None
    src = tmp_path / "out.mp4"
    src.write_bytes(b"fake-mp4")
    dest = _cache.store(job_dir, "k" * 64, str(src))
    assert _cache.lookup(job_dir, "k" * 64) == dest


# --- render record identity (contract 8) --------------------------------------------

def test_render_record_binds_identity(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    render_id = f"{jid}_{'ab12cd34ef56ab78'}"
    rec = {"render_id": render_id, "job_id": jid, "clip_index": 0,
           "project_version": 2, "preview_identity": "digest-1",
           "template_id": "clean-captions", "state": "rendered",
           "output_url": f"/videos/{jid}/hf_clip_0_v2_clean-captions.mp4"}
    hr._write_record(job_dir, render_id, rec)
    r = client.get(f"/api/hyperframes/render/{render_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["job_id"] == jid
    assert body["clip_index"] == 0
    assert body["project_version"] == 2
    assert body["preview_identity"] == "digest-1"


def test_render_status_unknown_404(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project())
    r = client.get(f"/api/hyperframes/render/{jid}_{'ab12cd34ef56ab78'}")
    assert r.status_code == 404


def test_render_id_format_rejected(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    r = client.get("/api/hyperframes/render/not-a-render-id")
    assert r.status_code == 400


# --- worker failure honesty -------------------------------------------------------

def test_worker_busy_surfaces_as_failed_state(tmp_path, monkeypatch):
    """A busy worker must surface honestly, never as a silent success."""
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    render_id = f"{jid}_{'ab12cd34ef56ab78'}"
    hr._write_record(job_dir, render_id, {
        "render_id": render_id, "job_id": jid, "clip_index": 0,
        "project_version": 2, "state": "failed",
        "error": "worker busy: another render job holds the worker lock (pid 1)"})
    r = client.get(f"/api/hyperframes/render/{render_id}")
    assert r.json()["state"] == "failed"
    assert "busy" in r.json()["error"]


# --- preview tokens (review-01 group 2 backend) ------------------------------------

import asyncio as _asyncio
import shutil as _shutil
import subprocess as _subprocess
from pathlib import Path as _Path


def _ffmpeg_available():
    return bool(_shutil.which("ffmpeg") and _shutil.which("ffprobe"))


def _tiny_mp4(path, w=608, h=1080, fps=30, dur=1.0):
    if not _ffmpeg_available():
        pytest.skip("ffmpeg/ffprobe not on PATH")
    _subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True)
    return str(path)


def _valid_compile_record(hr, job_dir, jid, cid, *,
                          width=608, height=1080, fps=30):
    """Write a compile record + manifest matching the CURRENT saved project
    (real digest via the compiler's digest function), like the compile
    endpoint would."""
    from clippyme.finishing import compile as _c
    project, version = hr._load_project_at_version(job_dir, 0, None)
    comp = os.path.join(job_dir, ".hf", cid)
    os.makedirs(comp, exist_ok=True)
    manifest = {
        "project_digest": _c.project_digest(project),
        "template": {"id": "clean-captions", "version": 1},
        "params": {"caption_size": "medium", "caption_position": "lower-third"},
        "base": {"sha256": "a" * 64},
        "fonts": [],
        "renderer": {"engine": "0.8.145"},
        "render": {"width": width, "height": height, "fps": fps},
        "duration": 5.0,
        "frames": 150,
    }
    with open(os.path.join(comp, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    crec_dir = os.path.join(job_dir, ".hf", "compiles")
    os.makedirs(crec_dir, exist_ok=True)
    rec = {"compile_id": cid, "job_id": jid, "clip_index": 0,
           "project_version": version,
           "project_digest": manifest["project_digest"],
           "preview_identity": "digest-" + cid[:8],
           "template": manifest["template"],
           "created_at": 0.0}
    with open(os.path.join(crec_dir, cid + ".json"), "w") as f:
        json.dump(rec, f)
    return rec, manifest


def _write_token_compile(hr, out, jid, cid):
    job_dir = _write_job(out, jid, _project())
    _valid_compile_record(hr, job_dir, jid, cid)
    with open(os.path.join(job_dir, ".hf", cid, "index.html"), "w") as f:
        f.write("<html>preview</html>")
    return job_dir


def _mint_token(client, jid, cid):
    r = client.post(f"/api/hyperframes/preview-token/{jid}/{cid}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["expires_in"] == 900
    assert isinstance(body["token"], str) and "." in body["token"]
    return body["token"]


def test_preview_token_mint_and_asset_access(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    _write_token_compile(hr, out, jid, cid)
    token = _mint_token(client, jid, cid)
    # The token works with NO session auth at all (overrides cleared).
    client.app.dependency_overrides.clear()
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/index.html?token={token}")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/html")


def test_preview_token_bad_signature_rejected(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    _write_token_compile(hr, out, jid, cid)
    token = _mint_token(client, jid, cid)
    bad = token[:-2] + ("AA" if not token.endswith("AA") else "BB")
    client.app.dependency_overrides.clear()
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/index.html?token={bad}")
    assert r.status_code == 401


def test_preview_token_expired_rejected(tmp_path, monkeypatch):
    from clippyme.finishing import preview_tokens as _pt
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    _write_token_compile(hr, out, jid, cid)
    token, _ = _pt.mint_preview_token(jid, cid, ttl_sec=-5)
    client.app.dependency_overrides.clear()
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/index.html?token={token}")
    assert r.status_code == 401
    assert "expired" in r.json()["detail"]


def test_preview_token_wrong_scope_rejected(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    _write_token_compile(hr, out, jid, cid)
    token = _mint_token(client, jid, cid)
    client.app.dependency_overrides.clear()
    # Token scoped to jid, used on another job's path.
    other = _job_id()
    _write_job(out, other, _project())
    r = client.get(
        f"/api/hyperframes/composition/{other}/{cid}/index.html?token={token}")
    assert r.status_code == 401
    # Token scoped to cid, used on another compile's path.
    cid2 = "1234567890abcdef"
    _valid_compile_record(hr, os.path.join(out, jid), jid, cid2)
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid2}/index.html?token={token}")
    assert r.status_code == 401


def test_preview_token_unknown_compile_rejected(tmp_path, monkeypatch):
    from clippyme.finishing import preview_tokens as _pt
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    _write_job(out, jid, _project())
    token, _ = _pt.mint_preview_token(jid, "ffffffffffffffff")
    client.app.dependency_overrides.clear()
    r = client.get(
        f"/api/hyperframes/composition/{jid}/ffffffffffffffff/index.html"
        f"?token={token}")
    assert r.status_code == 404


def test_preview_token_mint_requires_auth(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    _write_token_compile(hr, out, jid, cid)
    client.app.dependency_overrides.clear()
    monkeypatch.setenv("AUTH_DISABLED_DEV", "0")
    r = client.post(f"/api/hyperframes/preview-token/{jid}/{cid}")
    assert r.status_code == 401


def test_preview_token_respects_asset_allowlist(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    job_dir = _write_token_compile(hr, out, jid, cid)
    with open(os.path.join(job_dir, ".hf", cid, "manifest.json"), "w") as f:
        f.write("{}")
    token = _mint_token(client, jid, cid)
    client.app.dependency_overrides.clear()
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/manifest.json?token={token}")
    assert r.status_code == 404


# --- compile/export snapshot binding (review-01 group 3 backend) --------------------

def test_render_rejects_stale_compile_after_new_save(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())  # v2
    cid = "abcdef0123456789"
    _valid_compile_record(hr, job_dir, jid, cid)
    # A newer save lands after the preview was compiled...
    p3 = _project()
    p3["version"] = 3
    _write_job(out, jid, p3)
    r = client.post(f"/api/hyperframes/render/{jid}/0",
                    json={"compile_id": cid})
    assert r.status_code == 409
    assert "stale" in r.json()["detail"].lower()
    assert "re-preview" in r.json()["detail"].lower()


def test_render_rejects_compile_digest_mismatch(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    cid = "abcdef0123456789"
    rec, _ = _valid_compile_record(hr, job_dir, jid, cid)
    # Same version, but the record digest no longer matches the saved project.
    rec["project_digest"] = "0" * 64
    with open(os.path.join(job_dir, ".hf", "compiles", cid + ".json"),
              "w") as f:
        json.dump(rec, f)
    r = client.post(f"/api/hyperframes/render/{jid}/0",
                    json={"compile_id": cid})
    assert r.status_code == 409
    assert "stale" in r.json()["detail"].lower()


def test_render_accepts_current_compile_cache_hit(tmp_path, monkeypatch):
    from clippyme.finishing import cache as _cache
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    cid = "abcdef0123456789"
    _, manifest = _valid_compile_record(hr, job_dir, jid, cid)
    w, h = manifest["render"]["width"], manifest["render"]["height"]
    cached = _tiny_mp4(tmp_path / "cached.mp4", w=w, h=h)
    monkeypatch.setattr(_cache, "lookup", lambda *a: cached)
    r = client.post(f"/api/hyperframes/render/{jid}/0",
                    json={"compile_id": cid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "rendered" and body["cache_hit"] is True
    final = os.path.join(job_dir, body["output_url"].rsplit("/", 1)[-1])
    assert os.path.isfile(final)
    from clippyme.domain import project_render as _pr
    probe = _pr.probe_media(final)
    assert probe["has_video"] is True
    assert probe["width"] == w and probe["height"] == h
    assert list(_Path(job_dir).glob("*.part-*")) == []


# --- atomic publication (review-01 group 4) -----------------------------------------

def test_cache_hit_publish_failure_preserves_previous_output(tmp_path,
                                                            monkeypatch):
    """Promoted coordinator-review01-backend.py regression — now with a VALID
    compile record (real digest) so the corruption path is actually reached:
    an injected copy failure must not replace a previous complete export
    with partial bytes, must clean up its own temp file, and must fail
    honestly with a failed record."""
    from clippyme.finishing import cache as _cache
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    cid = "abcdef0123456789"
    _valid_compile_record(hr, job_dir, jid, cid)
    final = os.path.join(job_dir, hr._output_name(0, 2, "clean-captions"))
    _Path(final).write_bytes(b"previous-complete-export")
    cached = tmp_path / "cache.mp4"
    cached.write_bytes(b"new-complete-export")
    monkeypatch.setattr(_cache, "lookup", lambda *a: str(cached))

    def interrupted_copy(src, dst, *a, **kw):
        _Path(dst).write_bytes(b"partial-new-export")
        raise OSError("injected disk write failure")

    monkeypatch.setattr(hr.shutil, "copy2", interrupted_copy)
    r = client.post(f"/api/hyperframes/render/{jid}/0",
                    json={"compile_id": cid})
    assert r.status_code == 500
    assert "previous output preserved" in r.json()["detail"]
    assert _Path(final).read_bytes() == b"previous-complete-export"
    assert list(_Path(job_dir).glob("*.part-*")) == []
    recs = list((_Path(job_dir) / ".hf" / "renders").glob("*.json"))
    assert len(recs) == 1
    rec = json.loads(recs[0].read_text())
    assert rec["state"] == "failed"
    assert "publish failed" in rec["error"]


def test_stage_and_install_rejects_bad_media(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"previous-complete-export")
    src = tmp_path / "src.mp4"
    src.write_bytes(b"not-a-video")
    with pytest.raises(hr.PublishValidationError, match="no video stream"):
        hr._stage_and_install(str(src), str(final), width=608, height=1080)
    assert final.read_bytes() == b"previous-complete-export"
    assert list(tmp_path.glob("*.part-*")) == []


def test_stage_and_install_copy_failure_cleans_up(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"previous-complete-export")
    src = tmp_path / "src.mp4"
    src.write_bytes(b"new-bytes")

    def interrupted_copy(s, d, *a, **kw):
        _Path(d).write_bytes(b"partial")
        raise OSError("injected disk write failure")

    monkeypatch.setattr(hr.shutil, "copy2", interrupted_copy)
    with pytest.raises(OSError, match="injected disk write failure"):
        hr._stage_and_install(str(src), str(final), width=608, height=1080)
    assert final.read_bytes() == b"previous-complete-export"
    assert list(tmp_path.glob("*.part-*")) == []


def test_stage_and_install_success(tmp_path, monkeypatch):
    client, hr, out = _make_client(tmp_path, monkeypatch)
    final = tmp_path / "final.mp4"
    final.write_bytes(b"previous-complete-export")
    src = _tiny_mp4(tmp_path / "src.mp4", w=608, h=1080)
    hr._stage_and_install(src, str(final), width=608, height=1080)
    from clippyme.domain import project_render as _pr
    probe = _pr.probe_media(str(final))
    assert probe["has_video"] is True
    assert probe["width"] == 608 and probe["height"] == 1080
    assert list(tmp_path.glob("*.part-*")) == []


def test_publish_worker_output_installs_when_rendering(tmp_path, monkeypatch):
    """The fresh-worker publish path: cache install + atomic output install."""
    from clippyme.finishing import cache as _cache
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    render_id = f"{jid}_{'ab12cd34ef56ab78'}"
    hr._write_record(job_dir, render_id, {
        "render_id": render_id, "job_id": jid, "clip_index": 0,
        "project_version": 2, "state": "rendering"})
    output_path = os.path.join(job_dir, "hf_clip_0_v2_clean-captions.mp4")
    src = _tiny_mp4(tmp_path / "worker-out.mp4", w=608, h=1080)
    _asyncio.run(hr._publish_worker_output(
        job_dir=job_dir, render_id=render_id, src_path=src,
        cache_key="k" * 64, output_path=output_path,
        width=608, height=1080))
    assert os.path.isfile(output_path)
    assert _cache.lookup(job_dir, "k" * 64) is not None
    assert list(_Path(job_dir).glob("*.part-*")) == []


def test_publish_worker_output_cancel_race_never_installs(tmp_path,
                                                         monkeypatch):
    """A cancelled-then-completed worker never installs output; the record
    stays cancelled with an honest note (surfaced by the status endpoint)."""
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    render_id = f"{jid}_{'ab12cd34ef56ab78'}"
    hr._write_record(job_dir, render_id, {
        "render_id": render_id, "job_id": jid, "clip_index": 0,
        "project_version": 2, "state": "cancelled",
        "error": "cancel requested", "finished_at": 0.0})
    output_path = os.path.join(job_dir, "hf_clip_0_v2_clean-captions.mp4")
    src = tmp_path / "worker-out.mp4"
    src.write_bytes(b"worker-bytes")
    with pytest.raises(hr.PublishSkipped):
        _asyncio.run(hr._publish_worker_output(
            job_dir=job_dir, render_id=render_id, src_path=str(src),
            cache_key="k" * 64, output_path=output_path,
            width=608, height=1080))
    assert not os.path.exists(output_path)
    rec = hr._read_record(job_dir, render_id)
    assert rec["state"] == "cancelled"
    assert "not published" in rec["publish_note"]
    r = client.get(f"/api/hyperframes/render/{render_id}")
    assert r.json()["state"] == "cancelled"
    assert "not published" in (r.json()["note"] or "")

def test_publish_worker_output_cancel_during_staging_preserves_prior(tmp_path,
                                                                     monkeypatch):
    """Review-02: cancellation DURING staging/install must not replace the
    prior export. The commit gate re-reads the record after the staging
    thread settles; if cancelled, the prior bytes are restored."""
    import threading
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    job_dir = _write_job(out, jid, _project())
    render_id = f"{jid}_{'ab12cd34ef56ab78'}"
    hr._write_record(job_dir, render_id, {
        "render_id": render_id, "job_id": jid, "clip_index": 0,
        "project_version": 2, "state": "rendering"})
    output_path = os.path.join(job_dir, "hf_clip_0_v2_clean-captions.mp4")
    with open(output_path, "wb") as f:
        f.write(b"previous")
    src = tmp_path / "worker-out.mp4"
    src.write_bytes(b"new-bytes")
    release = threading.Event()

    def fake_stage_and_install(src_path, output_path, width, height):
        # Simulate a slow staging: copy, then block until the test
        # marks the record cancelled (as the cancel route would).
        import shutil
        shutil.copy2(src_path, output_path)
        assert release.wait(timeout=10)
        import os as _os
        _os.replace(src_path, output_path)

    monkeypatch.setattr(hr, "_stage_and_install", fake_stage_and_install)
    from clippyme.finishing import cache as _cache_mod
    monkeypatch.setattr(_cache_mod, "store", lambda *a, **k: None)

    async def run():
        task = _asyncio.create_task(hr._publish_worker_output(
            job_dir=job_dir, render_id=render_id, src_path=str(src),
            cache_key="k" * 64, output_path=output_path,
            width=608, height=1080))
        await _asyncio.sleep(0.2)  # let staging start and block
        # Cancel route's durable write.
        hr._write_record(job_dir, render_id, {
            "render_id": render_id, "job_id": jid, "clip_index": 0,
            "project_version": 2, "state": "cancelled",
            "error": "cancel requested", "finished_at": 0.0})
        release.set()
        try:
            await task
            return "no-raise"
        except hr.PublishSkipped:
            return "skipped"
    assert _asyncio.run(run()) == "skipped"
    with open(output_path, "rb") as f:
        assert f.read() == b"previous"
    rec = hr._read_record(job_dir, render_id)
    assert rec["state"] == "cancelled"


def test_composition_html_injects_token_into_allowlisted_assets(tmp_path,
                                                               monkeypatch):
    """Review-02: index.html served with a valid token rewrites allowlisted
    relative asset refs (base.mp4, fonts) to carry the same grant, because
    relative resources do not inherit the navigation query."""
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    job_dir = _write_token_compile(hr, out, jid, cid)
    comp = Path(job_dir) / ".hf" / cid
    (comp / "index.html").write_text(
        '<video src="base.mp4"></video>'
        "<style>@font-face{src:url('Anton-Regular.ttf')}</style>",
        encoding="utf-8")
    (comp / "base.mp4").write_bytes(b"v")
    (comp / "Anton-Regular.ttf").write_bytes(b"f")
    token = _mint_token(client, jid, cid)
    client.app.dependency_overrides.clear()
    monkeypatch.setenv("AUTH_DISABLED_DEV", "0")
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/index.html?token={token}")
    assert r.status_code == 200
    assert 'src="base.mp4?token=' in r.text
    assert "url('Anton-Regular.ttf?token=" in r.text
    # Non-allowlisted refs are never rewritten.
    assert "evil.mp4?token=" not in r.text


def test_composition_font_served_with_cors_for_opaque_origin(tmp_path,
                                                            monkeypatch):
    """Review-02: fonts served for the sandboxed (opaque-origin) iframe
    carry Access-Control-Allow-Origin so they are not blocked."""
    client, hr, out = _make_client(tmp_path, monkeypatch)
    jid = _job_id()
    cid = "abcdef0123456789"
    job_dir = _write_token_compile(hr, out, jid, cid)
    comp = Path(job_dir) / ".hf" / cid
    (comp / "Anton-Regular.ttf").write_bytes(b"fake-font")
    token = _mint_token(client, jid, cid)
    client.app.dependency_overrides.clear()
    monkeypatch.setenv("AUTH_DISABLED_DEV", "0")
    r = client.get(
        f"/api/hyperframes/composition/{jid}/{cid}/Anton-Regular.ttf"
        f"?token={token}")
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "*"
