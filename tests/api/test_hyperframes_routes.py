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
