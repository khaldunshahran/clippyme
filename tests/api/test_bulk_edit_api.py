"""Task 19 -- bulk operation API gates (host-runnable).

Exercises the real FastAPI app with the auth dependency overridden:
owner/job/clip/patch validation, idempotency binding, the full
prepare -> apply -> status -> undo roundtrip, and reconcile-after-interrupt.
Rate limiting is disabled for the suite (RATE_LIMIT_ENABLED=0).
"""
import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient

import clippyme.api.app as app_module
from clippyme.api.auth import AuthUser, get_current_user

ORIGIN = {"Origin": "http://localhost:5175"}


def _job_id():
    return str(uuid.uuid4())


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "0")
    app_module.app.dependency_overrides[get_current_user] = lambda: AuthUser(
        id="owner-a", email="a@x", role="user", is_admin=False)
    yield TestClient(app_module.app, headers=ORIGIN)
    app_module.app.dependency_overrides.pop(get_current_user, None)


def _project(clip_index, version=1, style="classic_white", duration=20.0):
    return {
        "schema": "nugget.clip-project/1",
        "version": version,
        "origin": "auto",
        "job_id": "x",
        "clip_index": clip_index,
        "idempotency_key": f"seed-{clip_index}",
        "source": {"file": f"source_{clip_index}.mp4", "width": 1920,
                   "height": 1080, "duration": duration},
        "segments": [{"id": f"seg{clip_index}", "start": 0, "end": duration,
                      "crop": {"x": 0, "y": 0, "w": 608, "h": 1080}}],
        "captions": {"style": style, "position": "bottom",
                     "words": [{"id": "w0", "w": "hi", "start": 0, "end": 1}]},
        "overlays": [{"type": "hook", "text": f"hook {clip_index}",
                      "style": "viral", "start": 0.0, "end": 3.0,
                      "position": "top"}],
        "render": {"width": 608, "height": 1080, "fps": 30},
    }


def _make_job(tmp_path, job_id, projects, user_id="owner-a"):
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / ".clippyme_runtime.json").write_text(
        json.dumps({"user_id": user_id}))
    for ci, proj in projects.items():
        (job_dir / f"clip-project-{ci}.json").write_text(json.dumps(proj))
    return job_dir


def _prepare(client, job_id, body):
    return client.post(f"/api/project/{job_id}/bulk/prepare", json=body)


def _intent(style="hormozi_bold"):
    return {"op": "caption_appearance",
            "params": {"caption_style": style}}


# --- validation --------------------------------------------------------------

def test_prepare_rejects_invalid_job_id(client):
    r = _prepare(client, "not-a-uuid", {"scope": {}, "intent": {}})
    assert r.status_code == 400


def test_prepare_rejects_unowned_job(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0)}, user_id="someone-else")
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                         "clip_indices": [0]},
                               "intent": _intent()})
    assert r.status_code == 404


def test_prepare_rejects_bad_intent_and_scope(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0)})
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [0]},
                              "intent": {"op": "nope", "params": {}}})
    assert r.status_code == 400
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": []},
                              "intent": _intent()})
    assert r.status_code == 400
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [9]},
                              "intent": _intent()})
    assert r.status_code == 400  # clip 9 has no project


def test_prepare_freezes_scope_and_base_versions(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0, version=2),
                              1: _project(1, version=5, style="neon_glow")})
    r = _prepare(client, jid, {"scope": {"mode": "all"},
                              "intent": _intent(),
                              "idempotency_key": "api-key-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scope"] == {"mode": "all", "clip_indices": [0, 1],
                             "count": 2}
    assert body["base_versions"] == {"0": 2, "1": 5}
    assert body["idempotency_key"] == "api-key-1"
    assert body["status"] == "proposed"
    items = {i["clip_index"]: i for i in body["items"]}
    # Clip 1's neon_glow is an individual override: preserved by default.
    assert items[1]["preserved"][0]["field"] == "caption_style"
    assert items[0]["changes"][0]["to"] == "hormozi_bold"


def test_prepare_idempotent_and_key_binding(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0)})
    payload = {"scope": {"mode": "selected", "clip_indices": [0]},
               "intent": _intent(), "idempotency_key": "api-key-2"}
    r1 = _prepare(client, jid, payload)
    r2 = _prepare(client, jid, payload)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["operation_id"] == r2.json()["operation_id"]
    # Same key, different payload -> 409.
    bad = dict(payload, intent=_intent(style="neon_glow"))
    r3 = _prepare(client, jid, bad)
    assert r3.status_code == 409


# --- roundtrip -----------------------------------------------------------------

def test_full_roundtrip_apply_status_undo(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0, version=1),
                              1: _project(1, version=1)})
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [0, 1]},
                              "intent": _intent(),
                              "idempotency_key": "api-rt-1"})
    assert r.status_code == 200, r.text
    op_id = r.json()["operation_id"]

    r = client.post(f"/api/project/{jid}/bulk/apply",
                    json={"operation_id": op_id,
                          "idempotency_key": "api-rt-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "applied"
    items = {i["clip_index"]: i for i in body["items"]}
    assert items[0]["outcome"] == "applied"
    assert items[0]["result_version"] == 2
    assert "base_snapshot" not in items[0]  # internal, never exposed

    # Duplicate apply: no new versions.
    r = client.post(f"/api/project/{jid}/bulk/apply",
                    json={"operation_id": op_id,
                          "idempotency_key": "api-rt-1"})
    assert r.status_code == 200
    assert r.json()["items"][0]["result_version"] == 2

    # Reconcile: the ledger is the truth after an interrupted response.
    r = client.get(f"/api/project/{jid}/bulk/{op_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "applied"

    # Undo restores both clips as new revisions.
    r = client.post(f"/api/project/{jid}/bulk/{op_id}/undo",
                    json={"idempotency_key": "api-rt-1"})
    assert r.status_code == 200, r.text
    undo = r.json()["undo"]
    assert undo["status"] == "undone"
    assert undo["items"]["0"]["outcome"] == "undone"
    assert undo["items"]["0"]["version"] == 3

    # Repeated undo is idempotent.
    r = client.post(f"/api/project/{jid}/bulk/{op_id}/undo",
                    json={"idempotency_key": "api-rt-1"})
    assert r.status_code == 200
    assert r.json()["undo"]["items"]["0"]["version"] == 3


def test_apply_unknown_operation_and_key_mismatch(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0)})
    r = client.post(f"/api/project/{jid}/bulk/apply",
                    json={"operation_id": "nope",
                          "idempotency_key": "k"})
    assert r.status_code == 404
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [0]},
                              "intent": _intent(),
                              "idempotency_key": "api-key-3"})
    op_id = r.json()["operation_id"]
    r = client.post(f"/api/project/{jid}/bulk/apply",
                    json={"operation_id": op_id,
                          "idempotency_key": "wrong-key"})
    assert r.status_code == 409


def test_retry_endpoint_only_touches_unresolved(client, tmp_path):
    jid = _job_id()
    # Clip 1 has no hook overlay -> hook_timing excludes it (terminal).
    p1 = _project(1)
    p1["overlays"] = []
    _make_job(tmp_path, jid, {0: _project(0), 1: p1})
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [0, 1]},
                              "intent": {"op": "hook_timing",
                                         "params": {"end": 2.0}},
                              "idempotency_key": "api-retry-1"})
    assert r.status_code == 200, r.text
    op_id = r.json()["operation_id"]
    r = client.post(f"/api/project/{jid}/bulk/apply",
                    json={"operation_id": op_id,
                          "idempotency_key": "api-retry-1"})
    assert r.status_code == 200
    items = {i["clip_index"]: i for i in r.json()["items"]}
    assert items[0]["outcome"] == "applied"
    assert items[1]["outcome"] == "excluded"
    # Retry with nothing eligible: the applied/excluded items are untouched.
    r = client.post(f"/api/project/{jid}/bulk/{op_id}/retry",
                    json={"idempotency_key": "api-retry-1"})
    assert r.status_code == 200
    items = {i["clip_index"]: i for i in r.json()["items"]}
    assert items[0]["result_version"] == 2
    assert items[1]["outcome"] == "excluded"


def test_undo_conflict_after_later_edit(client, tmp_path):
    jid = _job_id()
    _make_job(tmp_path, jid, {0: _project(0, version=1)})
    r = _prepare(client, jid, {"scope": {"mode": "selected",
                                        "clip_indices": [0]},
                              "intent": _intent(),
                              "idempotency_key": "api-uc-1"})
    op_id = r.json()["operation_id"]
    client.post(f"/api/project/{jid}/bulk/apply",
                json={"operation_id": op_id, "idempotency_key": "api-uc-1"})
    # A later single-clip edit (through the same versioned save path)
    # moves the clip past the bulk result.
    from clippyme.domain.clip_project import validate_project
    from clippyme.domain.project_render import save_project_version
    later = validate_project(_project(0, version=2, style="hormozi_bold"))
    later.captions.position = "top"
    job_dir = str(tmp_path / jid)
    v3, _ = save_project_version(job_dir=job_dir, clip_index=0,
                                 project=later, expected_version=2)
    assert v3 == 3
    r = client.post(f"/api/project/{jid}/bulk/{op_id}/undo",
                    json={"idempotency_key": "api-uc-1"})
    assert r.status_code == 200
    undo = r.json()["undo"]
    assert undo["items"]["0"]["outcome"] == "conflict"
    # The later edit survived.
    with open(tmp_path / jid / "clip-project-0.json",
              encoding="utf-8") as f:
        assert json.load(f)["captions"]["position"] == "top"
