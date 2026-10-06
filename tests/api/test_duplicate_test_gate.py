"""Gate tests for the duplicate-test endpoints (follow-up 6).

The endpoints are opt-in via CLIPPYME_ENABLE_TEST_ENDPOINTS and OFF by
default; when off they 404 as if they did not exist.
"""
import json

import pytest
from fastapi.testclient import TestClient

import clippyme.api.app as app_module

ORIGIN = {"Origin": "http://localhost:5175"}
# Valid UUID4 (version nibble 4, variant a).
SRC_ID = "123e4567-e89b-42d3-a456-426614174000"


@pytest.fixture
def client():
    return TestClient(app_module.app, headers=ORIGIN)


def _make_job(tmp_path, job_id):
    d = tmp_path / job_id
    d.mkdir()
    (d / "clip-project-0.json").write_text(
        json.dumps({"schema": "nugget.clip-project/1"}))
    (d / "job_metadata.json").write_text("{}")
    return d


def test_duplicate_test_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("CLIPPYME_ENABLE_TEST_ENDPOINTS", raising=False)
    r = client.post(f"/api/project/{SRC_ID}/duplicate-test")
    assert r.status_code == 404
    r = client.delete(f"/api/project/{SRC_ID}/duplicate-test")
    assert r.status_code == 404


def test_duplicate_test_enabled_roundtrip(client, tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPYME_ENABLE_TEST_ENDPOINTS", "1")
    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))
    _make_job(tmp_path, SRC_ID)
    r = client.post(f"/api/project/{SRC_ID}/duplicate-test")
    assert r.status_code == 200, r.text
    tid = r.json()["test_job_id"]
    assert (tmp_path / tid / ".test_job").is_file()
    # Source job untouched (isolation).
    assert (tmp_path / SRC_ID / "clip-project-0.json").is_file()
    # Delete works, and refuses non-test dirs.
    r = client.delete(f"/api/project/{tid}/duplicate-test")
    assert r.status_code == 200
    assert not (tmp_path / tid).exists()
    r = client.delete(f"/api/project/{SRC_ID}/duplicate-test")
    assert r.status_code == 400
