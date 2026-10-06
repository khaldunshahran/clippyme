"""Structured access-log middleware: shape + no-secret-leak tests."""

import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _clean_access_log(tmp_path, monkeypatch):
    """Redirect the access logger to a temp file for this test."""
    import logging.handlers as lh
    from clippyme.api import app as app_module

    log_path = str(tmp_path / "access.jsonl")
    logger = app_module._access_logger
    old_handlers = logger.handlers[:]
    for h in old_handlers:
        logger.removeHandler(h)
    handler = lh.TimedRotatingFileHandler(
        log_path, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    handler.setFormatter(__import__("logging").Formatter("%(message)s"))
    logger.addHandler(handler)
    yield log_path
    logger.removeHandler(handler)
    handler.close()
    for h in old_handlers:
        logger.addHandler(h)


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_access_log_shape(_clean_access_log):
    """One JSON line per request with exactly the documented fields."""
    from clippyme.api.app import app

    client = TestClient(app)
    r = client.get("/api/health", headers={"CF-Connecting-IP": "203.0.113.7"})
    assert r.status_code == 200

    lines = _read_lines(_clean_access_log)
    assert len(lines) >= 1
    entry = lines[-1]
    assert set(entry.keys()) == {
        "timestamp", "method", "path", "status", "user_id", "cf_connecting_ip",
    }, f"unexpected fields: {set(entry.keys())}"
    assert entry["method"] == "GET"
    assert entry["path"] == "/api/health"
    assert entry["status"] == 200
    assert entry["user_id"] is None  # health is unauthenticated
    assert entry["cf_connecting_ip"] == "203.0.113.7"
    # timestamp parses as ISO-8601
    assert "T" in entry["timestamp"]


def test_access_log_no_secrets(_clean_access_log):
    """Authorization headers, tokens, and keys never appear in the log."""
    from clippyme.api.app import app

    secret_like = "sk-ant-testsecret12345"
    client = TestClient(app)
    r = client.get(
        "/api/health",
        headers={
            "Authorization": f"Bearer {secret_like}",
            "X-API-Token": secret_like,
            "CF-Connecting-IP": "198.51.100.9",
        },
    )
    assert r.status_code in (200, 401, 403)

    with open(_clean_access_log, encoding="utf-8") as f:
        raw = f.read()
    assert secret_like not in raw, "secret material leaked into access log"
    assert "Bearer" not in raw, "auth scheme leaked into access log"
    # Only the allow-listed header value is present, and only as its own field
    lines = _read_lines(_clean_access_log)
    assert lines[-1]["cf_connecting_ip"] == "198.51.100.9"


def test_access_log_unauthenticated_user_null(_clean_access_log):
    """Requests without auth log user_id null (not a crash)."""
    from clippyme.api.app import app

    client = TestClient(app)
    client.get("/api/health")
    lines = _read_lines(_clean_access_log)
    assert lines[-1]["user_id"] is None
    assert lines[-1]["cf_connecting_ip"] is None
