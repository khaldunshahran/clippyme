"""Tests for multi-tenant job isolation and user-scoped status/history.

JWTs are ES256 verified against a mocked JWKS (see test_auth_jwks).
"""
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

import clippyme.api.app as app_module
import clippyme.api.auth as auth_mod
from clippyme.api.auth import AuthUser, reset_jwks_cache
from tests.api.test_auth_jwks import (
    FakeJWKClient,
    TEST_SUPABASE_URL,
    _pub_to_jwk,
    mint_valid,
)

ORIGIN = {"Origin": "http://localhost:5175"}
KID = "kid-mt-1"


@pytest.fixture
def mt_env(monkeypatch):
    """Auth enabled with a mocked JWKS; returns a token minter."""
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_URL", TEST_SUPABASE_URL)
    monkeypatch.delenv("AUTH_DISABLED_DEV", raising=False)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    priv = ec.generate_private_key(ec.SECP256R1())
    fake = FakeJWKClient([_pub_to_jwk(priv.public_key(), KID)])
    monkeypatch.setattr(auth_mod, "_get_client", lambda: fake)
    reset_jwks_cache()
    auth_mod._jwks_last_refetch = 0.0
    yield lambda sub, **kw: mint_valid(sub=sub, priv=priv, kid=KID, **kw)
    reset_jwks_cache()


@pytest.fixture
def client():
    return TestClient(app_module.app, headers=ORIGIN)


def test_user_job_isolation(client, mt_env, monkeypatch):
    mint = mt_env
    token_a = mint("user_a")
    token_b = mint("user_b")
    headers_a = {"Authorization": f"Bearer {token_a}", **ORIGIN}
    headers_b = {"Authorization": f"Bearer {token_b}", **ORIGIN}

    # Manually register a job for User A in app_module.jobs
    job_id = "00000000-0000-4000-8000-000000000001"
    app_module.jobs[job_id] = {
        "status": "processing",
        "user_id": "user_a",
        "logs": ["Processing started..."],
        "result": None,
    }

    try:
        # User A checks status -> 200 OK
        resp_a = client.get(f"/api/status/{job_id}", headers=headers_a)
        assert resp_a.status_code == 200
        assert resp_a.json()["status"] == "processing"

        # User B checks status of User A's job -> 404 Not Found (no cross-tenant leakage)
        resp_b = client.get(f"/api/status/{job_id}", headers=headers_b)
        assert resp_b.status_code == 404

        # User B tries to cancel User A's job -> 404 Not Found
        cancel_b = client.post(f"/api/cancel/{job_id}", headers=headers_b)
        assert cancel_b.status_code == 404

        # User B tries to access transcript of User A's job -> 404 Not Found
        transcript_b = client.get(f"/api/transcript/{job_id}/0", headers=headers_b)
        assert transcript_b.status_code == 404

        # User B tries to trigger smartcut on User A's job -> 404 Not Found
        smartcut_b = client.post(f"/api/smartcut/{job_id}/0", headers=headers_b)
        assert smartcut_b.status_code == 404

        # User B tries to compose User A's job -> 404 Not Found
        compose_b = client.post(f"/api/compose/{job_id}/0", json={"toggles": {}}, headers=headers_b)
        assert compose_b.status_code == 404

        # User B tries to publish User A's job -> 404 Not Found
        publish_b = client.post(f"/api/publish/{job_id}/0", json={"platforms": [{"platform": "youtube", "accountId": "acc123"}]}, headers=headers_b)
        assert publish_b.status_code == 404

        # User B tries to delete User A's job -> 404 Not Found
        delete_b = client.delete(f"/api/history/{job_id}", headers=headers_b)
        assert delete_b.status_code == 404

        # Active jobs for User A lists the job
        active_a = client.get("/api/jobs/active", headers=headers_a)
        assert any(j["jobId"] == job_id for j in active_a.json()["jobs"])

        # Active jobs for User B does NOT list the job
        active_b = client.get("/api/jobs/active", headers=headers_b)
        assert not any(j["jobId"] == job_id for j in active_b.json()["jobs"])
    finally:
        app_module.jobs.pop(job_id, None)


def test_config_endpoints_gated_by_admin(client, mt_env, monkeypatch):
    mint = mt_env
    # Admin comes from ADMIN_USER_IDS (no email-based admin anymore).
    monkeypatch.setenv("ADMIN_USER_IDS", "admin_user")

    regular_token = mint("user_regular")
    admin_token = mint("admin_user")

    # Regular user attempting to access /api/config -> 403 Forbidden
    resp = client.get("/api/config", headers={"Authorization": f"Bearer {regular_token}", **ORIGIN})
    assert resp.status_code == 403

    # Admin user accessing /api/config -> 200 OK
    resp_admin = client.get("/api/config", headers={"Authorization": f"Bearer {admin_token}", **ORIGIN})
    assert resp_admin.status_code == 200
