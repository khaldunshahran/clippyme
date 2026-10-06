"""Auth hardening: 401 matrix, spoofed-header rejection, TRUST_PROXY proofs,
two-user ownership, per-user rate limits.

JWKS is mocked (see test_auth_jwks for crypto-level tests); these tests use
the same EC keypair fixture approach via a shared helper.
"""
import time
from types import SimpleNamespace

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi.testclient import TestClient

import clippyme.api.app as app_module
import clippyme.api.auth as auth_mod
import clippyme.api.security as sec_mod
from clippyme.api.security import (
    enforce_rate_limit,
    require_trusted_config_request,
)
from tests.api.test_auth_jwks import FakeJWKClient, _pub_to_jwk, mint_valid, TEST_SUPABASE_URL

KID = "kid-harden-1"
ORIGIN = {"Origin": "http://localhost:5175"}


@pytest.fixture
def auth_client(monkeypatch):
    """TestClient with auth enabled against a mocked JWKS (single EC key)."""
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_URL", TEST_SUPABASE_URL)
    monkeypatch.delenv("AUTH_DISABLED_DEV", raising=False)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    monkeypatch.delenv("CLIPPYME_API_TOKEN", raising=False)
    priv = ec.generate_private_key(ec.SECP256R1())
    pub = priv.public_key()
    fake = FakeJWKClient([_pub_to_jwk(pub, KID)])
    monkeypatch.setattr(auth_mod, "_get_client", lambda: fake)
    auth_mod.reset_jwks_cache()
    auth_mod._jwks_last_refetch = 0.0
    client = TestClient(app_module.app, headers=ORIGIN)
    yield {"client": client, "priv": priv, "pub": pub}
    auth_mod.reset_jwks_cache()


def mint_for(sub, priv):
    return mint_valid(sub=sub, priv=priv, kid=KID)


# --- 401 matrix -----------------------------------------------------------------

JOB = "00000000-0000-4000-8000-0000000000a1"
MON = "00000000-0000-4000-8000-0000000000b2"

ROUTES_401 = [
    # app.py — state-changing
    ("POST", "/api/process"), ("POST", "/api/batch"),
    ("POST", "/api/validate-url"),
    ("POST", f"/api/cancel/{JOB}"), ("POST", f"/api/pause/{JOB}"),
    ("POST", f"/api/resume/{JOB}"), ("POST", f"/api/stop/{JOB}"),
    ("POST", f"/api/jobs/{JOB}/retry"), ("POST", f"/api/jobs/{JOB}/rescore"),
    ("POST", f"/api/jobs/{JOB}/steer"),
    ("POST", f"/api/smartcut/{JOB}/0"), ("POST", f"/api/edit-ai/{JOB}/0"),
    ("POST", f"/api/generate-metadata/{JOB}"),
    ("POST", f"/api/generate-metadata/{JOB}/0"),
    ("POST", f"/api/reframe/{JOB}/0"),
    ("POST", "/api/storage/cleanup"),
    ("DELETE", f"/api/history/{JOB}"),
    ("POST", f"/api/compose/{JOB}/0"),
    ("PUT", f"/api/project/{JOB}/0"),
    ("POST", f"/api/project/{JOB}/duplicate-test"),
    ("DELETE", f"/api/project/{JOB}/duplicate-test"),
    ("POST", "/api/audio/upload"),
    ("POST", f"/api/project/{JOB}/batch-apply"),
    ("POST", f"/api/project/{JOB}/backfill"),
    ("POST", f"/api/publish/{JOB}/0"),
    ("POST", f"/api/clips/{JOB}/0/duplicate"),
    ("POST", f"/api/clips/{JOB}/0/schedule"),
    ("POST", f"/api/clips/{JOB}/0/upscale"),
    ("POST", "/api/analytics/sync"), ("POST", "/api/analytics/track"),
    ("POST", "/api/live-monitor/start"), ("POST", "/api/live-monitor/stop"),
    ("POST", f"/api/live-monitor/{MON}/config"),
    ("POST", f"/api/live-monitor/{MON}/publishing"),
    ("POST", f"/api/live-monitor/{MON}/publish-clip/clip1"),
    ("DELETE", f"/api/live-monitor/{MON}/pending-clip/clip1"),
    ("POST", f"/api/live-monitor/{MON}/publish-all"),
    ("POST", f"/api/history/{JOB}/restore"),
    # app.py — data-returning GETs
    ("GET", "/api/jobs/active"), ("GET", f"/api/status/{JOB}"),
    ("GET", f"/api/progress/{JOB}"), ("GET", f"/api/transcript/{JOB}/0"),
    ("GET", "/api/history"), ("GET", "/api/storage/breakdown"),
    ("GET", f"/api/project/{JOB}/0"), ("GET", f"/api/project/{JOB}/0/versions"),
    ("GET", f"/api/clips/{JOB}/0/schedule"), ("GET", f"/api/clips/{JOB}/0/export-xml"),
    ("GET", "/api/analytics/summary"), ("GET", "/api/analytics/insights"),
    ("GET", "/api/live-monitor/status"),
    ("GET", f"/api/live-monitor/{MON}/pending-clips"),
    # studio / ugc / highlights / dubbing
    ("POST", "/api/studio/titles"), ("POST", "/api/studio/titles/refine"),
    ("POST", "/api/studio/chapters"), ("POST", "/api/studio/thumbnail"),
    ("POST", "/api/ugc/research"), ("POST", "/api/ugc/scripts"),
    ("POST", f"/api/highlights/{JOB}/plan"),
    ("POST", f"/api/highlights/{JOB}/render"),
    ("POST", f"/api/highlights/{JOB}"),
    ("POST", f"/api/highlights/{JOB}/generate-all"),
    ("POST", f"/api/highlights/{JOB}/hl1/apply-edit"),
    ("GET", f"/api/highlights/{JOB}"),
    ("DELETE", f"/api/highlights/{JOB}/file.mp4"),
    ("GET", "/api/dubbing/languages"),
    ("POST", f"/api/dubbing/{JOB}/0"),
    # trends
    ("GET", "/api/trends"), ("POST", "/api/trends/scan"),
    ("POST", "/api/trends/clip"), ("GET", "/api/trends/config"),
    ("POST", "/api/trends/config"),
    # config (admin-only)
    ("GET", "/api/config"), ("GET", "/api/config/models"),
    ("POST", "/api/config"), ("POST", "/api/config/cookies"),
    ("GET", "/api/config/cookies/status"), ("DELETE", "/api/config/cookies"),
    ("GET", "/api/config/fonts"), ("POST", "/api/config/fonts"),
    ("DELETE", "/api/config/fonts/somefont"),
    ("GET", "/api/config/logo/status"), ("POST", "/api/config/logo"),
    ("DELETE", "/api/config/logo"),
    ("GET", "/api/config/zernio"), ("POST", "/api/config/zernio"),
    ("GET", "/api/zernio/accounts"),
    ("GET", "/api/config/watchdog"), ("POST", "/api/config/watchdog"),
    ("POST", "/api/config/watchdog/test"),
    # billing / channels
    ("GET", "/api/billing/usage"), ("POST", "/api/billing/checkout"),
    ("GET", "/api/channels/match"), ("GET", "/api/channels/chan1"),
    ("PUT", "/api/channels/chan1"), ("DELETE", "/api/channels/chan1"),
]


@pytest.mark.parametrize("method,path", ROUTES_401)
def test_unauthenticated_gets_401(auth_client, method, path):
    client = auth_client["client"]
    # No Authorization header at all; Origin present (browser-like).
    r = client.request(method, path, headers=ORIGIN)
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}, expected 401"


def test_health_and_root_stay_open(auth_client):
    client = auth_client["client"]
    assert client.get("/", headers=ORIGIN).status_code == 200
    assert client.get("/api/health", headers=ORIGIN).status_code == 200


# --- forged Referer / Host are NOT trusted ---------------------------------------

def _fake_request(headers, client_host):
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host=client_host))


def test_forged_referer_not_trusted(monkeypatch):
    monkeypatch.setattr(sec_mod, "ALLOWED_ORIGINS",
                        ["http://localhost:5175", "https://tellagbe.com"])
    req = _fake_request(
        {"referer": "https://tellagbe.com/some/page",
         "host": "127.0.0.1:8000"},
        client_host="8.8.8.8",
    )
    with pytest.raises(Exception) as exc:
        require_trusted_config_request(req)
    assert getattr(exc.value, "status_code", None) == 403


def test_forged_host_not_trusted(monkeypatch):
    monkeypatch.setattr(sec_mod, "ALLOWED_ORIGINS",
                        ["http://localhost:5175", "https://tellagbe.com"])
    req = _fake_request({"host": "tellagbe.com"}, client_host="8.8.8.8")
    with pytest.raises(Exception) as exc:
        require_trusted_config_request(req)
    assert getattr(exc.value, "status_code", None) == 403


def test_allowlisted_origin_still_passes(monkeypatch):
    monkeypatch.setattr(sec_mod, "ALLOWED_ORIGINS",
                        ["http://localhost:5175", "https://tellagbe.com"])
    req = _fake_request({"origin": "https://tellagbe.com"}, client_host="8.8.8.8")
    require_trusted_config_request(req)  # must not raise


# --- TRUST_PROXY proofs (tests only; flag stays off in env) -----------------------

def test_trust_proxy_legit_frontend_passes(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    monkeypatch.setattr(sec_mod, "ALLOWED_ORIGINS",
                        ["http://localhost:5175", "https://tellagbe.com"])
    # socket peer = cloudflared (loopback), XFF = real public client IP
    req = _fake_request(
        {"origin": "https://tellagbe.com", "x-forwarded-for": "8.8.4.4"},
        client_host="127.0.0.1",
    )
    require_trusted_config_request(req)  # layer 1 decides; must not raise


def test_trust_proxy_bare_curl_from_public_ip_fails(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    # direct connection: socket peer IS the public IP, no Origin/ SFS headers
    req = _fake_request({}, client_host="8.8.4.4")
    with pytest.raises(Exception) as exc:
        require_trusted_config_request(req)
    assert getattr(exc.value, "status_code", None) == 403


def test_trust_proxy_forged_xff_ignored_when_peer_untrusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    req = _fake_request({"x-forwarded-for": "127.0.0.1"}, client_host="8.8.4.4")
    with pytest.raises(Exception) as exc:
        require_trusted_config_request(req)
    assert getattr(exc.value, "status_code", None) == 403


# --- two-user ownership -------------------------------------------------------------

def _job_for(user_id):
    return {"status": "processing", "user_id": user_id, "logs": [], "result": None}


def test_two_user_ownership(auth_client, monkeypatch, tmp_path):
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ALLOWED_USER_IDS", "user_a,user_b")

    headers_a = {"Authorization": f"Bearer {mint_for('user_a', priv)}", **ORIGIN}
    headers_b = {"Authorization": f"Bearer {mint_for('user_b', priv)}", **ORIGIN}

    # user A owns a job (in-memory + on-disk runtime for router-level checks)
    app_module.jobs[JOB] = _job_for("user_a")
    job_dir = tmp_path / JOB
    job_dir.mkdir()
    (job_dir / ".clippyme_runtime.json").write_text('{"user_id": "user_a"}')
    import clippyme.api.highlight_routes as hl_mod
    import clippyme.api.dubbing_routes as dub_mod
    monkeypatch.setattr(hl_mod, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(dub_mod, "OUTPUT_DIR", str(tmp_path))

    try:
        # A sees own job; B does not
        assert client.get(f"/api/status/{JOB}", headers=headers_a).status_code == 200
        assert client.get(f"/api/status/{JOB}", headers=headers_b).status_code == 404

        # B cannot touch A's job through job-scoped routes
        assert client.post(f"/api/cancel/{JOB}", headers=headers_b).status_code == 404
        assert client.get(f"/api/transcript/{JOB}/0", headers=headers_b).status_code == 404
        assert client.get(f"/api/project/{JOB}/0", headers=headers_b).status_code == 404
        assert client.get(f"/api/highlights/{JOB}", headers=headers_b).status_code == 404
        assert client.delete(f"/api/highlights/{JOB}/x.mp4", headers=headers_b).status_code == 404

        # B cannot see A's job in the active list
        active_b = client.get("/api/jobs/active", headers=headers_b).json()["jobs"]
        assert all(j["jobId"] != JOB for j in active_b)
        active_a = client.get("/api/jobs/active", headers=headers_a).json()["jobs"]
        assert any(j["jobId"] == JOB for j in active_a)
    finally:
        app_module.jobs.pop(JOB, None)


def test_admin_sees_all_jobs(auth_client, monkeypatch):
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ADMIN_USER_IDS", "user_admin")
    headers_admin = {"Authorization": f"Bearer {mint_for('user_admin', priv)}", **ORIGIN}
    app_module.jobs[JOB] = _job_for("user_a")
    try:
        r = client.get("/api/jobs/active", headers=headers_admin)
        assert r.status_code == 200
        assert any(j["jobId"] == JOB for j in r.json()["jobs"])
        # admin bypasses ownership
        assert client.get(f"/api/status/{JOB}", headers=headers_admin).status_code == 200
    finally:
        app_module.jobs.pop(JOB, None)


# --- per-user rate limits -------------------------------------------------------------

def test_rate_limit_keyed_per_user(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    sec_mod._rate_state.clear()
    req_a = _fake_request({}, client_host="10.0.0.1")
    req_b = _fake_request({}, client_host="10.0.0.1")  # same IP, different user
    # capacity 2, no refill
    assert enforce_rate_limit(req_a, "testbucket", 2, 0, user_id="u1") is None
    assert enforce_rate_limit(req_a, "testbucket", 2, 0, user_id="u1") is None
    with pytest.raises(Exception) as exc:
        enforce_rate_limit(req_a, "testbucket", 2, 0, user_id="u1")
    assert getattr(exc.value, "status_code", None) == 429
    # user 2 has a fresh budget despite sharing the IP
    assert enforce_rate_limit(req_b, "testbucket", 2, 0, user_id="u2") is None
    sec_mod._rate_state.clear()
