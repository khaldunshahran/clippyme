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


JOB = "00000000-0000-4000-8000-0000000000a1"
MON = "00000000-0000-4000-8000-0000000000b2"


# --- 401 matrix (generated from route table) ---

# --- route-table-generated 401 matrix (item 2) ---------------------------------
# Every APIRoute in the live app must require authentication, except an
# explicit exemption list. Adding a new route without a guard FAILS the suite.

from fastapi.routing import APIRoute

# (method, path) pairs that are deliberately open. Everything else must 401.
EXEMPT_401 = {
    ("GET", "/"): "root",
    ("GET", "/api/health"): "health check",
    ("POST", "/api/billing/webhook"): "Stripe HMAC signature (own auth)",
}
# Framework routes (/docs, /openapi.json, /redoc) are Starlette Route objects,
# not APIRoute, and are not covered here; they expose only the API schema.

_PATH_PARAM_DUMMIES = {
    "job_id": "00000000-0000-4000-8000-0000000000a1",
    "monitor_id": "00000000-0000-4000-8000-0000000000b2",
    "clip_index": "0",
    "filename": "file.mp4",
    "clip_id": "clip1",
    "channel_id": "chan1",
    "name": "somefont",
}


def _concrete_path(path: str) -> str:
    import re
    def _sub(m):
        name = m.group(1)
        return _PATH_PARAM_DUMMIES.get(name, "x")
    return re.sub(r"\{(\w+)\}", _sub, path)


def _iter_guarded_routes():
    """Yield (method, concrete_path) for every APIRoute needing auth."""
    seen = set()
    for route in app_module.app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods):
            if method in ("HEAD", "OPTIONS"):
                continue
            key = (method, route.path)
            if key in EXEMPT_401:
                continue
            seen.add((method, _concrete_path(route.path)))
    return sorted(seen)


def test_401_matrix_covers_all_guarded_routes(auth_client):
    """Meta-test: report the exact route count and assert exemptions are valid."""
    routes = _iter_guarded_routes()
    # Sanity: the exemption list must only contain routes that actually exist.
    all_paths = {(m, r.path) for r in app_module.app.routes
                 if isinstance(r, APIRoute) for m in r.methods}
    for key in EXEMPT_401:
        assert key in all_paths, f"stale exemption {key}"
    print(f"\n401 matrix covers {len(routes)} routes "
          f"({len(EXEMPT_401)} exempt)")
    assert len(routes) > 90, "route table unexpectedly small"


@pytest.mark.parametrize("method,path", _iter_guarded_routes())
def test_unauthenticated_gets_401(auth_client, method, path):
    client = auth_client["client"]
    # No Authorization header at all; Origin present (browser-like).
    # GETs with a body are not sent; POST/PUT/DELETE send empty JSON.
    kwargs = {"headers": ORIGIN}
    if method in ("POST", "PUT", "DELETE", "PATCH"):
        kwargs["json"] = {}
    r = client.request(method, path, **kwargs)
    assert r.status_code == 401, f"{method} {path} -> {r.status_code}, expected 401"


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


@pytest.mark.parametrize("peer", ["::1", "::ffff:127.0.0.1"])
def test_trust_proxy_ipv6_loopback_peers_trusted(monkeypatch, peer):
    """IPv6 loopback forms must be treated as trusted peers."""
    monkeypatch.setenv("TRUST_PROXY", "1")
    # No Origin/SFS: falls through to the private-IP gate, which must pass
    # for loopback peers in any representation.
    req = _fake_request({}, client_host=peer)
    require_trusted_config_request(req)  # must not raise


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

# --- item 7: machine-readable 403 codes -----------------------------------------

def test_403_code_not_allowlisted(auth_client, monkeypatch):
    """Valid JWT, sub not on ALLOWED_USER_IDS -> 403 NOT_ALLOWLISTED."""
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ALLOWED_USER_IDS", "user_allowed")
    token = mint_for("user_intruder", priv)
    r = client.get(f"/api/status/{JOB}", headers={
        "Authorization": f"Bearer {token}", **ORIGIN})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "NOT_ALLOWLISTED"
    assert "allow-list" in body["detail"]


def test_403_code_admin_only(auth_client, monkeypatch):
    """Allow-listed non-admin on admin route -> 403 ADMIN_ONLY."""
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ALLOWED_USER_IDS", "user_pleb")
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    token = mint_for("user_pleb", priv)
    r = client.get("/api/config", headers={
        "Authorization": f"Bearer {token}", **ORIGIN})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "ADMIN_ONLY"
    assert "Administrative" in body["detail"]


def test_403_code_origin_rejected(auth_client, monkeypatch):
    """Evil Origin browser request -> 403 ORIGIN_REJECTED (not 401)."""
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ALLOWED_USER_IDS", "user_ok")
    token = mint_for("user_ok", priv)
    r = client.post("/api/process", headers={
        "Authorization": f"Bearer {token}",
        "Origin": "https://evil.example",
        "Sec-Fetch-Site": "cross-site",
    }, json={})
    assert r.status_code == 403
    body = r.json()
    assert body["code"] == "ORIGIN_REJECTED"
    assert body["detail"]


def test_403_body_shape(auth_client, monkeypatch):
    """All 403 bodies are {"detail": str, "code": str}."""
    client = auth_client["client"]
    priv = auth_client["priv"]
    monkeypatch.setenv("ALLOWED_USER_IDS", "user_allowed")
    token = mint_for("user_intruder", priv)
    r = client.get(f"/api/status/{JOB}", headers={
        "Authorization": f"Bearer {token}", **ORIGIN})
    body = r.json()
    assert set(body.keys()) == {"detail", "code"}
    assert isinstance(body["detail"], str)
    assert isinstance(body["code"], str)
