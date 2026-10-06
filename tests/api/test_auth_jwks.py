"""Tests for ES256/JWKS JWT verification (Supabase asymmetric signing).

Uses a locally generated EC P-256 keypair + mocked JWKS — never touches the
real Supabase project.
"""
import base64
import json
import time

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

import clippyme.api.auth as auth_mod
from clippyme.api.auth import decode_jwt, get_current_user, reset_jwks_cache

TEST_SUPABASE_URL = "https://test.supabase.co"
TEST_ISSUER = TEST_SUPABASE_URL + "/auth/v1"
KID_1 = "kid-aaa-111"
KID_2 = "kid-bbb-222"


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _pub_to_jwk(pub_key, kid):
    nums = pub_key.public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64u(nums.x.to_bytes(32, "big")),
        "y": _b64u(nums.y.to_bytes(32, "big")),
        "kid": kid,
        "use": "sig",
        "alg": "ES256",
    }


class FakeJWKClient:
    """Stand-in for PyJWKClient returning a canned JWKS document."""

    def __init__(self, keys, fail=False):
        self._keys = keys
        self.fail = fail
        self.fetch_count = 0

    def fetch_data(self):
        self.fetch_count += 1
        if self.fail:
            raise ConnectionError("mock network failure")
        return {"keys": self._keys}


@pytest.fixture
def ec_keys():
    priv1 = ec.generate_private_key(ec.SECP256R1())
    priv2 = ec.generate_private_key(ec.SECP256R1())
    return {
        KID_1: (priv1, priv1.public_key()),
        KID_2: (priv2, priv2.public_key()),
    }


@pytest.fixture
def jwks_env(monkeypatch, ec_keys):
    """Auth enabled against a mocked JWKS containing only KID_1."""
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_URL", TEST_SUPABASE_URL)
    monkeypatch.delenv("AUTH_DISABLED_DEV", raising=False)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    priv1, pub1 = ec_keys[KID_1]
    fake = FakeJWKClient([_pub_to_jwk(pub1, KID_1)])
    monkeypatch.setattr(auth_mod, "_get_client", lambda: fake)
    reset_jwks_cache()
    auth_mod._jwks_last_refetch = 0.0
    yield {"fake": fake, "keys": ec_keys}
    reset_jwks_cache()


def mint(payload, priv_key, kid, alg="ES256"):
    payload = {"iss": TEST_ISSUER, "aud": "authenticated", **payload}
    return pyjwt.encode(payload, priv_key, algorithm=alg, headers={"kid": kid})


def mint_valid(sub="user-123", priv=None, kid=KID_1, ec_keys=None, **overrides):
    priv_key = priv or ec_keys[kid][0]
    no_exp = overrides.pop("no_exp", False)
    no_sub = overrides.pop("no_sub", False)
    payload = {"sub": sub, "exp": time.time() + 3600}
    payload.update(overrides)
    if no_exp:
        payload.pop("exp", None)
    if no_sub:
        payload.pop("sub", None)
    return mint(payload, priv_key, kid)


# --- positive + kid handling -------------------------------------------------

def test_valid_es256_token_passes(jwks_env):
    token = mint_valid(ec_keys=jwks_env["keys"])
    claims = decode_jwt(token)
    assert claims["sub"] == "user-123"
    assert claims["iss"] == TEST_ISSUER
    assert claims["aud"] == "authenticated"


def test_unknown_kid_rejected(jwks_env, ec_keys):
    # Signed by a real key, but the kid is not in the (mocked) JWKS.
    token = mint_valid(priv=ec_keys[KID_2][0], kid="kid-unknown", ec_keys=ec_keys)
    with pytest.raises(ValueError, match="Unknown JWT key id"):
        decode_jwt(token)


def test_rotation_new_kid_picked_up(jwks_env, ec_keys, monkeypatch):
    # Cache primed with KID_1 only; then KID_2 appears in JWKS.
    decode_jwt(mint_valid(ec_keys=ec_keys))  # prime cache with kid 1
    priv2, pub2 = ec_keys[KID_2]
    priv1, pub1 = ec_keys[KID_1]
    fake2 = FakeJWKClient([_pub_to_jwk(pub1, KID_1), _pub_to_jwk(pub2, KID_2)])
    monkeypatch.setattr(auth_mod, "_get_client", lambda: fake2)
    auth_mod._jwks_last_refetch = 0.0  # allow immediate refetch
    claims = decode_jwt(mint_valid(kid=KID_2, ec_keys=ec_keys))
    assert claims["sub"] == "user-123"
    assert fake2.fetch_count >= 1


def test_jwks_fetch_failure_known_kid_still_served(jwks_env, ec_keys, monkeypatch):
    good_token = mint_valid(ec_keys=jwks_env["keys"])
    decode_jwt(good_token)  # prime cache
    jwks_env["fake"].fail = True
    auth_mod._jwks_last_refetch = 0.0
    # Unknown kid + failed refetch -> fail closed ...
    with pytest.raises(ValueError):
        decode_jwt(mint_valid(priv=ec_keys[KID_2][0], kid="kid-nope", ec_keys=ec_keys))
    # ... but the cached known kid still verifies (no hard crash on blips).
    claims = decode_jwt(good_token)
    assert claims["sub"] == "user-123"


# --- algorithm pinning ---------------------------------------------------------

def test_hs256_key_confusion_rejected(jwks_env, ec_keys):
    """Classic key-confusion: HS256 token keyed with the EC *public* key bytes.

    Crafted manually with hmac because PyJWT's own encoder refuses to misuse
    an asymmetric key as an HMAC secret — the attacker wouldn't be so polite.
    """
    import hashlib
    import hmac as hmac_mod

    priv1, pub1 = ec_keys[KID_1]
    pub_der = pub1.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header_b64 = _b64u(json.dumps({"alg": "HS256", "kid": KID_1}).encode())
    payload_b64 = _b64u(json.dumps({
        "sub": "attacker", "iss": TEST_ISSUER, "aud": "authenticated",
        "exp": time.time() + 3600,
    }).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig_b64 = _b64u(hmac_mod.new(pub_der, signing_input, hashlib.sha256).digest())
    token = f"{header_b64}.{payload_b64}.{sig_b64}"
    with pytest.raises(ValueError, match="JWT verification failed"):
        decode_jwt(token)


def test_legacy_hs256_shared_secret_rejected(jwks_env):
    """The legacy HS256 dashboard secret is useless even if leaked: HS256 never accepted."""
    token = mint({"sub": "attacker", "exp": time.time() + 3600},
                 "leaked-legacy-hmac-secret", KID_1, alg="HS256")
    with pytest.raises(ValueError, match="JWT verification failed"):
        decode_jwt(token)


def test_alg_none_rejected(jwks_env):
    header = _b64u(json.dumps({"alg": "none", "kid": KID_1}).encode())
    payload = _b64u(json.dumps({
        "sub": "attacker", "iss": TEST_ISSUER, "aud": "authenticated",
        "exp": time.time() + 3600,
    }).encode())
    token = f"{header}.{payload}."
    with pytest.raises(ValueError, match="JWT verification failed"):
        decode_jwt(token)


def test_missing_kid_rejected(jwks_env, ec_keys):
    priv1, _ = ec_keys[KID_1]
    token = pyjwt.encode(
        {"sub": "x", "iss": TEST_ISSUER, "aud": "authenticated", "exp": time.time() + 3600},
        priv1, algorithm="ES256",
    )  # no kid header
    with pytest.raises(ValueError, match="kid"):
        decode_jwt(token)


# --- claim requirements ----------------------------------------------------------

def test_expired_rejected(jwks_env, ec_keys):
    token = mint_valid(ec_keys=ec_keys, exp=time.time() - 10)
    with pytest.raises(ValueError, match="verification failed"):
        decode_jwt(token)


def test_missing_exp_rejected(jwks_env, ec_keys):
    token = mint_valid(ec_keys=ec_keys, no_exp=True)
    with pytest.raises(ValueError, match="verification failed"):
        decode_jwt(token)


def test_missing_sub_rejected(jwks_env, ec_keys):
    token = mint_valid(ec_keys=ec_keys, no_sub=True)
    with pytest.raises(ValueError, match="verification failed"):
        decode_jwt(token)


def test_wrong_iss_rejected(jwks_env, ec_keys):
    token = mint_valid(ec_keys=ec_keys, iss="https://evil.example/auth/v1")
    with pytest.raises(ValueError, match="verification failed"):
        decode_jwt(token)


def test_wrong_aud_rejected(jwks_env, ec_keys):
    token = mint_valid(ec_keys=ec_keys, aud="wrong-audience")
    with pytest.raises(ValueError, match="verification failed"):
        decode_jwt(token)


# --- get_current_user integration ---------------------------------------------------

class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def _bearer(token):
    return MockRequest(headers={"authorization": f"Bearer {token}"})


def test_anon_key_shaped_token_rejected(jwks_env, ec_keys):
    """A publishable/anon-key-shaped token (role=anon, no user sub) gets 401."""
    token = mint({"role": "anon", "exp": time.time() + 3600},
                 ec_keys[KID_1][0], KID_1)  # no sub claim
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 401


def test_non_allowlisted_sub_gets_403(jwks_env, ec_keys, monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-allowed-1,user-allowed-2")
    token = mint_valid(sub="user-intruder", ec_keys=ec_keys)
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 403


def test_allowlisted_sub_passes(jwks_env, ec_keys, monkeypatch):
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-123")
    token = mint_valid(sub="user-123", ec_keys=ec_keys)
    user = get_current_user(_bearer(token))
    assert user.id == "user-123"
    assert not user.is_admin


def test_admin_allowlist_grants_admin(jwks_env, ec_keys, monkeypatch):
    monkeypatch.setenv("ADMIN_USER_IDS", "user-boss")
    monkeypatch.setenv("ALLOWED_USER_IDS", "someone-else")
    token = mint_valid(sub="user-boss", ec_keys=ec_keys)
    user = get_current_user(_bearer(token))
    assert user.is_admin


def test_production_empty_allowlist_denies_everyone(jwks_env, ec_keys, monkeypatch):
    monkeypatch.setenv("ENV", "production")
    token = mint_valid(sub="user-123", ec_keys=ec_keys)
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 403


def test_no_token_no_bypass_401(jwks_env):
    with pytest.raises(HTTPException) as exc:
        get_current_user(MockRequest(headers={}))
    assert exc.value.status_code == 401


def test_origin_present_skips_lan_token(jwks_env, ec_keys, monkeypatch):
    """CLIPPYME_API_TOKEN is never accepted from browser callers (Origin present)."""
    monkeypatch.setenv("CLIPPYME_API_TOKEN", "lan-secret-token")
    req = MockRequest(headers={
        "origin": "https://tellagbe.com",
        "authorization": "Bearer lan-secret-token",
    })
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401  # pushed onto the JWT path, not token path


def test_no_origin_lan_token_accepted(monkeypatch):
    """Non-browser caller (no Origin) can still use CLIPPYME_API_TOKEN."""
    monkeypatch.setenv("CLIPPYME_API_TOKEN", "lan-secret-token")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    req = MockRequest(headers={"authorization": "Bearer lan-secret-token"})
    user = get_current_user(req)
    assert user.is_admin


# --- item 3 additions: hostile iss/jku, no-kid, malformed, refetch cap --------

def _b64url_json(obj):
    import base64, json
    return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")


def test_hostile_iss_gets_401_not_500(jwks_env, ec_keys, monkeypatch):
    """A token with an attacker-controlled iss is rejected (401, not 500)."""
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-123")
    token = mint_valid(ec_keys=ec_keys, iss="https://evil.example/auth/v1")
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 401


def test_jku_header_is_ignored(jwks_env, ec_keys, monkeypatch):
    """The jku/x5u headers must never influence key selection.

    A token pointing jku at an attacker URL with an unknown kid must be
    rejected — we only ever use the JWKS URL derived from SUPABASE_URL.
    """
    import jwt as pyjwt_local
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-123")
    priv1 = ec_keys[KID_1][0]
    payload = {"sub": "user-123", "exp": time.time() + 3600,
               "iss": TEST_ISSUER, "aud": "authenticated"}
    token = pyjwt_local.encode(
        payload, priv1, algorithm="ES256",
        headers={"kid": "kid-attacker-unknown",
                 "jku": "https://evil.example/.well-known/jwks.json",
                 "x5u": "https://evil.example/cert.pem"})
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 401


def test_no_kid_gets_401_not_500(jwks_env, ec_keys, monkeypatch):
    """A token without a kid header -> 401 (never a 500)."""
    import jwt as pyjwt_local
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-123")
    priv1 = ec_keys[KID_1][0]
    payload = {"sub": "user-123", "exp": time.time() + 3600,
               "iss": TEST_ISSUER, "aud": "authenticated"}
    # pyjwt always sets kid only if provided; craft manually without kid.
    token = pyjwt_local.encode(payload, priv1, algorithm="ES256")
    assert "kid" not in pyjwt_local.get_unverified_header(token)
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 401


@pytest.mark.parametrize("bad_token", [
    "not-a-jwt",
    "only.two",
    "a.b.c.d",
    "!!!.@@@.###",
])
def test_malformed_token_gets_401_not_500(jwks_env, bad_token, monkeypatch):
    """Malformed tokens -> 401 (never a 500)."""
    monkeypatch.setenv("ALLOWED_USER_IDS", "user-123")
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(bad_token))
    assert exc.value.status_code == 401


def test_refetch_cap_under_junk_kid_flood(jwks_env, ec_keys):
    """Junk-kid flooding triggers at most one JWKS refetch per 60s."""
    fake = jwks_env["fake"]
    fake.fetch_count = 0
    # prime: one valid decode populates cache (may fetch once)
    decode_jwt(mint_valid(ec_keys=ec_keys))
    fetches_after_prime = fake.fetch_count
    # flood with junk kids
    for i in range(10):
        token = mint_valid(priv=ec_keys[KID_1][0],
                           kid=f"junk-kid-{i}", ec_keys=ec_keys)
        with pytest.raises(ValueError, match="[Uu]nknown JWT key id|refetch"):
            decode_jwt(token)
    # At most one refetch for the whole flood (the first junk kid); the rest
    # are throttled by the 60s cap.
    assert fake.fetch_count - fetches_after_prime <= 1


def test_empty_allowlist_denies_outside_production_too(jwks_env, ec_keys, monkeypatch):
    """Empty ALLOWED_USER_IDS denies everyone, even outside production.

    (Changed from the old 'empty-outside-production-passes' behavior.)
    """
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    monkeypatch.delenv("ADMIN_USER_IDS", raising=False)
    monkeypatch.delenv("ENV", raising=False)  # explicitly NOT production
    token = mint_valid(sub="user-123", ec_keys=ec_keys)
    with pytest.raises(HTTPException) as exc:
        get_current_user(_bearer(token))
    assert exc.value.status_code == 403
