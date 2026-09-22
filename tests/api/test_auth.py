"""Tests for Supabase JWT decoding, user extraction, and admin authorization."""
import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from clippyme.api.auth import (
    AuthUser,
    decode_jwt,
    get_current_user,
    require_admin,
)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def make_test_jwt(payload: dict, secret: str = "test-secret-key") -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def test_decode_valid_jwt():
    payload = {"sub": "user-123", "email": "test@example.com", "exp": time.time() + 3600}
    token = make_test_jwt(payload, "my-secret")
    decoded = decode_jwt(token, secret="my-secret")
    assert decoded["sub"] == "user-123"
    assert decoded["email"] == "test@example.com"


def test_decode_tampered_signature_rejected():
    payload = {"sub": "user-123", "exp": time.time() + 3600}
    token = make_test_jwt(payload, "secret-one")
    with pytest.raises(ValueError, match="signature verification failed"):
        decode_jwt(token, secret="wrong-secret")


def test_decode_expired_token_rejected():
    payload = {"sub": "user-123", "exp": time.time() - 100}
    token = make_test_jwt(payload, "secret")
    with pytest.raises(ValueError, match="has expired"):
        decode_jwt(token, secret="secret")


def test_decode_malformed_token():
    with pytest.raises(ValueError, match="three parts"):
        decode_jwt("not.a.valid.jwt.token", secret="secret")


def test_decode_jwt_requires_secret():
    with pytest.raises(ValueError, match="valid non-empty secret is required"):
        decode_jwt("a.b.c", secret="")


def test_get_current_user_auth_enabled_missing_secret_fails_safely(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    req = MockRequest(headers={"authorization": "Bearer any.valid.token"})
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 500
    assert "misconfigured" in exc.value.detail


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_get_current_user_auth_disabled_returns_default(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "0")
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    req = MockRequest()
    user = get_current_user(req)
    assert user.id == "default_user"
    assert user.is_admin is True


def test_get_current_user_auth_enabled_requires_token(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "secret-key")
    req = MockRequest()
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401


def test_get_current_user_auth_enabled_valid_token(monkeypatch):
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWT_SECRET", "secret-key")
    payload = {"sub": "uuid-999", "email": "creator@clippyme.com", "exp": time.time() + 600}
    token = make_test_jwt(payload, "secret-key")
    req = MockRequest(headers={"authorization": f"Bearer {token}"})
    user = get_current_user(req)
    assert user.id == "uuid-999"
    assert user.email == "creator@clippyme.com"
    assert user.is_admin is False


def test_require_admin_guard(monkeypatch):
    regular_user = AuthUser(id="user1", email="user@test.com", is_admin=False)
    admin_user = AuthUser(id="admin", email="admin@test.com", is_admin=True)
    req = MockRequest()

    assert require_admin(req, admin_user).is_admin is True

    with pytest.raises(HTTPException) as exc:
        require_admin(req, regular_user)
    assert exc.value.status_code == 403
