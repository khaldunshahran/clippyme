"""Tests for auth helpers: env flags, bearer extraction, admin gate, dev bypass.

JWT crypto is covered in test_auth_jwks.py; route-level auth in
test_auth_hardening.py.
"""
import pytest
from fastapi import HTTPException

import clippyme.api.auth as auth_mod
from clippyme.api.auth import (
    AuthUser,
    dev_bypass_enabled,
    extract_bearer_token,
    get_current_user,
    is_auth_enabled,
    is_production,
    require_admin,
)


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_extract_bearer_token():
    assert extract_bearer_token(MockRequest({"authorization": "Bearer abc123"})) == "abc123"
    assert extract_bearer_token(MockRequest({"authorization": "bearer xyz"})) == "xyz"
    assert extract_bearer_token(MockRequest({})) is None
    assert extract_bearer_token(MockRequest({"authorization": "Basic abc"})) is None


def test_is_auth_enabled(monkeypatch):
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    assert not is_auth_enabled()
    monkeypatch.setenv("AUTH_ENABLED", "1")
    assert is_auth_enabled()
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    assert is_auth_enabled()


def test_is_production(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    assert not is_production()
    monkeypatch.setenv("ENV", "production")
    assert is_production()
    monkeypatch.setenv("ENV", "Production")
    assert is_production()


def test_dev_bypass_enabled(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED_DEV", raising=False)
    assert not dev_bypass_enabled()
    monkeypatch.setenv("AUTH_DISABLED_DEV", "1")
    assert dev_bypass_enabled()


def test_dev_bypass_returns_local_admin(monkeypatch):
    monkeypatch.setenv("AUTH_DISABLED_DEV", "1")
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    user = get_current_user(MockRequest({"host": "localhost:8000"}))
    assert user.id == "default_user"
    assert user.is_admin


def test_dev_bypass_rejected_when_auth_explicitly_enabled(monkeypatch):
    """AUTH_ENABLED=1 always wins over the dev bypass."""
    monkeypatch.setenv("AUTH_DISABLED_DEV", "1")
    monkeypatch.setenv("AUTH_ENABLED", "1")
    monkeypatch.delenv("ENV", raising=False)
    with pytest.raises(HTTPException) as exc:
        get_current_user(MockRequest({}))
    assert exc.value.status_code == 401


def test_no_auth_no_bypass_fails_closed(monkeypatch):
    monkeypatch.delenv("AUTH_DISABLED_DEV", raising=False)
    monkeypatch.delenv("AUTH_ENABLED", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    with pytest.raises(HTTPException) as exc:
        get_current_user(MockRequest({}))
    assert exc.value.status_code == 401


def test_require_admin_passes_for_admin():
    user = AuthUser(id="boss", is_admin=True)
    assert require_admin(MockRequest({}), user).is_admin


def test_require_admin_rejects_non_admin():
    with pytest.raises(HTTPException) as exc:
        require_admin(MockRequest({}), AuthUser(id="pleb", is_admin=False))
    assert exc.value.status_code == 403


def test_require_admin_accepts_admin_secret(monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET_KEY", "topsecret")
    req = MockRequest({"x-admin-secret": "topsecret"})
    user = require_admin(req, AuthUser(id="pleb", is_admin=False))
    assert user.is_admin


def test_admin_secret_header_grants_admin(monkeypatch):
    monkeypatch.setenv("ADMIN_SECRET_KEY", "topsecret")
    req = MockRequest({"x-admin-secret": "topsecret"})
    user = get_current_user(req)
    assert user.is_admin
    assert user.id == "admin"
