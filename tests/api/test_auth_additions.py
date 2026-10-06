"""Auth hardening additions: dev-bypass Cloudflare refusal (item 1).

The explicit dev bypass (AUTH_DISABLED_DEV=1) must never grant admin to a
request that arrived via Cloudflare. These tests use get_current_user
directly with the conftest default (AUTH_DISABLED_DEV=1, no AUTH_ENABLED).
"""
import pytest
from fastapi import HTTPException

from clippyme.api.auth import get_current_user


class MockRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}


def test_dev_bypass_refuses_cf_connecting_ip():
    req = MockRequest(headers={
        "cf-connecting-ip": "8.8.8.8",
        "host": "localhost:8000",
    })
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401


def test_dev_bypass_refuses_cf_ray():
    req = MockRequest(headers={
        "cf-ray": "abc123def456-EWR",
        "host": "localhost:8000",
    })
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401


def test_dev_bypass_refuses_nonlocal_host():
    req = MockRequest(headers={"host": "nugget-api.tellagbe.com"})
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401


def test_dev_bypass_refuses_empty_host():
    req = MockRequest(headers={})
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401


@pytest.mark.parametrize("host", [
    "localhost:8000", "localhost", "127.0.0.1:8000", "127.0.0.1",
    "[::1]:8000", "::1",
])
def test_dev_bypass_allows_local_hosts(host):
    req = MockRequest(headers={"host": host})
    user = get_current_user(req)
    assert user.is_admin
    assert user.id == "default_user"


def test_dev_bypass_spoofed_cf_headers_fail_closed():
    # Even a direct (non-Cloudflare) client that forges the headers is
    # refused: presence alone fails closed.
    req = MockRequest(headers={
        "cf-connecting-ip": "127.0.0.1",
        "host": "127.0.0.1:8000",
    })
    with pytest.raises(HTTPException) as exc:
        get_current_user(req)
    assert exc.value.status_code == 401
