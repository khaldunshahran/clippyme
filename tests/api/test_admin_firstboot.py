"""First-boot: ADMIN_USER_IDS alone admits the admin (implies access)."""

import pytest
from fastapi import HTTPException

from clippyme.api.auth import get_current_user

from tests.api.test_auth_jwks import (
    FakeJWKClient,
    _bearer,
    _pub_to_jwk,
    ec_keys,  # noqa: F401  (fixture)
    jwks_env,  # noqa: F401  (fixture)
    mint_valid,
    KID_1,
    TEST_SUPABASE_URL,
)

ADMIN_SUB = "admin-user-001"
OTHER_SUB = "other-user-002"


@pytest.fixture
def admin_only_env(jwks_env, monkeypatch):
    """Only ADMIN_USER_IDS set; ALLOWED_USER_IDS empty/unset (first boot)."""
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_SUB)
    monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
    yield jwks_env


def test_admin_implies_access_first_boot(admin_only_env, ec_keys):
    """Admin JWT gets in with is_admin=True when only ADMIN_USER_IDS is set."""
    token = mint_valid(sub=ADMIN_SUB, ec_keys=ec_keys)
    user = get_current_user(_bearer(token))
    assert user.id == ADMIN_SUB
    assert user.is_admin is True


def test_non_listed_denied_first_boot(admin_only_env, ec_keys):
    """Valid JWT for non-listed sub gets 403 NOT_ALLOWLISTED."""
    token = mint_valid(sub=OTHER_SUB, ec_keys=ec_keys)
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(_bearer(token))
    assert exc_info.value.status_code == 403
    assert exc_info.value.detail["code"] == "NOT_ALLOWLISTED"


def test_admin_not_in_allowed_list_still_admin(jwks_env, monkeypatch, ec_keys):
    """Admin in ADMIN_USER_IDS but explicitly absent from ALLOWED_USER_IDS."""
    monkeypatch.setenv("ADMIN_USER_IDS", ADMIN_SUB)
    monkeypatch.setenv("ALLOWED_USER_IDS", "someone-else")
    token = mint_valid(sub=ADMIN_SUB, ec_keys=ec_keys)
    user = get_current_user(_bearer(token))
    assert user.is_admin is True
