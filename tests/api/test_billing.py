"""Unit tests for quota management and billing routes."""
import json
import pytest
from fastapi.testclient import TestClient

from clippyme.api.app import app
from clippyme.api.auth import AuthUser
from clippyme.domain.quota_service import (
    check_user_quota,
    get_user_usage,
    record_user_usage,
    set_user_pro_status,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_quota_service_disabled_by_default(monkeypatch):
    monkeypatch.delenv("BILLING_ENABLED", raising=False)
    user = AuthUser(id="user_123", email="u@example.com")
    allowed, _ = check_user_quota(user, estimated_minutes=9999.0)
    assert allowed is True


def test_quota_service_limit_enforced(monkeypatch):
    monkeypatch.setenv("BILLING_ENABLED", "1")
    monkeypatch.setenv("FREE_TIER_MONTHLY_MINUTES", "10")

    user = AuthUser(id="user_free", email="free@example.com")
    # Initial usage should allow 5 minutes
    allowed, _ = check_user_quota(user, estimated_minutes=5.0)
    assert allowed is True

    # Record 8 minutes used
    record_user_usage(user.user_id, 8.0)
    usage = get_user_usage(user.user_id)
    assert usage["used_minutes"] == 8.0

    # Another 5 minutes would exceed 10 minutes
    allowed, msg = check_user_quota(user, estimated_minutes=5.0)
    assert allowed is False
    assert "limit reached" in msg

    # Admin bypasses quota limit
    admin = AuthUser(id="admin_user", is_admin=True)
    allowed_admin, _ = check_user_quota(admin, estimated_minutes=100.0)
    assert allowed_admin is True

    # Upgrade user to Pro bypasses quota limit
    set_user_pro_status(user.user_id, is_pro=True)
    allowed_pro, _ = check_user_quota(user, estimated_minutes=100.0)
    assert allowed_pro is True


def test_billing_usage_endpoint(client):
    resp = client.get("/api/billing/usage")
    assert resp.status_code == 200
    data = resp.json()
    assert "used_minutes" in data
    assert "limit_minutes" in data
    assert "tier" in data


def test_billing_checkout_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("BILLING_ENABLED", raising=False)
    resp = client.post(
        "/api/billing/checkout",
        json={"success_url": "https://example.com/success", "cancel_url": "https://example.com/cancel"},
    )
    assert resp.status_code == 400
    assert "Billing is not enabled" in resp.json()["detail"]


def _make_stripe_sig(payload_bytes: bytes, secret: str, ts: float | None = None) -> str:
    import hashlib
    import hmac
    import time
    t_str = str(int(ts if ts is not None else time.time()))
    signed_payload = f"{t_str}.".encode("utf-8") + payload_bytes
    sig = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return f"t={t_str},v1={sig}"


def test_billing_webhook_upgrade_downgrade(client, monkeypatch):
    secret = "whsec_test_secret_123"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)

    target_user = "user_webhook_test"
    set_user_pro_status(target_user, is_pro=False)
    assert get_user_usage(target_user)["is_pro"] is False

    # Simulate checkout.session.completed
    checkout_event = {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "client_reference_id": target_user,
            }
        }
    }
    raw_body = json.dumps(checkout_event).encode("utf-8")
    sig_hdr = _make_stripe_sig(raw_body, secret)
    resp = client.post("/api/billing/webhook", content=raw_body, headers={"stripe-signature": sig_hdr, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert get_user_usage(target_user)["is_pro"] is True

    # Simulate customer.subscription.deleted
    cancel_event = {
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "metadata": {"user_id": target_user},
            }
        }
    }
    raw_cancel = json.dumps(cancel_event).encode("utf-8")
    sig_cancel = _make_stripe_sig(raw_cancel, secret)
    resp = client.post("/api/billing/webhook", content=raw_cancel, headers={"stripe-signature": sig_cancel, "Content-Type": "application/json"})
    assert resp.status_code == 200
    assert get_user_usage(target_user)["is_pro"] is False


def test_billing_webhook_security_guards(client, monkeypatch):
    secret = "whsec_guard_test"
    payload = json.dumps({"type": "test"}).encode("utf-8")

    # 1. Unset STRIPE_WEBHOOK_SECRET -> 500 error
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    resp = client.post("/api/billing/webhook", content=payload)
    assert resp.status_code == 500
    assert "not configured" in resp.json()["detail"]

    # 2. Configured secret but missing signature header -> 400
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    resp = client.post("/api/billing/webhook", content=payload)
    assert resp.status_code == 400
    assert "Missing stripe-signature" in resp.json()["detail"]

    # 3. Invalid signature -> 400
    resp = client.post("/api/billing/webhook", content=payload, headers={"stripe-signature": "t=12345,v1=invalidsig"})
    assert resp.status_code == 400

    # 4. Expired signature (> 300s) -> 400
    import time
    old_sig = _make_stripe_sig(payload, secret, ts=time.time() - 400)
    resp = client.post("/api/billing/webhook", content=payload, headers={"stripe-signature": old_sig})
    assert resp.status_code == 400
    assert "expired" in resp.json()["detail"]
