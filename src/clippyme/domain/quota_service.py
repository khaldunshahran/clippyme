"""User quota tracking and billing service.

Enforces monthly processing limits for free-tier users and manages
Stripe checkout sessions and subscription status.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from clippyme.api.auth import AuthUser

logger = logging.getLogger("clippyme")

DEFAULT_FREE_MINUTES = 60.0
# In-memory storage fallback for usage when external DB is not active
_LOCAL_USAGE: Dict[str, Dict[str, float]] = {}


def is_billing_enabled() -> bool:
    """Return True if billing and quotas are enforced."""
    return os.getenv("BILLING_ENABLED", "0").lower() in ("1", "true", "yes")


def get_free_tier_minutes() -> float:
    """Return monthly free minutes allowance."""
    try:
        return float(os.getenv("FREE_TIER_MONTHLY_MINUTES", str(DEFAULT_FREE_MINUTES)))
    except (ValueError, TypeError):
        return DEFAULT_FREE_MINUTES


def _current_month_key() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def get_user_usage(user_id: str) -> Dict[str, Any]:
    """Return user's usage and tier information."""
    month_key = _current_month_key()
    user_records = _LOCAL_USAGE.setdefault(user_id, {})
    used_minutes = user_records.get(month_key, 0.0)
    is_pro = user_records.get("is_pro", 0.0) == 1.0

    return {
        "user_id": user_id,
        "month": month_key,
        "used_minutes": round(used_minutes, 2),
        "limit_minutes": 999999.0 if is_pro else get_free_tier_minutes(),
        "is_pro": is_pro,
        "tier": "pro" if is_pro else "free",
    }


def check_user_quota(user: AuthUser, estimated_minutes: float = 1.0) -> Tuple[bool, str]:
    """Check if the user is allowed to process a video.

    Returns (allowed: bool, reason: str).
    """
    if not is_billing_enabled():
        return True, ""

    if user.is_admin:
        return True, ""

    usage = get_user_usage(user.user_id)
    if usage["is_pro"]:
        return True, ""

    free_limit = get_free_tier_minutes()
    if usage["used_minutes"] + estimated_minutes > free_limit:
        msg = (
            f"Monthly free limit reached ({usage['used_minutes']:.1f}/{free_limit:.0f} mins used). "
            "Please upgrade to Pro for unlimited video processing."
        )
        return False, msg

    return True, ""


def record_user_usage(user_id: str, minutes_processed: float) -> None:
    """Record minutes processed for the current month."""
    if minutes_processed <= 0:
        return

    month_key = _current_month_key()
    user_records = _LOCAL_USAGE.setdefault(user_id, {})
    user_records[month_key] = user_records.get(month_key, 0.0) + minutes_processed
    logger.info(
        "Recorded %.2f mins for user %s (total %s: %.2f mins)",
        minutes_processed,
        user_id,
        month_key,
        user_records[month_key],
    )


def set_user_pro_status(user_id: str, is_pro: bool) -> None:
    """Set the Pro tier status for a user."""
    user_records = _LOCAL_USAGE.setdefault(user_id, {})
    user_records["is_pro"] = 1.0 if is_pro else 0.0
    logger.info("Updated pro status for user %s: %s", user_id, is_pro)


def create_stripe_checkout(user: AuthUser, success_url: str, cancel_url: str) -> Dict[str, Any]:
    """Create a Stripe Checkout session using standard HTTP API."""
    stripe_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    price_id = os.getenv("STRIPE_PRO_PRICE_ID", "").strip()

    if not stripe_key:
        raise ValueError("STRIPE_SECRET_KEY is not configured")
    if not price_id:
        raise ValueError("STRIPE_PRO_PRICE_ID is not configured")

    payload = {
        "success_url": success_url,
        "cancel_url": cancel_url,
        "payment_method_types[0]": "card",
        "mode": "subscription",
        "client_reference_id": user.user_id,
        "customer_email": user.email or "",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "metadata[user_id]": user.user_id,
    }

    data = urllib.parse.urlencode({k: v for k, v in payload.items() if v}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.stripe.com/v1/checkout/sessions",
        data=data,
        headers={
            "Authorization": f"Bearer {stripe_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
        return {
            "session_id": result.get("id"),
            "checkout_url": result.get("url"),
        }
