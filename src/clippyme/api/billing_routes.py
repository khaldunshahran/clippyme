"""Billing and subscription routes for ClippyMe SaaS."""
from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, HttpUrl

from clippyme.api.auth import AuthUser, get_current_user
from clippyme.domain.quota_service import (
    create_stripe_checkout,
    get_user_usage,
    is_billing_enabled,
    set_user_pro_status,
)

logger = logging.getLogger("clippyme")
router = APIRouter(prefix="/api/billing", tags=["billing"])


class CheckoutRequest(BaseModel):
    success_url: str
    cancel_url: str


@router.get("/usage")
async def get_usage(user: AuthUser = Depends(get_current_user)):
    """Return the current user's monthly usage and quota."""
    return get_user_usage(user.user_id)


@router.post("/checkout")
async def create_checkout(
    req: CheckoutRequest,
    user: AuthUser = Depends(get_current_user),
):
    """Create a Stripe checkout session for upgrading to Pro."""
    if not is_billing_enabled():
        raise HTTPException(
            status_code=400,
            detail="Billing is not enabled on this instance.",
        )
    try:
        session = create_stripe_checkout(
            user=user,
            success_url=req.success_url,
            cancel_url=req.cancel_url,
        )
        return session
    except Exception as exc:
        logger.error("Checkout session creation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="stripe-signature"),
):
    """Handle Stripe subscription lifecycle webhooks."""
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        logger.error("Stripe webhook received but STRIPE_WEBHOOK_SECRET is not configured.")
        raise HTTPException(status_code=500, detail="Stripe webhook secret not configured")

    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    body = await request.body()

    try:
        # Parse stripe-signature header: t=...,v1=...
        sig_dict = {}
        for item in stripe_signature.split(","):
            k, _, v = item.partition("=")
            sig_dict[k.strip()] = v.strip()
        timestamp = sig_dict.get("t", "")
        expected_sig = sig_dict.get("v1", "")
        if not timestamp or not expected_sig:
            raise HTTPException(status_code=400, detail="Malformed stripe-signature header")

        # Verify timestamp replay tolerance (5 minutes)
        try:
            ts_float = float(timestamp)
            if abs(time.time() - ts_float) > 300:
                raise HTTPException(status_code=400, detail="Stripe webhook signature expired / replay detected")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp in stripe signature")

        signed_payload = f"{timestamp}.".encode("utf-8") + body
        computed_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(computed_sig, expected_sig):
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Stripe signature validation failed: %s", exc)
        raise HTTPException(status_code=400, detail="Signature error")

    try:
        event = json.loads(body.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = event.get("type", "")
    data_object = event.get("data", {}).get("object", {})

    logger.info("Received Stripe webhook event: %s", event_type)

    if event_type in ("checkout.session.completed", "customer.subscription.created"):
        user_id = data_object.get("client_reference_id") or data_object.get("metadata", {}).get("user_id")
        if user_id:
            set_user_pro_status(user_id, is_pro=True)
            logger.info("Upgraded user %s to Pro via webhook %s", user_id, event_type)

    elif event_type in ("customer.subscription.deleted", "customer.subscription.paused"):
        user_id = data_object.get("metadata", {}).get("user_id")
        if user_id:
            set_user_pro_status(user_id, is_pro=False)
            logger.info("Revoked Pro status for user %s via webhook %s", user_id, event_type)

    return {"received": True}
