"""Multi-tenant authentication and Supabase JWT verification for ClippyMe."""
import base64
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request

logger = logging.getLogger("clippyme")


@dataclass
class AuthUser:
    """Authenticated user context for multi-tenant requests."""

    id: str
    email: Optional[str] = None
    role: str = "authenticated"
    is_admin: bool = False

    @property
    def user_id(self) -> str:
        return self.id


def _b64url_decode(raw: str) -> bytes:
    """Decode a base64url-encoded string with missing padding restored."""
    rem = len(raw) % 4
    if rem > 0:
        raw += "=" * (4 - rem)
    return base64.urlsafe_b64decode(raw.encode("ascii"))


def decode_jwt(token: str, secret: str) -> Dict[str, Any]:
    """Decode and verify an HS256 JWT using standard library hmac + hashlib.

    Raises ValueError on structural malformation, invalid signature, or expired token.
    """
    if not secret or not secret.strip():
        raise ValueError("A valid non-empty secret is required for JWT signature verification")

    parts = token.strip().split(".")
    if len(parts) != 3:
        raise ValueError("JWT must have exactly three parts (header.payload.signature)")

    header_b64, payload_b64, sig_b64 = parts

    try:
        header_bytes = _b64url_decode(header_b64)
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JWT header: {exc}") from exc

    alg = header.get("alg", "")
    if alg != "HS256":
        raise ValueError(f"Unsupported JWT algorithm '{alg}'. Only HS256 is supported.")

    try:
        payload_bytes = _b64url_decode(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JWT payload: {exc}") from exc

    # Recompute expected signature
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(
        secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    try:
        provided_sig = _b64url_decode(sig_b64)
    except Exception as exc:
        raise ValueError(f"Invalid signature encoding: {exc}") from exc

    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("JWT signature verification failed")

    # Expiry verification
    exp = payload.get("exp")
    if exp is not None:
        try:
            if float(exp) < time.time():
                raise ValueError("JWT token has expired")
        except (TypeError, ValueError) as exc:
            if str(exc) == "JWT token has expired":
                raise
            raise ValueError("Invalid exp claim in JWT") from exc

    return payload


def is_auth_enabled() -> bool:
    """Whether multi-tenant authentication is enforced.

    Defaults to False for backward-compatible local self-hosting.
    Set AUTH_ENABLED=1 or SUPABASE_JWT_SECRET to enable.
    """
    if os.environ.get("AUTH_ENABLED", "0") in ("1", "true", "yes", "on"):
        return True
    return bool(os.environ.get("SUPABASE_JWT_SECRET", "").strip())


def configured_admin_secret() -> Optional[str]:
    """Admin secret key for bypassing auth or configuring global settings."""
    secret = os.environ.get("ADMIN_SECRET_KEY", "").strip()
    return secret or None


def extract_bearer_token(request: Request) -> Optional[str]:
    """Extract JWT token from Authorization: Bearer <token>."""
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return None


def get_current_user(request: Request) -> AuthUser:
    """FastAPI dependency to extract and validate the current authenticated user.

    In development / single-tenant mode (AUTH_ENABLED=0), unauthenticated requests
    receive a default local admin user. In multi-tenant mode (AUTH_ENABLED=1),
    valid Supabase JWT authentication is strictly required.
    """
    admin_secret = configured_admin_secret()
    req_admin_secret = request.headers.get("x-admin-secret", "").strip()
    if admin_secret and req_admin_secret and hmac.compare_digest(admin_secret, req_admin_secret):
        return AuthUser(id="admin", email="admin@clippyme.internal", role="admin", is_admin=True)

    lan_token = os.environ.get("CLIPPYME_API_TOKEN", "").strip()
    if lan_token:
        supplied = request.headers.get("x-api-token", "").strip()
        auth_hdr = request.headers.get("authorization", "").strip()
        if not supplied and auth_hdr.lower().startswith("bearer "):
            supplied = auth_hdr[7:].strip()
        if supplied and hmac.compare_digest(lan_token, supplied):
            return AuthUser(id="admin", email="lan_admin@clippyme.internal", role="admin", is_admin=True)

    token = extract_bearer_token(request)
    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET", "").strip() or None

    if not token:
        if not is_auth_enabled():
            return AuthUser(
                id="default_user",
                email="local@clippyme.dev",
                role="admin",
                is_admin=True,
            )
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please sign in to continue.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if is_auth_enabled() and not jwt_secret:
        logger.error("Authentication is enabled but SUPABASE_JWT_SECRET is not configured.")
        raise HTTPException(
            status_code=500,
            detail="Server authentication misconfigured: missing JWT secret.",
        )

    if not jwt_secret:
        # Dev / single-tenant mode without JWT secret configured
        return AuthUser(
            id="default_user",
            email="local@clippyme.dev",
            role="admin",
            is_admin=True,
        )

    try:
        claims = decode_jwt(token, secret=jwt_secret)
    except ValueError as exc:
        logger.warning("Token verification failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authentication token missing user identifier ('sub').",
            headers={"WWW-Authenticate": "Bearer"},
        )

    email = claims.get("email")
    role = claims.get("role", "authenticated")
    admin_email = os.environ.get("ADMIN_EMAIL", "").strip().lower()
    is_admin = bool((role == "admin") or (email and admin_email and email.lower() == admin_email))

    return AuthUser(id=str(user_id), email=email, role=role, is_admin=is_admin)


def require_admin(
    request: Request,
    user: Optional[AuthUser] = None,
) -> AuthUser:
    """Require that the current caller has administrative privileges."""
    if user is None or not isinstance(user, AuthUser):
        user = get_current_user(request)

    if user.is_admin:
        return user

    admin_secret = configured_admin_secret()
    req_admin_secret = request.headers.get("x-admin-secret", "").strip()
    if admin_secret and req_admin_secret and hmac.compare_digest(admin_secret, req_admin_secret):
        return AuthUser(id="admin", email="admin@clippyme.internal", role="admin", is_admin=True)

    raise HTTPException(
        status_code=403,
        detail="Administrative privileges required to access this resource.",
    )
