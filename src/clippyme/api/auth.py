"""Multi-tenant authentication and Supabase JWT verification for ClippyMe.

Supabase signs its JWTs asymmetrically (ES256, P-256). Verification uses the
project's JWKS endpoint — there is NO shared JWT secret anywhere in env or
code. The legacy HS256 dashboard secret must never be configured; even if it
leaks, it cannot mint a valid token because HS256 is never accepted.
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import jwt as pyjwt
from jwt import PyJWKClient
from jwt.algorithms import ECAlgorithm
from fastapi import HTTPException, Request

logger = logging.getLogger("clippyme")

# --- JWT verification parameters (pinned, per security review) --------------
JWT_ALGORITHMS = ["ES256"]  # exactly ES256 — never HS256/HS384/HS512/none
JWT_AUDIENCE = "authenticated"

_TRUE_VALUES = ("1", "true", "yes", "on")

# Minimum interval between JWKS refetches triggered by unknown kids. Prevents
# an attacker from forcing a network fetch on every request by spamming
# unknown kids; legitimate rotation still resolves within a minute.
_JWKS_MIN_REFETCH_INTERVAL = 60.0


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


def supabase_url() -> str:
    """Project base URL, e.g. https://frdlpogmqnozhgyrrhgf.supabase.co."""
    return os.environ.get("SUPABASE_URL", "").strip().rstrip("/")


def jwks_url() -> Optional[str]:
    base = supabase_url()
    return f"{base}/auth/v1/.well-known/jwks.json" if base else None


def jwt_issuer() -> Optional[str]:
    base = supabase_url()
    return f"{base}/auth/v1" if base else None


# --- JWKS client + key cache -------------------------------------------------
_jwks_client: Optional[PyJWKClient] = None
_jwks_client_url: Optional[str] = None
_jwks_keys: Dict[str, Any] = {}  # kid -> cryptography EC public key
_jwks_last_refetch: float = 0.0
_jwks_lock = threading.Lock()


def _get_client() -> Optional[PyJWKClient]:
    """PyJWKClient for the configured JWKS URL (recreated if URL changes)."""
    global _jwks_client, _jwks_client_url
    url = jwks_url()
    if not url:
        return None
    with _jwks_lock:
        if _jwks_client is None or _jwks_client_url != url:
            _jwks_client = PyJWKClient(url)
            _jwks_client_url = url
            _jwks_keys.clear()
        return _jwks_client


def _fetch_jwks_keys() -> Dict[str, Any]:
    """Fetch the JWKS document and return {kid: EC public key}.

    Only keys parseable as EC keys are kept; anything else is skipped.
    Raises on network/parse failure.
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("SUPABASE_URL is not set; cannot fetch JWKS")
    data = client.fetch_data()
    keys: Dict[str, Any] = {}
    for jwk_dict in data.get("keys", []):
        kid = jwk_dict.get("kid")
        if not kid:
            continue
        try:
            keys[kid] = ECAlgorithm.from_jwk(json.dumps(jwk_dict))
        except Exception as exc:
            logger.warning("Skipping unparseable JWKS key %r: %s", kid, exc)
    return keys


def prime_jwks_cache() -> int:
    """Fetch JWKS now and populate the key cache. Raises on failure.

    Called once at production startup — a failed boot fetch refuses to start.
    """
    global _jwks_last_refetch
    keys = _fetch_jwks_keys()
    if not keys:
        raise RuntimeError("JWKS fetch returned no usable signing keys")
    with _jwks_lock:
        _jwks_keys.clear()
        _jwks_keys.update(keys)
        _jwks_last_refetch = time.monotonic()
    return len(keys)


def reset_jwks_cache() -> None:
    """Clear cached keys/client (tests only)."""
    global _jwks_client, _jwks_client_url, _jwks_last_refetch
    with _jwks_lock:
        _jwks_client = None
        _jwks_client_url = None
        _jwks_keys.clear()
        _jwks_last_refetch = 0.0


def _signing_key_for(kid: str) -> Any:
    """Return the EC public key for kid, refetching at most once per minute.

    Unknown kids after a fresh refetch, or a failed refetch, fail closed.
    A failed refetch keeps serving already-cached (known) kids so transient
    network blips after boot don't hard-crash verification.
    """
    global _jwks_last_refetch
    with _jwks_lock:
        cached = _jwks_keys.get(kid)
    if cached is not None:
        return cached
    now = time.monotonic()
    with _jwks_lock:
        throttled = now - _jwks_last_refetch < _JWKS_MIN_REFETCH_INTERVAL
        if not throttled:
            _jwks_last_refetch = now
    if throttled:
        raise ValueError(f"Unknown JWT key id '{kid}'")
    try:
        fresh = _fetch_jwks_keys()
    except Exception as exc:
        raise ValueError(f"JWKS refetch failed: {exc}") from exc
    with _jwks_lock:
        _jwks_keys.update(fresh)
        key = _jwks_keys.get(kid)
    if key is None:
        raise ValueError(f"Unknown JWT key id '{kid}'")
    return key


def decode_jwt(token: str) -> Dict[str, Any]:
    """Verify a Supabase ES256 JWT via JWKS and return its claims.

    - ``kid`` selects the EC public key (unknown -> refetch once/min, else fail).
    - Algorithm pinned to ES256 (HS256/none/key-confusion fail).
    - ``iss`` must be ``<SUPABASE_URL>/auth/v1``; ``aud`` must be
      ``authenticated``; ``exp`` and ``sub`` are REQUIRED.

    Raises ValueError on any verification problem.
    """
    try:
        kid = pyjwt.get_unverified_header(token).get("kid")
    except Exception as exc:
        raise ValueError(f"Invalid JWT header: {exc}") from exc
    if not kid:
        raise ValueError("JWT is missing 'kid' header")
    key = _signing_key_for(kid)
    try:
        return pyjwt.decode(
            token,
            key,
            algorithms=JWT_ALGORITHMS,
            issuer=jwt_issuer(),
            audience=JWT_AUDIENCE,
            options={"require": ["exp", "sub"]},
        )
    except pyjwt.InvalidTokenError as exc:
        raise ValueError(f"JWT verification failed: {exc}") from exc


# --- env helpers --------------------------------------------------------------
def is_auth_enabled() -> bool:
    """Whether multi-tenant authentication is enforced.

    Set AUTH_ENABLED=1 (SUPABASE_URL alone also enables it, mirroring the old
    "secret set -> enabled" contract). When disabled the server fails closed
    (401) unless the explicit dev bypass AUTH_DISABLED_DEV=1 is present.
    """
    if os.environ.get("AUTH_ENABLED", "0").strip().lower() in _TRUE_VALUES:
        return True
    return bool(supabase_url())


def is_production() -> bool:
    """Whether the server runs in production (ENV=production)."""
    return os.environ.get("ENV", "").strip().lower() == "production"


def dev_bypass_enabled() -> bool:
    """Explicit dev bypass: unauthenticated callers get the local default admin.

    Must be set deliberately (AUTH_DISABLED_DEV=1). It is rejected at startup
    when ENV=production.
    """
    return os.environ.get("AUTH_DISABLED_DEV", "0").strip().lower() in _TRUE_VALUES


def _parse_id_set(env_name: str) -> Set[str]:
    raw = os.environ.get(env_name, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def allowed_user_ids() -> Set[str]:
    """Supabase ``sub`` values permitted to call the API (env ALLOWED_USER_IDS).

    NOTE: this is the *non-admin* allow-list. Admin IDs from
    ``ADMIN_USER_IDS`` are ALWAYS allowed (admin implies access) and do not
    need to be listed here. See ``admin_user_ids``.
    """
    return _parse_id_set("ALLOWED_USER_IDS")


def admin_user_ids() -> Set[str]:
    """Supabase ``sub`` values granted admin privileges (env ADMIN_USER_IDS).

    ADMIN IMPLIES ACCESS: every admin ID is automatically allowed to call
    the API (``get_current_user`` unions this set into the effective
    allow-list). Admins never need a separate ``ALLOWED_USER_IDS`` entry.
    First-boot: setting ONLY ``ADMIN_USER_IDS`` (with ``ALLOWED_USER_IDS``
    empty/unset) admits the admin and denies everyone else.
    """
    return _parse_id_set("ADMIN_USER_IDS")


def _dev_bypass_request_ok(request: Request) -> bool:
    """Whether the explicit dev bypass may apply to this request.

    The bypass must NEVER grant admin to a request that arrived via
    Cloudflare. Cloudflare always attaches ``CF-Connecting-IP`` and ``CF-Ray``
    to tunneled requests; their presence (even if spoofed by a direct client)
    fails closed here. Additionally the Host must look like local development
    (localhost / 127.0.0.1 / ::1, with or without port). ``testserver`` is
    Starlette TestClient's default Host and is allowed so the test-suite can
    exercise the bypass; it is not a routable name and the bypass is rejected
    at startup in production anyway.
    """
    if request.headers.get("cf-connecting-ip") or request.headers.get("cf-ray"):
        return False
    host = request.headers.get("host", "").strip().lower()
    if not host:
        return False
    if host.startswith("["):
        # Bracketed IPv6 literal with optional port: [::1]:8000
        hostname = host[1:].split("]")[0]
    elif host.count(":") > 1:
        # Bare IPv6 literal (a port would require brackets): ::1
        hostname = host
    else:
        hostname = host.split(":")[0]
    return hostname in {"localhost", "127.0.0.1", "::1", "testserver"}


def _default_local_admin() -> AuthUser:
    return AuthUser(
        id="default_user",
        email="local@clippyme.dev",
        role="admin",
        is_admin=True,
    )


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


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(request: Request) -> AuthUser:
    """FastAPI dependency to extract and validate the current authenticated user.

    Order of checks:
    1. ``x-admin-secret`` header matches ADMIN_SECRET_KEY -> admin.
    2. ``CLIPPYME_API_TOKEN`` (``X-API-Token`` or ``Authorization: Bearer``)
       -> admin, but NEVER for browser callers: a request carrying an
       ``Origin`` header skips this path entirely and must present a JWT.
    3. Explicit dev bypass AUTH_DISABLED_DEV=1 (never in production) ->
       local default admin.
    4. Supabase JWT bearer token -> verified via JWKS (ES256 pinned,
       ``iss``/``aud`` checked, ``exp`` + ``sub`` required).

    Allow-lists: ``ALLOWED_USER_IDS`` (comma-separated Supabase ``sub``
    values). A valid JWT whose sub is not allow-listed -> 403, and an empty
    allow-list denies everyone (fail closed in every environment). The
    explicit dev bypass (``AUTH_DISABLED_DEV=1``) is the only exception, and
    it refuses requests that look Cloudflare-originated. ``ADMIN_USER_IDS``
    grants admin (and implies access).
    """
    import hmac

    admin_secret = configured_admin_secret()
    req_admin_secret = request.headers.get("x-admin-secret", "").strip()
    if admin_secret and req_admin_secret and hmac.compare_digest(admin_secret, req_admin_secret):
        if hasattr(request, "state"):
            request.state.user_id = "admin"
        return AuthUser(id="admin", email="admin@clippyme.internal", role="admin", is_admin=True)

    # Shared LAN token: non-browser callers only. Browsers always send
    # Origin; a spoofed Origin on a non-browser client only pushes it onto
    # the stricter JWT path, so this fails closed.
    if request.headers.get("origin") is None:
        lan_token = os.environ.get("CLIPPYME_API_TOKEN", "").strip()
        if lan_token:
            supplied = request.headers.get("x-api-token", "").strip()
            auth_hdr = request.headers.get("authorization", "").strip()
            if not supplied and auth_hdr.lower().startswith("bearer "):
                supplied = auth_hdr[7:].strip()
            if supplied and hmac.compare_digest(lan_token, supplied):
                if hasattr(request, "state"):
                    request.state.user_id = "admin"
                return AuthUser(id="admin", email="lan_admin@clippyme.internal", role="admin", is_admin=True)

    # Explicit dev bypass. An explicitly enabled auth configuration
    # (AUTH_ENABLED=1) always wins over the bypass so the two cannot be
    # combined by accident; SUPABASE_URL alone does not block the bypass.
    if dev_bypass_enabled() and not is_production():
        if os.environ.get("AUTH_ENABLED", "0").strip().lower() not in _TRUE_VALUES:
            if _dev_bypass_request_ok(request):
                _admin = _default_local_admin()
                if hasattr(request, "state"):
                    request.state.user_id = _admin.id
                return _admin
            # Bypass refused: request looks Cloudflare-originated (or at least
            # not local dev). Fall through to the 401 below.

    token = extract_bearer_token(request)
    if not token:
        raise _unauthorized("Authentication required. Please sign in to continue.")

    if not is_auth_enabled():
        raise _unauthorized("Authentication required. Please sign in to continue.")

    try:
        claims = decode_jwt(token)
    except ValueError as exc:
        logger.warning("Token verification failed: %s", exc)
        raise _unauthorized(f"Invalid authentication token: {exc}") from exc

    # decode_jwt requires the sub claim; this is a second, explicit gate.
    user_id = claims.get("sub")
    if not user_id:
        raise _unauthorized("Authentication token missing user identifier ('sub').")
    user_id = str(user_id)

    admins = admin_user_ids()
    is_admin = user_id in admins
    if not is_admin:
        # Default-deny: the allow-list is the gate. An empty/unset
        # ALLOWED_USER_IDS admits nobody (fail closed in every environment;
        # the explicit dev bypass above is the only way around it).
        effective_allowed = allowed_user_ids() | admins
        if user_id not in effective_allowed:
            raise HTTPException(
                status_code=403,
                detail={"message": "Access denied: user is not on the server allow-list.",
                        "code": "NOT_ALLOWLISTED"},
            )

    email = claims.get("email")
    role = claims.get("role", "authenticated")
    if hasattr(request, "state"):
        request.state.user_id = user_id
    return AuthUser(id=user_id, email=email, role=role, is_admin=is_admin)


def require_admin(
    request: Request,
    user: Optional[AuthUser] = None,
) -> AuthUser:
    """Require that the current caller has administrative privileges."""
    import hmac

    if user is None or not isinstance(user, AuthUser):
        user = get_current_user(request)

    if user.is_admin:
        return user

    admin_secret = configured_admin_secret()
    req_admin_secret = request.headers.get("x-admin-secret", "").strip()
    if admin_secret and req_admin_secret and hmac.compare_digest(admin_secret, req_admin_secret):
        if hasattr(request, "state"):
            request.state.user_id = "admin"
        return AuthUser(id="admin", email="admin@clippyme.internal", role="admin", is_admin=True)

    raise HTTPException(
        status_code=403,
        detail={"message": "Administrative privileges required to access this resource.",
                "code": "ADMIN_ONLY"},
    )


def verify_job_ownership_on_disk(job_id: str, user: AuthUser, output_dir: str) -> None:
    """Ownership check for routers that lack app.py's in-memory jobs dict.

    Reads ``user_id`` from the job's ``.clippyme_runtime.json`` (falling back
    to ``*_metadata.json``), mirroring the on-disk fallback in app.py.
    Raises 404 when the job does not exist or belongs to another user.
    Admins always pass.
    """
    import glob
    import json

    if user.is_admin:
        return
    job_dir = os.path.join(output_dir, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    job_user = None
    runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
    if os.path.isfile(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                job_user = json.load(f).get("user_id")
        except Exception:
            pass
    if not job_user:
        meta_files = glob.glob(os.path.join(job_dir, "*_metadata.json"))
        if meta_files:
            try:
                with open(meta_files[0], "r", encoding="utf-8") as f:
                    job_user = json.load(f).get("user_id")
            except Exception:
                pass
    job_user = job_user or "default_user"
    if job_user != user.id:
        raise HTTPException(status_code=404, detail="Job not found")
