"""Trust/origin helpers for ClippyMe config endpoints."""
import hmac
import ipaddress
import os
import time
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException, Request

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
)


def parse_allowed_origins(raw_value: Optional[str] = None) -> List[str]:
    """Parse a comma-separated ALLOWED_ORIGINS env var with safe localhost defaults."""
    if raw_value is None:
        return list(DEFAULT_ALLOWED_ORIGINS)

    origins = [origin.strip().rstrip("/") for origin in raw_value.split(",") if origin.strip()]
    return origins or list(DEFAULT_ALLOWED_ORIGINS)


# Import-time snapshot used ONLY to construct CORSMiddleware (Starlette bakes
# allow_origins at add_middleware — a restart is required for CORS preflight
# to see ALLOWED_ORIGINS changes). The per-request trust gate reads the live
# environment via allowed_origins() instead; see its docstring.
ALLOWED_ORIGINS = parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))


def _trust_proxy_enabled() -> bool:
    """Whether to honour X-Forwarded-For / X-Real-IP for client identity.

    OFF by default: when the app is reachable directly, those headers are
    fully attacker-controlled, so trusting them would let any client spoof
    its IP to dodge rate limiting or forge a "trusted" private address.
    Set TRUST_PROXY=1 only when ClippyMe sits behind exactly one reverse
    proxy (nginx/Traefik/uvicorn --proxy-headers). The proxy may either
    overwrite X-Forwarded-For or append to it (nginx's
    $proxy_add_x_forwarded_for) — client_ip reads the LAST hop, which is
    the address the proxy itself wrote, so a client-forged prefix is
    ignored either way.
    """
    return os.environ.get("TRUST_PROXY", "0") == "1"


def client_ip(request: Request) -> str:
    """Best-effort real client IP used for trust + rate-limit decisions.

    Forwarded headers (X-Forwarded-For / X-Real-IP) are honoured ONLY when
    both (a) TRUST_PROXY=1 and (b) the immediate TCP peer is itself a trusted
    private/loopback address — i.e. the reverse proxy. This second gate means
    that even if TRUST_PROXY is accidentally left on while the app is also
    reachable directly from the internet, a public attacker connecting
    straight to the socket cannot spoof its IP (its peer address is public, so
    its forged X-Forwarded-For is ignored and the real peer IP is used). A
    legitimate LAN client behind the proxy is unaffected: its real address
    arrives via X-Forwarded-For while the peer is the (private) proxy.
    Otherwise we use the socket peer address, which the client can't spoof.

    Within X-Forwarded-For we take the LAST hop, not the first. The shipped
    nginx.conf uses $proxy_add_x_forwarded_for, which APPENDS the peer to any
    incoming header instead of overwriting it — so the first hop is
    client-controlled ("X-Forwarded-For: 127.0.0.1" would spoof loopback
    trust and dodge per-IP rate limits), while the last hop is the value the
    trusted proxy itself appended: the real client address. With a proxy
    that overwrites the header there is only one hop and last == first, so
    this is safe for both proxy configurations. (Only a single trusted
    proxy is supported; a chain would need a trusted-hop count.)
    """
    peer = request.client.host if request.client else ""
    if _trust_proxy_enabled() and is_trusted_client_host(peer):
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[-1].strip()
        real = request.headers.get("x-real-ip")
        if real:
            return real.strip()
    return peer


TAILSCALE_IPV4_NET = ipaddress.ip_network("100.64.0.0/10")
TAILSCALE_IPV6_NET = ipaddress.ip_network("fd7a:115c:a1e0::/48")


def allowed_origins() -> List[str]:
    """The effective Origin allow-list, read from the environment on every call.

    Unlike the ``ALLOWED_ORIGINS`` import-time snapshot (kept for the CORS
    middleware, which Starlette bakes at ``add_middleware`` and therefore
    needs a restart to pick up changes), the per-request trust gate uses
    this live reader so an ``ALLOWED_ORIGINS`` change takes effect without
    a restart for actual request authorization. CORS *preflight*
    (OPTIONS) responses still use the startup snapshot — restart the
    backend after changing ALLOWED_ORIGINS to keep preflight in sync.
    """
    return parse_allowed_origins(os.environ.get("ALLOWED_ORIGINS"))


def is_trusted_origin(origin: Optional[str]) -> bool:
    if not origin:
        return False
    norm = origin.rstrip("/").lower()
    if norm in allowed_origins():
        return True
    if norm.endswith(".ts.net") or ".ts.net:" in norm:
        return True
    try:
        from urllib.parse import urlparse
        host = urlparse(norm).hostname
        if host and is_trusted_client_host(host):
            return True
    except Exception:
        pass
    return False


def is_trusted_client_host(client_host: Optional[str]) -> bool:
    if not client_host:
        return False

    normalized_host = client_host.strip().lower()
    if normalized_host in {"127.0.0.1", "::1", "localhost"}:
        return True
    if normalized_host.endswith(".ts.net"):
        return True

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return False

    return (
        address.is_loopback
        or address.is_private
        or (isinstance(address, ipaddress.IPv4Address) and address in TAILSCALE_IPV4_NET)
        or (isinstance(address, ipaddress.IPv6Address) and address in TAILSCALE_IPV6_NET)
    )


def require_trusted_config_request(request: Request) -> None:
    """Protect config + state-changing endpoints from cross-site browser access.

    Four layers, checked in order:

    0. Verified Supabase JWT — the public SaaS frontend authenticates with a
       Supabase session token (``Authorization: Bearer <jwt>``); when the JWT
       gate middleware has verified it, the request is trusted regardless of
       fetch metadata. This is the deliberate public path — everything below
       is the unchanged local-first path.
    1. ``Sec-Fetch-Site`` — a *forbidden* request header set by the browser and
       not writable from JavaScript. A value of ``cross-site`` means a
       different site initiated the request, so we reject outright. A value
       of ``same-site`` (e.g. https://tellagbe.com calling
       https://nugget-api.tellagbe.com, which share eTLD+1) falls through to
       the Origin allow-list check below — the header is unforgeable and a
       ``same-site`` value can only originate from the operator's own domains.
       This preserves the CSRF hole closure: a plain HTML ``<form>`` POST
       from an attacker's site sends ``cross-site`` and is rejected here,
       before it could fall through to the private-IP branch.
    2. ``Origin`` — when present it must match the allow-list.
    3. Private/loopback client IP — only reached for non-browser clients (curl,
       CLI scripts) that send neither ``Sec-Fetch-Site`` nor ``Origin``.
       Proxy-guarded: a loopback peer may be cloudflared forwarding public
       traffic, so the Host must also be local (or absent) — a public Host
       from a loopback peer is rejected.
    """
    if request_supabase_user(request):
        return

    sec_fetch_site = request.headers.get("sec-fetch-site")
    if sec_fetch_site == "cross-site":
        raise HTTPException(status_code=403, detail="Cross-site requests are not allowed.")
    if sec_fetch_site == "same-origin":
        return

    origin = request.headers.get("origin")
    if origin:
        if is_trusted_origin(origin):
            return
        raise HTTPException(status_code=403, detail="Origin not allowed for config access.")

    referer = request.headers.get("referer")
    if referer and is_trusted_origin(referer):
        return

    host = request.headers.get("host")
    if host and is_trusted_origin(f"http://{host}"):
        return

    client_host = client_ip(request)
    if is_trusted_client_host(client_host):
        # Proxy guard: a loopback/private peer may be a local reverse proxy
        # (cloudflared) forwarding PUBLIC internet traffic, not a local
        # client. The proxy always preserves the original Host header, so a
        # public Host arriving from a loopback peer is proxied public traffic
        # and must not inherit loopback trust — otherwise any internet client
        # could reach trusted endpoints with curl (no Origin/Sec-Fetch-Site).
        # Only requests addressed to a local interface (or with no Host at
        # all, i.e. direct pre-HTTP/1.1 local calls — a proxy always sets
        # Host for routing) keep the peer trust.
        if not host or is_trusted_origin(f"http://{host}"):
            return

    raise HTTPException(status_code=403, detail="Config access requires a trusted local origin.")


# --- Supabase JWT authentication (public SaaS frontend) ---------------------
# Lets the public Cloudflare Pages frontend call the laptop backend through
# the Cloudflare Tunnel. The frontend signs in with Supabase (Google OAuth)
# and sends the session JWT as ``Authorization: Bearer <jwt>`` on every
# /api request (see dashboard/src/lib/supabaseClient.js -> apiToken.js).
# A *valid* JWT (HS256, signed with the project's JWT secret, unexpired)
# marks the request trusted, bypassing the Sec-Fetch-Site cross-site
# rejection in require_trusted_config_request. Without a valid JWT the
# local-first behaviour is completely unchanged.
#
# The secret NEVER lives in code or chat: it is read from the
# SUPABASE_JWT_SECRET env var, or from data/supabase_jwt_secret.txt
# (one line, no trailing newline issues — we strip whitespace).


# Short-TTL cache for the file-backed JWT secret: verify_supabase_jwt runs
# on every /api request inside the async middleware, and a disk read per
# request is wasteful. 30s TTL keeps rotation responsive (a rotated secret
# takes effect within half a minute) while making the steady-state cost zero.
_JWT_SECRET_CACHE_TTL = 30.0
_jwt_secret_cache: Tuple[Optional[str], float] = (None, 0.0)


def _read_jwt_secret_file() -> Optional[str]:
    candidates = [
        os.path.join(os.getcwd(), "data", "supabase_jwt_secret.txt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "supabase_jwt_secret.txt"),
    ]
    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                value = handle.read().strip()
            if value:
                return value
        except OSError:
            continue
    return None


def supabase_jwt_secret() -> Optional[str]:
    """The Supabase project JWT secret, or None when SaaS auth is off.

    Env var wins (containers); otherwise data/supabase_jwt_secret.txt
    relative to the working directory (laptop: C:\\nugget) or the repo root.
    The file read is cached for 30s (see _JWT_SECRET_CACHE_TTL) so the
    per-request JWT verification doesn't hit disk every time; a rotation
    still takes effect within half a minute and tests can swap it freely
    (set SUPABASE_JWT_SECRET env to bypass the cache entirely).
    """
    env_value = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
    if env_value:
        return env_value
    global _jwt_secret_cache
    cached_value, cached_at = _jwt_secret_cache
    if time.monotonic() - cached_at < _JWT_SECRET_CACHE_TTL:
        return cached_value
    value = _read_jwt_secret_file()
    _jwt_secret_cache = (value, time.monotonic())
    return value


def _invalidate_jwt_secret_cache() -> None:
    """Force the next supabase_jwt_secret() call to re-read the file (tests)."""
    global _jwt_secret_cache
    _jwt_secret_cache = (None, 0.0)


def verify_supabase_jwt(token: str) -> Optional[Dict]:
    """Verify a Supabase session JWT. Returns its claims, or None.

    Strict: HS256 only (Supabase signs with the JWT secret), signature must
    verify, token must not be expired, and it must carry a subject (the
    Supabase user id). Any failure — bad signature, wrong algorithm, expired,
    missing secret — returns None (fail closed: the request falls back to the
    normal local-trust path instead of being trusted).
    """
    token = (token or "").strip()
    if not token:
        return None
    secret = supabase_jwt_secret()
    if not secret:
        return None
    try:
        import jwt as pyjwt
    except ImportError:
        return None
    try:
        claims = pyjwt.decode(
            token,
            secret,
            algorithms=["HS256"],
            # Signature + expiry + subject are the trust decision. We do NOT
            # verify `aud`: Supabase mints aud="authenticated" and the claim
            # carries no security weight here — anyone holding the project
            # secret (the only way to forge a signature) already wins.
            options={"require": ["exp", "sub"], "verify_aud": False},
            leeway=30,
        )
    except Exception:
        return None
    if not isinstance(claims, dict) or not claims.get("sub"):
        return None
    return claims


def request_supabase_user(request: Request) -> Optional[str]:
    """The verified Supabase user id for this request, or None.

    Set by the JWT gate middleware in app.py; the dependency below reads it.
    Defensive about ``request.state``: unit-test doubles may not provide it.
    """
    state = getattr(request, "state", None)
    if state is None:
        return None
    return getattr(state, "supabase_user", None)


def request_supabase_claims(request: Request) -> Optional[Dict]:
    """The verified Supabase JWT claims for this request, or None.

    Set alongside ``supabase_user`` by the JWT gate middleware in app.py.
    Lets ``get_current_user`` build a per-user identity (id + email for the
    admin allow-list) without verifying the token a second time.
    """
    state = getattr(request, "state", None)
    if state is None:
        return None
    claims = getattr(state, "supabase_claims", None)
    return claims if isinstance(claims, dict) else None


# --- optional API token (deliberate LAN deployments) ------------------------
# The trust model above treats every private-network peer as an authorized
# client, so CLIPPYME_BIND=0.0.0.0 extends config/state access to the whole
# LAN. CLIPPYME_API_TOKEN restores per-client auth for that case: when set,
# every /api request must also carry the shared secret. Unset (the default,
# loopback self-host) the check is a no-op and behavior is unchanged.


def configured_api_token() -> Optional[str]:
    """The shared secret from CLIPPYME_API_TOKEN, or None when auth is off.

    Read per-request (not cached at import) so tests and container restarts
    with a changed env behave predictably.
    """
    token = os.environ.get("CLIPPYME_API_TOKEN", "").strip()
    return token or None


def enforce_api_token(request: Request) -> None:
    """Raise HTTP 401 unless the request carries the configured token.

    Accepts either ``X-API-Token: <token>`` or ``Authorization: Bearer
    <token>``. Comparison is constant-time (hmac.compare_digest) so the token
    can't be recovered byte-by-byte via timing. No-op when no token is set.
    """
    expected = configured_api_token()
    if expected is None:
        return

    supplied = request.headers.get("x-api-token", "").strip()
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()

    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Valid API token required.")


# --- in-process rate limiting ----------------------------------------------
# Dependency-free per-client token bucket. Protects the compute-heavy endpoints
# (process/batch/publish) from a flood that would exhaust the job queue or
# Zernio quota. Status polling is intentionally NOT limited. State is in-memory
# (single-process self-host model); not shared across replicas.
_rate_state: Dict[Tuple[str, str], Tuple[float, float]] = {}
# Hard cap on tracked buckets so a flood of unique client IPs can't grow the
# dict without bound (memory-exhaustion DoS). When exceeded we evict the
# entries that are already refilled to capacity (idle clients) first.
_RATE_STATE_MAX = int(os.environ.get("RATE_LIMIT_MAX_BUCKETS", "10000"))


def _evict_rate_state(now: float) -> None:
    """Drop fully-refilled (idle) buckets when the table grows too large.

    Each entry's idle check uses its OWN bucket's capacity/refill (buckets
    have different capacities — process=20, publish/compose=30). Judging a
    process entry against the current request's capacity would either never
    see it as refilled or evict it while still draining.
    """
    if len(_rate_state) < _RATE_STATE_MAX:
        return
    stale = [
        k for k, (tokens, last) in _rate_state.items()
        if min(capacity_for(k), tokens + max(0.0, now - last) * refill_for(k))
        >= capacity_for(k)
    ]
    for k in stale:
        _rate_state.pop(k, None)
    # If still over budget (all buckets active), clear the oldest half.
    if len(_rate_state) >= _RATE_STATE_MAX:
        for k in sorted(_rate_state, key=lambda kk: _rate_state[kk][1])[: _RATE_STATE_MAX // 2]:
            _rate_state.pop(k, None)


# Per-bucket refill rates + capacities so eviction can recompute "is this
# idle" correctly for every entry, not just the current request's bucket.
_bucket_refill: Dict[str, float] = {}
_bucket_capacity: Dict[str, float] = {}


def refill_for(key: Tuple[str, str]) -> float:
    return _bucket_refill.get(key[0], 1.0)


def capacity_for(key: Tuple[str, str]) -> float:
    return _bucket_capacity.get(key[0], 1.0)


def _rate_limit_allow(key: Tuple[str, str], capacity: float, refill_per_sec: float, now: float) -> Optional[float]:
    """Token-bucket check.

    Returns None when a token was available (and consumed); otherwise the
    seconds until the next token becomes available (for Retry-After).
    """
    _bucket_refill[key[0]] = refill_per_sec
    _bucket_capacity[key[0]] = capacity
    _evict_rate_state(now)
    tokens, last = _rate_state.get(key, (capacity, now))
    tokens = min(capacity, tokens + max(0.0, now - last) * refill_per_sec)
    if tokens < 1.0:
        _rate_state[key] = (tokens, now)
        if refill_per_sec > 0:
            return max(0.0, (1.0 - tokens) / refill_per_sec)
        return 60.0
    _rate_state[key] = (tokens - 1.0, now)
    return None


def enforce_rate_limit(request: Request, bucket: str, capacity: float, refill_per_sec: float) -> None:
    """Raise HTTP 429 when the per-client bucket is empty.

    Disabled by setting RATE_LIMIT_ENABLED=0 (e.g. for load tests). Keyed by
    client IP + bucket name so different endpoints don't share a budget.
    """
    if os.environ.get("RATE_LIMIT_ENABLED", "1") != "1":
        return
    # Behind the Cloudflare Tunnel every public client arrives via cloudflared
    # on loopback, so IP-keying would put ALL SaaS users in one shared bucket
    # and they'd 429 each other. Verified JWT users get their own per-user
    # bucket; everyone else keeps the existing IP-keyed behaviour.
    jwt_user = request_supabase_user(request)
    client_key = f"jwt:{jwt_user}" if jwt_user else (client_ip(request) or "unknown")
    retry_after = _rate_limit_allow((bucket, client_key), capacity, refill_per_sec, time.monotonic())
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down and retry shortly.",
            headers={"Retry-After": str(max(1, int(retry_after) + 1))},
        )
