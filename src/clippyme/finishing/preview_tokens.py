"""Task 18 review-01 — short-lived HMAC preview tokens (contract 2 backend).

Browser <iframe>/<video> navigation cannot attach the apiFetch Bearer
<redacted> so the authed composition asset route also accepts a bounded
preview token::

    POST /api/hyperframes/preview-token/{job_id}/{compile_id}
        (authed via get_current_user, job ownership checked)
        -> {"token": ..., "expires_in": 900}

    GET /api/hyperframes/composition/{job_id}/{compile_id}/{asset}?token=...

The token is an HMAC-SHA256 signature over a JSON payload
``{"v": 1, "job_id": ..., "compile_id": ..., "exp": ...}``. The signing key
is derived (domain-separated SHA256) from the app's EXISTING shared secret
env — ``CLIPPYME_API_TOKEN`` (the LAN shared-secret auth, see api/app.py)
falling back to ``ADMIN_SECRET_KEY``. No new credential store is invented
and no Supabase JWT ever appears in a URL. When neither secret is
configured (dev/test), a process-local random key is used so minting still
works; such tokens verify only within this process (logged once).

Rejected: bad signature, expired token, job/compile scope mismatch, unknown
compile (checked by the route). Tokens are single-compile scoped with a
short (~15 min) expiry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time

logger = logging.getLogger(__name__)

TOKEN_TTL_SEC = 15 * 60  # ~15 minutes, per contract
_TOKEN_VERSION = 1
_DOMAIN = b"nugget-hyperframes-preview/v1:"

_process_key: bytes | None = None
_process_key_warned = False


def _signing_key() -> bytes:
    """Domain-separated HMAC key from an existing app secret env."""
    secret = os.environ.get("CLIPPYME_API_TOKEN", "").strip()
    if not secret:
        secret = os.environ.get("ADMIN_SECRET_KEY", "").strip()
    if secret:
        return hashlib.sha256(_DOMAIN + secret.encode("utf-8")).digest()
    global _process_key, _process_key_warned
    if _process_key is None:
        _process_key = secrets.token_bytes(32)
    if not _process_key_warned:
        _process_key_warned = True
        logger.warning(
            "hyperframes preview tokens: neither CLIPPYME_API_TOKEN nor "
            "ADMIN_SECRET_KEY is configured; using a process-local signing "
            "key (tokens verify only within this process)")
    return hashlib.sha256(_DOMAIN + _process_key).digest()


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("ascii"))


def mint_preview_token(job_id: str, compile_id: str,
                       ttl_sec: int = TOKEN_TTL_SEC) -> tuple[str, int]:
    """Mint a single-compile preview token. Returns (token, expires_in)."""
    exp = int(time.time()) + int(ttl_sec)
    payload = {"v": _TOKEN_VERSION, "job_id": job_id,
               "compile_id": compile_id, "exp": exp}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sig = hmac.new(_signing_key(), raw, hashlib.sha256).digest()
    return _b64e(raw) + "." + _b64e(sig), int(ttl_sec)


class PreviewTokenError(Exception):
    """A preview token was rejected (bad signature/expired/scope)."""


def verify_preview_token(token: str, *, job_id: str,
                         compile_id: str) -> dict:
    """Verify a preview token against the request path scope.

    Raises PreviewTokenError on: malformed token, bad signature, expired
    token, or job_id/compile_id scope mismatch. Returns the payload dict.
    """
    if not token or not isinstance(token, str):
        raise PreviewTokenError("missing preview token")
    parts = token.split(".")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise PreviewTokenError("malformed preview token")
    try:
        raw, sig = _b64d(parts[0]), _b64d(parts[1])
    except Exception as exc:
        raise PreviewTokenError("malformed preview token") from exc
    expected = hmac.new(_signing_key(), raw, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, sig):
        raise PreviewTokenError("invalid preview token signature")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise PreviewTokenError("malformed preview token payload") from exc
    if not isinstance(payload, dict) or payload.get("v") != _TOKEN_VERSION:
        raise PreviewTokenError("unsupported preview token version")
    exp = payload.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool) \
            or exp < int(time.time()):
        raise PreviewTokenError("preview token expired")
    if payload.get("job_id") != job_id \
            or payload.get("compile_id") != compile_id:
        raise PreviewTokenError("preview token scope mismatch")
    return payload
