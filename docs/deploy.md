# Deploy rule (Phase A, 2026-10-07)

**Production deploys come only from a merged or tagged commit.** Never deploy
a working tree, a detached hotfix commit, or an unpushed branch tip.

- Backend (`:8000`) and frontend (`nugget-chat` Pages project) releases are
  tagged `prod-YYYYMMDD-<slug>` on the release commit, e.g.
  `prod-20261007-slice1`.
- Push the tag with the release: `git push myrepo tag prod-20261007-slice1`
  (`gh-push.py` pushes branches; tags go explicitly).
- A deploy is not done until: commit pushed, tag pushed, service restarted
  (backend) or Pages deployment created (frontend), and the live artifact
  verified (health check + bundle grep).
- The tag is the source of truth for "what is live". `git describe` on the
  laptop checkout must name the tag before any restart.

Release history:
- `prod-20261007-slice1` -> `425cc04` — Slice 1 (transcript editor, stable
  word IDs, serialized edit queue, English-only Gemini output).

## Auth env contract (2026-10-07 hardening)

The API authenticates with Supabase **ES256 (asymmetric) JWTs verified against
the project's JWKS**. There is no JWT secret — `SUPABASE_JWT_SECRET` must not
exist in env or code (the legacy HS256 dashboard secret is useless even if
leaked: HS256 is never accepted).

Required env (laptop, production):
- `ENV=production`
- `AUTH_ENABLED=1`
- `SUPABASE_URL=https://frdlpogmqnozhgyrrhgf.supabase.co`
- `ALLOWED_USER_IDS=<comma-separated Supabase sub values>` — valid JWTs not on
  the list get 403. An empty/unset list denies everyone in every environment
  (fail closed); the explicit dev bypass is the only exception.
- `ADMIN_USER_IDS=<comma-separated Supabase sub values>` — admin privileges.
  No email-based admin exists anymore.

Optional:
- `CLIPPYME_API_TOKEN` — accepted ONLY for non-browser callers (no `Origin`
  header). Browsers must use JWT. Never put it in the frontend bundle.
- `ADMIN_SECRET_KEY` — `x-admin-secret` header bypass (ops use).
- `TRUST_PROXY` — leave `0`/unset. Enabling `1` is blocked until the proving
  tests pass (forged Referer/Host, XFF spoofing, public-IP direct).
- `LOG_PEER_ADDRESS=1` — one-time, off-by-default debug log of the socket peer
  address as seen by `client_ip()`. Logs once per process at INFO level, then
  never again (no log spam). Intended for diagnosing trust decisions (e.g. why
  a tunneled request's peer is not loopback). Leave off in production.

Dev:
- `AUTH_DISABLED_DEV=1` — explicit dev bypass (local default admin). Rejected
  at startup when `ENV=production`. An explicit `AUTH_ENABLED=1` always wins
  over the bypass. The bypass additionally refuses any request carrying
  `CF-Connecting-IP`/`CF-Ray` headers or a non-local Host header, so a
  Cloudflare-originated request can never gain admin through it.
- Production startup refuses to boot unless `AUTH_ENABLED=1` + `SUPABASE_URL`
  are set AND the JWKS fetch succeeds (one boot fetch; failure = no start).

Per-user rate limits (authenticated LLM/spend routes): studio 10–20/hr,
UGC research 10/hr + scripts 20/hr, highlights plan 10/hr / render 5/hr /
generate-all 5/hr / apply-edit 10/hr, dubbing 5/hr. See `security.py`.

## Structured access log

Every request logs one JSON line to `logs/access.jsonl` with exactly these
fields: `timestamp` (UTC ISO-8601), `method`, `path`, `status`,
`user_id` (from the auth context, `null` if unauthenticated),
`cf_connecting_ip` (the `CF-Connecting-IP` header, `null` if absent).

NEVER logged: request/response bodies, headers (other than
`CF-Connecting-IP`), tokens, or API keys.

Retention: the log rotates daily at midnight and keeps 30 days
(`TimedRotatingFileHandler`, `backupCount=30`). Older files are deleted
automatically.

## Admin allow-list semantics

- `ADMIN_USER_IDS` (comma-separated Supabase `sub` values) grants admin.
- **Admin implies access**: every admin ID is automatically allowed to call
  the API. Admins do NOT need a separate `ALLOWED_USER_IDS` entry.
  (`get_current_user` unions `ADMIN_USER_IDS` into the effective allow-list.)
- First-boot: setting ONLY `ADMIN_USER_IDS` (with `ALLOWED_USER_IDS`
  empty/unset) admits the admin and denies everyone else (403
  `NOT_ALLOWLISTED`).
- An empty/unset `ALLOWED_USER_IDS` denies everyone except admins
  (fail closed in every environment; the explicit `AUTH_DISABLED_DEV=1`
  dev bypass is the only exception, and it refuses Cloudflare-originated
  requests).
