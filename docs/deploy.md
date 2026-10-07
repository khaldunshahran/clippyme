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
- `prod-20261007-slice1` -> `425cc04` ƒ?" Slice 1 (transcript editor, stable
  word IDs, serialized edit queue, English-only Gemini output).

## Auth env contract (2026-10-07 hardening)

The API authenticates with Supabase **ES256 (asymmetric) JWTs verified against
the project's JWKS**. There is no JWT secret ƒ?" `SUPABASE_JWT_SECRET` must not
exist in env or code (the legacy HS256 dashboard secret is useless even if
leaked: HS256 is never accepted).

Required env (laptop, production):
- `ENV=production`
- `AUTH_ENABLED=1`
- `SUPABASE_URL=https://frdlpogmqnozhgyrrhgf.supabase.co`
- `ALLOWED_USER_IDS=<comma-separated Supabase sub values>` ƒ?" valid JWTs not on
  the list get 403. An empty/unset list denies everyone in every environment
  (fail closed); the explicit dev bypass is the only exception.
- `ADMIN_USER_IDS=<comma-separated Supabase sub values>` ƒ?" admin privileges.
  No email-based admin exists anymore.

Optional:
- `CLIPPYME_API_TOKEN` ƒ?" accepted ONLY for non-browser callers (no `Origin`
  header). Browsers must use JWT. Never put it in the frontend bundle.
- `ADMIN_SECRET_KEY` ƒ?" `x-admin-secret` header bypass (ops use).
- `TRUST_PROXY` ƒ?" leave `0`/unset. Enabling `1` is blocked until the proving
  tests pass (forged Referer/Host, XFF spoofing, public-IP direct).
- `LOG_PEER_ADDRESS=1` ƒ?" one-time, off-by-default debug log of the socket peer
  address as seen by `client_ip()`. Logs once per process at INFO level, then
  never again (no log spam). Intended for diagnosing trust decisions (e.g. why
  a tunneled request's peer is not loopback). Leave off in production.

Dev:
- `AUTH_DISABLED_DEV=1` ƒ?" explicit dev bypass (local default admin). Rejected
  at startup when `ENV=production`. An explicit `AUTH_ENABLED=1` always wins
  over the bypass. The bypass additionally refuses any request carrying
  `CF-Connecting-IP`/`CF-Ray` headers or a non-local Host header, so a
  Cloudflare-originated request can never gain admin through it.
- Production startup refuses to boot unless `AUTH_ENABLED=1` + `SUPABASE_URL`
  are set AND the JWKS fetch succeeds (one boot fetch; failure = no start).

Per-user rate limits (authenticated LLM/spend routes): studio 10ƒ?"20/hr,
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
## Config-integrity check methodology (2026-10-07)

**Question:** During the public-exposure window (2026-10-03 ~22:30 EEST tunnel ingress
added A›ƒ?ÿƒ?T 2026-10-07 00:45 EEST ingress removed), did anyone modify backend config,
jobs, or secrets without attribution?

**Method** (all timestamps Europe/Helsinki unless noted):

1. **File mtimes vs. known actor actions.** For each mutable file, compared
   `LastWriteTime` against the timeline of actions by known actors (assistant,
   Khaldun):
   - `data/config.json`: 2026-10-07 01:53:02 A›ƒ?ÿƒ?T 02:23:51 (assistant's auth-test writes; expected)
   - `data/jobs_journal.json`: 2026-10-07 01:53:13 A›ƒ?ÿƒ?T 02:23:13 (assistant's auth-test writes; expected)
   - `.env`: 2026-10-03 22:30:36 A›ƒ?ÿƒ?T **2026-10-07 02:20:37** (Khaldun set 5 keys: ALLOWED_USER_IDS,
     ADMIN_USER_IDS, ENV, AUTH_ENABLED, SUPABASE_URL; confirmed by Khaldun)
   - `C:\Users\khald\.cloudflared\config.yml`: 00:41:20 A›ƒ?ÿƒ?T **02:25:34** (Khaldun re-enabled
     the API ingress himself; confirmed by Khaldun)

2. **Git baseline.** `git status --porcelain` on the repo showed only expected
   untracked junk (`dashboard-chat/dogfood/`, `logs/`, `*.bat`); no modified
   tracked files outside the auth work. `git diff HEAD` for `src/` matched the
   pushed commits.

3. **Backup comparison.** `.bak` files (e.g., `config.yml.bak-api-shutdown-20261007`)
   preserved pre-change state; diffed to confirm only the intended ingress removal.

4. **Log review.** Uvicorn access logs lack client IPs (cannot reconstruct who
   probed); tunnel logs have no per-request data. Live probes during the window
   returned 403 on sensitive routes (`/api/config`, `/api/process`), 200 on
   `/api/status/{fake}` (metadata only, no secrets). Config masking verified
   (keys masked in API responses).

**Conclusion:** Every mtime change in the window maps to a known actor action.
No unattributed modifications to config, jobs, or secrets were found. The
`logs/access.jsonl` pollution by pytest TestClient runs (`user_id: "default_user"`)
is a test-hygiene issue (fixed: `ACCESS_LOG_PATH` env override), not a breach.

**Residual gaps:** Cannot definitively rule out passive probing (no IP logs);
the pre-auth code trusted an allow-listed `Referer` header (step 5, since removed),
but probes sent no `Referer`, leaving that path untested live.


