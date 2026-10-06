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
  the list get 403. Empty in production denies everyone (fail closed).
- `ADMIN_USER_IDS=<comma-separated Supabase sub values>` — admin privileges.
  No email-based admin exists anymore.

Optional:
- `CLIPPYME_API_TOKEN` — accepted ONLY for non-browser callers (no `Origin`
  header). Browsers must use JWT. Never put it in the frontend bundle.
- `ADMIN_SECRET_KEY` — `x-admin-secret` header bypass (ops use).
- `TRUST_PROXY` — leave `0`/unset. Enabling `1` is blocked until the proving
  tests pass (forged Referer/Host, XFF spoofing, public-IP direct).

Dev:
- `AUTH_DISABLED_DEV=1` — explicit dev bypass (local default admin). Rejected
  at startup when `ENV=production`. An explicit `AUTH_ENABLED=1` always wins
  over the bypass.
- Production startup refuses to boot unless `AUTH_ENABLED=1` + `SUPABASE_URL`
  are set AND the JWKS fetch succeeds (one boot fetch; failure = no start).

Per-user rate limits (authenticated LLM/spend routes): studio 10–20/hr,
UGC research 10/hr + scripts 20/hr, highlights plan 10/hr / render 5/hr /
generate-all 5/hr / apply-edit 10/hr, dubbing 5/hr. See `security.py`.
