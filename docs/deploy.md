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
