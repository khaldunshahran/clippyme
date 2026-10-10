# nugget-hyperframes-worker (Task 18 pilot)

Isolated local Hyperframes render worker. Pinned official packages only:

| package | version | integrity (sha512) |
|---|---|---|
| `@hyperframes/engine` | 0.8.145 | `sha512-ArbLkMpWJyJdP3VujAc10pcbTcFSVQ3fMQYszqDMU4AGK2T2JxgWl+aXdaiLM9Pw5EPZNC6BDZ7xelmBX+1SXQ==` |
| `@hyperframes/core` | 0.8.145 | `sha512-3Wh5GIzCo5Sn7IPs+1IilqEcZqFm2opw3QPpOVvXWtkA5uTQSAok3AN9hmC0ULtWvufAVjfa6ZA6ZsE2CDCUjw==` |
| `@hyperframes/player` | 0.8.145 | `sha512-jMLgaXhvXsZIOWRmJmKQ8e2laOMUahxY9gGLkFw48pBpzKSu6LfF+iGkg+yIrdyEKm76eIdzTGl8+NhxcqJBmA==` |

License: Apache-2.0 (see `node_modules/@hyperframes/engine/LICENSE`).

## Setup

```powershell
cd hyperframes-worker
npm ci            # installs from package-lock.json (pinned)
# Chromium (chrome-headless-shell) downloads into .puppeteer-cache/ inside
# this directory — never the user's browser profiles, never global:
$env:PUPPETEER_CACHE_DIR = "$PWD\.puppeteer-cache"
node node_modules/puppeteer/lib/puppeteer/node/cli.js browsers install chrome-headless-shell
```

Requirements: Node.js >= 22, FFmpeg on PATH (laptop: FFmpeg 8.1 full build).

## Run

```powershell
$env:PUPPETEER_CACHE_DIR = "$PWD\.puppeteer-cache"
# The engine resolves its browser via HYPERFRAMES_BROWSER_PATH (not
# PUPPETEER_CACHE_DIR); point it at the isolated chrome-headless-shell.
# (The FastAPI spawner sets this automatically via worker_client.)
$env:HYPERFRAMES_BROWSER_PATH = "$PWD\.puppeteer-cache\chrome-headless-shell\win64-155.0.8059.39\chrome-headless-shell-win64\chrome-headless-shell.exe"
node render.js --manifest <manifest.json> --out <output.mp4> --workdir <dir>
```

The manifest is written by the typed compiler (`src/clippyme/finishing/compile.py`);
`render.js` validates it (bounds: duration ≤ 180 s, frames ≤ 10800, JSON ≤ 1 MB).

Exit codes: `0` ok · `2` bad input · `3` busy (another job holds the lock) ·
`4` deadline · `5` cancelled · `6` render/encode/mux failure · `7` output validation failure.

`<workdir>/result.json` carries measured metrics on every terminal path:
`runtime_sec`, `frames_captured`, `peak_memory_mb` (Chrome RSS sampler),
`temp_disk_peak_bytes`, `capture_mode`, `encode_ms`, `validated` (streams/dims/duration).

One job at a time (`.render.lock`, stale-lock takeover only for dead pids).
Cancellation: create `<workdir>/CANCEL`. Temp frames are always cleaned up;
the output is installed atomically only after ffprobe validation.
