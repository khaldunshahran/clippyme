"""Task 18 — local worker invocation (contract 6).

Spawns the isolated ``hyperframes-worker/render.js`` as a subprocess with a
finite deadline. One job at a time is enforced by the worker's own lockfile;
a BUSY exit maps to a retryable WorkerBusy error. Cancellation is
cooperative: the caller creates ``<workdir>/CANCEL`` (or calls
``request_cancel``), the worker aborts, cleans temp state, and exits 5.

No global client, no cross-job diagnostics: every invocation owns its
workdir, its browser session (pool disabled in the worker), and its result.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Worker exit codes (must match render.js).
EXIT_BUSY = 3
EXIT_DEADLINE = 4
EXIT_CANCELLED = 5

_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "hyperframes-worker", "render.js")


class WorkerError(Exception):
    """Base worker failure."""


class WorkerBusy(WorkerError):
    """Another render holds the worker lock (retryable)."""


class WorkerCancelled(WorkerError):
    """The render was cancelled."""


class WorkerDeadline(WorkerError):
    """The render exceeded its finite deadline."""


class WorkerUnavailable(WorkerError):
    """Node, the worker script, or Chromium is missing."""


@dataclass
class WorkerResult:
    status: str  # rendered | failed | cancelled | deadline
    output: str | None = None
    metrics: dict = field(default_factory=dict)
    cache_hit: bool = False


def worker_paths() -> dict:
    """Resolve worker script + Chromium; raise WorkerUnavailable if missing."""
    script = os.path.abspath(_WORKER_SCRIPT)
    if not os.path.isfile(script):
        raise WorkerUnavailable(f"worker script missing: {script}")
    worker_dir = os.path.dirname(script)
    cache_dir = os.path.join(worker_dir, ".puppeteer-cache", "chrome-headless-shell")
    chromium = None
    if os.path.isdir(cache_dir):
        for ver in sorted(os.listdir(cache_dir)):
            exe = os.path.join(cache_dir, ver,
                               "chrome-headless-shell-win64",
                               "chrome-headless-shell.exe")
            if os.path.isfile(exe):
                chromium = exe
    node_modules = os.path.join(worker_dir, "node_modules", "@hyperframes",
                                "engine", "package.json")
    if not os.path.isfile(node_modules):
        raise WorkerUnavailable("worker dependencies not installed (npm ci)")
    return {"script": script, "worker_dir": worker_dir, "chromium": chromium,
            "engine_installed": True}


def check_available() -> dict:
    """Honest capability probe: what the worker can actually do right now."""
    try:
        paths = worker_paths()
    except WorkerUnavailable as exc:
        return {"available": False, "reason": str(exc)}
    node = _find_node()
    if node is None:
        return {"available": False, "reason": "node not found on PATH"}
    if paths["chromium"] is None:
        return {"available": False,
                "reason": "chromium not downloaded (see hyperframes-worker/README.md)"}
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return {"available": False, "reason": "ffmpeg not found on PATH"}
    return {"available": True, "reason": "",
            "node": node, "chromium": paths["chromium"], "ffmpeg": ffmpeg,
            "engine": "0.8.145"}


def _find_node() -> str | None:
    for name in ("node", "node.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            c = os.path.join(d, name)
            if os.path.isfile(c):
                return c
    return None


def _find_ffmpeg() -> str | None:
    for name in ("ffmpeg", "ffmpeg.exe"):
        for d in os.environ.get("PATH", "").split(os.pathsep):
            c = os.path.join(d, name)
            if os.path.isfile(c):
                return c
    return None


def request_cancel(workdir: str) -> None:
    """Ask a running worker to abort (cooperative cancellation)."""
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "CANCEL"), "w", encoding="utf-8") as f:
        f.write("cancel requested\n")


def run_worker_render(*, manifest_path: str, out_path: str, workdir: str,
                      deadline_sec: int = 1200,
                      extra_env: dict | None = None) -> WorkerResult:
    """Run one bounded worker render; raise WorkerError subclasses on failure.

    The subprocess gets its own deadline (deadline_sec + margin): the worker
    enforces the finite deadline itself; the margin only guards a hung child.
    """
    paths = worker_paths()  # raises WorkerUnavailable
    node = _find_node()
    if node is None:
        raise WorkerUnavailable("node not found on PATH")
    chromium = paths.get("chromium")
    if not chromium:
        raise WorkerUnavailable(
            "chromium not downloaded into the isolated workspace "
            "(see hyperframes-worker/README.md)")

    os.makedirs(workdir, exist_ok=True)
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    manifest["deadline_sec"] = int(deadline_sec)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    env = dict(os.environ)
    env["PUPPETEER_CACHE_DIR"] = os.path.join(paths["worker_dir"], ".puppeteer-cache")
    # The engine resolves its browser via HYPERFRAMES_BROWSER_PATH (or its
    # own cache dirs) — never PUPPETEER_CACHE_DIR — and puppeteer's fallback
    # with headless:true wants the full chrome build. Point the engine at
    # our isolated chrome-headless-shell explicitly; the binary stays inside
    # the workspace, never a user profile or global install.
    env["HYPERFRAMES_BROWSER_PATH"] = chromium
    if extra_env:
        env.update(extra_env)

    cmd = [node, paths["script"], "--manifest", manifest_path,
           "--out", out_path, "--workdir", workdir]
    logger.info("hyperframes worker: spawning %s", " ".join(cmd))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=deadline_sec + 120)
    except subprocess.TimeoutExpired as exc:
        raise WorkerDeadline(
            f"worker supervisor timeout after {deadline_sec + 120}s") from exc
    elapsed = time.monotonic() - t0

    result_path = os.path.join(workdir, "result.json")
    metrics: dict = {}
    if os.path.isfile(result_path):
        try:
            metrics = json.load(open(result_path, encoding="utf-8"))
        except (OSError, ValueError):
            pass
    metrics.setdefault("supervisor_elapsed_sec", round(elapsed, 2))

    rc = proc.returncode
    if rc == 0:
        status = metrics.get("status", "rendered")
        return WorkerResult(status=status, output=out_path, metrics=metrics)
    if rc == EXIT_BUSY:
        raise WorkerBusy(metrics.get("error") or "worker busy")
    if rc == EXIT_CANCELLED:
        raise WorkerCancelled(metrics.get("error") or "render cancelled")
    if rc == EXIT_DEADLINE:
        raise WorkerDeadline(metrics.get("error") or "render deadline exceeded")
    raise WorkerError(
        f"worker failed (exit {rc}): "
        f"{metrics.get('error') or (proc.stderr or '')[-500:]}")
