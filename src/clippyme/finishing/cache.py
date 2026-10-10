"""Task 18 — render cache (contract 6).

The cache key binds: clean-base identity + acknowledged project digest/version
+ template/font/renderer/asset versions. No cache reuse across changed inputs:
any change to the project, template, fonts, renderer, or base invalidates.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil


def cache_key(*, base_sha256: str, project_digest: str, project_version: int,
              template_id: str, template_version: int, params: dict,
              font_shas: list[str], renderer: dict, width: int, height: int,
              fps: int) -> str:
    """Deterministic cache key over every render input."""
    canonical = json.dumps({
        "base_sha256": base_sha256,
        "project_digest": project_digest,
        "project_version": int(project_version),
        "template_id": template_id,
        "template_version": int(template_version),
        "params": params,
        "font_shas": sorted(font_shas),
        "renderer": renderer,
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_dir(job_dir: str) -> str:
    return os.path.join(os.path.abspath(job_dir), ".hf", "cache")


def lookup(job_dir: str, key: str) -> str | None:
    """Return the cached output path, or None on miss."""
    candidate = os.path.join(cache_dir(job_dir), key + ".mp4")
    if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
        return candidate
    return None


def store(job_dir: str, key: str, src_path: str) -> str:
    """Atomically install src_path into the cache; return the cached path."""
    cdir = cache_dir(job_dir)
    os.makedirs(cdir, exist_ok=True)
    dest = os.path.join(cdir, key + ".mp4")
    tmp = dest + ".part"
    shutil.copy2(src_path, tmp)
    os.replace(tmp, dest)
    return dest


def invalidate_job(job_dir: str) -> int:
    """Remove all cached renders for a job; return the count removed."""
    cdir = cache_dir(job_dir)
    removed = 0
    if os.path.isdir(cdir):
        for name in os.listdir(cdir):
            if name.endswith(".mp4"):
                try:
                    os.remove(os.path.join(cdir, name))
                    removed += 1
                except OSError:
                    pass
    return removed
