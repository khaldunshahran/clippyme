"""Task 18 — clean-base audit and materialization (contract 3).

The Hyperframes pilot renders graphics OVER preserved accepted footage, so it
needs a base video that provably carries no baked captions/hooks.

Audit result (recorded 2026-10-10, see RESULT.md):
- ``project.reframed_base`` lineage points at the pipeline's ``clip_final``,
  which is the fully composed clip — captions/hooks may be baked in and the
  lineage record carries no graphics-free provenance. It is therefore NOT
  reused as a Hyperframes base.
- The accepted ``_materialize_timeline`` path (project_render) produces the
  cut/cropped/concatenated timeline with audio and NO graphics layers — it is
  the pre-compose footage by construction, so it is a verified clean base.
- Projects carrying a Task 13 ``framing`` trajectory without an eligible
  lineage base cannot be safely reconstructed (Task 13 review-01 rule):
  fail clearly instead of publishing a static crop.

This module never overwrites originals, sidecars, user projects, or
transactional snapshots: the clean base is a fresh ``.hf_cleanbase_*`` file
per (clip, project version) inside the job dir.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import os

logger = logging.getLogger(__name__)


class CleanBaseError(Exception):
    """Base class for clean-base failures (fail clearly, never guess)."""


class CleanBaseUnavailable(CleanBaseError):
    """No safe clean base exists for this project (legacy path offered)."""


class CleanBaseInvalid(CleanBaseError):
    """The materialized base failed media validation."""


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_base_filename(clip_index: int, version: int) -> str:
    return f".hf_cleanbase_{clip_index}_v{version}.mp4"


async def resolve_clean_base(project, job_dir: str, clip_index: int) -> dict:
    """Return ``{"path", "sha256", "provenance"}`` for a verified clean base.

    Raises:
        CleanBaseUnavailable: framing trajectory without a safe base, or any
            case where reusing footage would invent geometry.
        CleanBaseInvalid: the materialized base failed media validation.
    """
    from clippyme.domain import project_render as pr

    # Task 13 gate, mirrored: a project carrying a framing trajectory but no
    # eligible lineage base has no validated pixel base to replay onto.
    # Dynamic replay is not implemented in this pilot — fail clearly.
    if getattr(project, "framing", None) is not None:
        raise CleanBaseUnavailable(
            "project carries a framing trajectory with no verified clean base; "
            "Hyperframes cannot safely reconstruct the camera plan in this "
            "pilot — use the legacy render path"
        )

    # Lineage audit: reframed_base points at clip_final (post-graphics), and
    # the lineage record has no graphics-free provenance marker, so it is
    # never reused here — even when check_lineage_eligible passes for the
    # accepted renderer.
    lineage = getattr(project, "reframed_base", None)
    if lineage is not None:
        logger.info(
            "hyperframes clean-base: lineage base %r present but not verifiably "
            "graphics-free (clip_final provenance); materializing clean timeline",
            getattr(lineage, "base_ref", "?"),
        )

    # Verified clean: the accepted pre-compose timeline materialization.
    timeline_path, timeline_duration = await pr._materialize_timeline(
        project, job_dir, clip_index
    )
    try:
        probe = await asyncio.to_thread(pr.probe_media, timeline_path)
    except Exception as exc:
        raise CleanBaseInvalid(f"cannot probe materialized base: {exc}") from exc

    def _finite_pos(v):
        return isinstance(v, (int, float)) and math.isfinite(v) and v > 0

    if probe.get("has_video") is not True:
        raise CleanBaseInvalid("materialized base has no video stream")
    bw, bh = probe.get("width"), probe.get("height")
    bdur, bfps = probe.get("duration"), probe.get("fps")
    if not _finite_pos(bdur):
        raise CleanBaseInvalid(f"materialized base duration invalid: {bdur!r}")
    if not _finite_pos(bfps):
        raise CleanBaseInvalid(f"materialized base FPS invalid: {bfps!r}")
    rw, rh = project.render.width, project.render.height
    if not (bw == rw and bh == rh):
        raise CleanBaseInvalid(
            f"materialized base dimensions {bw}x{bh} != project render {rw}x{rh}"
        )
    if abs(float(bfps) - float(project.render.fps)) > 0.1:
        raise CleanBaseInvalid(
            f"materialized base FPS {bfps} mismatches project {project.render.fps}"
        )
    if abs(float(bdur) - float(timeline_duration)) > max(0.5, timeline_duration * 0.05):
        raise CleanBaseInvalid(
            f"materialized base duration {bdur} mismatches timeline {timeline_duration}"
        )

    # Publish under an explicit hyperframes name (fresh file per version).
    dest = os.path.join(job_dir, clean_base_filename(clip_index, project.version))
    if os.path.abspath(timeline_path) != os.path.abspath(dest):
        with open(timeline_path, "rb") as src, open(dest, "wb") as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(chunk)
    digest = await asyncio.to_thread(_sha256_file, dest)
    return {
        "path": dest,
        "sha256": digest,
        "provenance": "materialized-pre-compose-timeline",
        "duration": float(timeline_duration),
        "width": int(bw),
        "height": int(bh),
        "fps": float(bfps),
    }
