"""Task 18 — narrow Hyperframes finishing API (pilot).

Endpoints (all authed, job-ownership checked, job-relative paths only):

- GET  /api/hyperframes/capability
    Honest capability state: worker availability, pinned renderer versions,
    the three reviewed templates, and the default-OFF rollout flag.
- POST /api/hyperframes/compile/{job_id}/{clip_index}
    Compile the frozen accepted project snapshot (the acknowledged version)
    into a deterministic composition. Returns the preview URL (served from
    the same authed composition endpoint the export renders) and the
    preview/export identity digest.
- GET  /api/hyperframes/composition/{job_id}/{compile_id}/{asset}
    Serve the compiled composition and its allowlisted local assets
    (index.html, base.mp4, template fonts). Traversal/symlink-safe.
- POST /api/hyperframes/render/{job_id}/{clip_index}
    Consume the frozen accepted project snapshot through the current
    compose route/version contract and dispatch ONE bounded local worker
    render. Cache-aware; never reuses cache across changed inputs.
- GET  /api/hyperframes/render/{render_id}
    Render job state: pending | rendering | rendered | failed | cancelled.
    The record binds job_id + clip_index + project version + compile digest,
    so a delayed response cannot replace another selection's preview.
- POST /api/hyperframes/render/{render_id}/cancel
    Cooperative cancellation.

The legacy renderer is untouched: when the project does not opt into the
hyperframes renderer (default OFF), these endpoints refuse with an honest
error and the accepted compose path stays the way to render.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from clippyme.api.auth import (
    AuthUser,
    get_current_user,
    verify_job_ownership_on_disk,
)
from clippyme.api.security import enforce_rate_limit
from clippyme.domain.history_service import is_valid_job_id

logger = logging.getLogger("clippyme.hyperframes")

router = APIRouter()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")

# Compile/render records live job-relative under .hf/ (never served by the
# static /videos mount: SafeStaticFiles blocks dot-prefixed leaves).
_HF_DIRNAME = ".hf"
_RENDER_RECORDS = "renders"

# Request bounds (contract 6).
_MAX_BODY_PARAMS_BYTES = 64 * 1024
_DEADLINE_MIN, _DEADLINE_MAX = 30, 3600

# In-process background render tasks (pilot: single worker anyway).
_RENDER_TASKS: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Request models (strict validation at the boundary)
# ---------------------------------------------------------------------------

class CompileRequest(BaseModel):
    # Optional preview overrides: validated when present, applied to this
    # compile only (the response marks preview_override=true). Persisting a
    # template choice always goes through the project save (Apply).
    template_id: str | None = Field(default=None, min_length=1, max_length=64)
    params: dict = Field(default_factory=dict)
    expected_version: int = Field(ge=1)


class RenderRequest(BaseModel):
    # Export always renders the project's saved finishing selection (the
    # accepted snapshot) — no per-render template overrides, so a render can
    # never silently diverge from what was applied.
    compile_id: str | None = Field(default=None, min_length=1, max_length=64)
    expected_version: int | None = Field(default=None, ge=1)
    deadline_sec: int = Field(default=1200, ge=_DEADLINE_MIN, le=_DEADLINE_MAX)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _job_dir(job_id: str) -> str:
    return os.path.join(OUTPUT_DIR, job_id)


def _hf_dir(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), _HF_DIRNAME)


def _guard_job(job_id: str, user: AuthUser) -> str:
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    verify_job_ownership_on_disk(job_id, user, OUTPUT_DIR)
    job_dir = _job_dir(job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found on disk")
    return job_dir


def _compile_id_ok(compile_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{16}", compile_id or ""))


def _render_id_ok(render_id: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f-]{36}_[0-9a-f]{16}", render_id or ""))


def _resolve_hf_asset(job_dir: str, compile_id: str, asset: str) -> str:
    """Resolve an allowlisted composition asset; reject traversal/symlinks."""
    if not _compile_id_ok(compile_id):
        raise HTTPException(status_code=400, detail="Invalid compile ID")
    allowed = {"index.html", "base.mp4"}
    # Template fonts: closed allowlist from the finishing templates.
    from clippyme.finishing import templates as _t
    for spec in _t.TEMPLATES.values():
        allowed.add(spec.font_file)
        if spec.hook_font_file:
            allowed.add(spec.hook_font_file)
    if asset not in allowed:
        raise HTTPException(status_code=404, detail="Unknown asset")
    comp_dir = os.path.join(job_dir, _HF_DIRNAME, compile_id)
    candidate = os.path.normpath(os.path.join(comp_dir, asset))
    if not candidate.startswith(os.path.abspath(comp_dir) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid asset path")
    if os.path.islink(candidate):
        raise HTTPException(status_code=400, detail="Refusing symlinked asset")
    if not os.path.isfile(candidate):
        raise HTTPException(status_code=404, detail="Asset not found")
    return candidate


def _load_project_at_version(job_dir: str, clip_index: int,
                             expected_version: int | None):
    """Load the frozen accepted project snapshot (version-guarded).

    Mirrors the compose route/version contract: the exact saved version the
    caller acknowledged. expected_version=None loads the latest.
    """
    from clippyme.domain.project_render import (
        get_project_path, latest_project_version)
    from clippyme.domain.clip_project import validate_project

    latest = latest_project_version(job_dir, clip_index)
    if latest < 1:
        raise HTTPException(
            status_code=404, detail="No project found for this clip")
    want = expected_version if expected_version else latest
    if want != latest:
        raise HTTPException(
            status_code=409,
            detail=f"Stale project version: acknowledged v{want}, "
                   f"latest is v{latest}. Reload the clip and retry.")
    path = get_project_path(job_dir, clip_index, want)
    if not path:
        raise HTTPException(status_code=404, detail="Project version not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    from clippyme.domain.clip_project import _backfill_word_ids
    data = _backfill_word_ids(data, repair_duplicates=True)
    try:
        project = validate_project(data)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Saved project failed validation: {exc}")
    return project, want


def _capability() -> dict:
    from clippyme.finishing import templates as _t
    from clippyme.finishing import worker_client as _wc
    from clippyme.finishing import compile as _c
    worker = _wc.check_available()
    return {
        "available": bool(worker.get("available")),
        "reason": worker.get("reason", ""),
        "renderer_default": "legacy",
        "templates": [
            {"id": s.id, "version": s.version, "title": s.title,
             "description": s.description,
             "params": {k: list(v) for k, v in s.param_enums.items()}}
            for s in _t.TEMPLATES.values()
        ],
        "renderer": _c.renderer_versions(),
        "worker": {k: v for k, v in worker.items()
                   if k in ("node", "chromium", "ffmpeg", "engine")},
    }


def _record_path(job_dir: str, render_id: str) -> str:
    return os.path.join(job_dir, _HF_DIRNAME, _RENDER_RECORDS, render_id + ".json")


def _write_record(job_dir: str, render_id: str, record: dict) -> None:
    rec_dir = os.path.join(job_dir, _HF_DIRNAME, _RENDER_RECORDS)
    os.makedirs(rec_dir, exist_ok=True)
    tmp = os.path.join(rec_dir, render_id + ".json.part")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, _record_path(job_dir, render_id))


def _read_record(job_dir: str, render_id: str) -> dict | None:
    path = _record_path(job_dir, render_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

@router.get("/api/hyperframes/capability")
async def hyperframes_capability(request: Request,
                                 user: AuthUser = Depends(get_current_user)):
    """Honest capability state (default OFF; unavailable stays honest)."""
    return _capability()


# ---------------------------------------------------------------------------
# Compile (preview)
# ---------------------------------------------------------------------------

@router.post("/api/hyperframes/compile/{job_id}/{clip_index}")
async def hyperframes_compile(job_id: str, clip_index: int,
                              body: CompileRequest, request: Request,
                              user: AuthUser = Depends(get_current_user)):
    """Compile the acknowledged project snapshot into a preview composition."""
    enforce_rate_limit(request, "hyperframes_compile",
                       capacity=20, refill_per_sec=20 / 60)
    job_dir = _guard_job(job_id, user)
    if clip_index < 0 or clip_index > 9999:
        raise HTTPException(status_code=400, detail="Invalid clip index")
    if len(json.dumps(body.params)) > _MAX_BODY_PARAMS_BYTES:
        raise HTTPException(status_code=413, detail="Params too large")

    project, version = _load_project_at_version(
        job_dir, clip_index, body.expected_version)
    finishing = getattr(project, "finishing", None)
    if not finishing or finishing.renderer != "hyperframes":
        raise HTTPException(
            status_code=400,
            detail="Hyperframes renderer is not enabled for this project "
                   "(default OFF). Apply a Hyperframes template first; the "
                   "accepted legacy renderer is unchanged.")

    # Effective template/params: the project's saved finishing selection,
    # optionally overridden for THIS preview only (validated; overrides are
    # marked in the response and never persisted).
    eff_template = body.template_id or finishing.template
    if not eff_template:
        raise HTTPException(
            status_code=400,
            detail="No template selected: apply a Hyperframes template first "
                   "or pass template_id for a preview override.")
    preview_override = bool(body.template_id or body.params)
    eff_params = {"caption_size": finishing.caption_size,
                  "caption_position": finishing.caption_position,
                  **body.params}

    from clippyme.finishing import compile as _c
    try:
        compiled = await _c.compile_hyperframes_composition(
            project,
            template_id=eff_template,
            params=eff_params,
            job_dir=job_dir,
            clip_index=clip_index,
            acknowledged_version=version,
            renderer_enabled=True,
        )
    except _c.FinishingDisabled as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except _c.CompileError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001 - never leak internals
        logger.exception("hyperframes compile failed for %s clip %d",
                         job_id, clip_index)
        raise HTTPException(status_code=500, detail="Compile failed")

    # Persist the compile record (the render endpoint consumes the SAME dir).
    rec = {
        "compile_id": compiled.compile_id,
        "job_id": job_id,
        "clip_index": clip_index,
        "project_version": version,
        "project_digest": compiled.manifest["project_digest"],
        "preview_identity": compiled.digest,
        "template": compiled.manifest["template"],
        "created_at": time.time(),
    }
    crec_dir = os.path.join(_hf_dir(job_id), "compiles")
    os.makedirs(crec_dir, exist_ok=True)
    with open(os.path.join(crec_dir, compiled.compile_id + ".json"),
              "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    return {
        "state": "compiled",
        "preview_override": preview_override,
        "compile_id": compiled.compile_id,
        "composition_url": (
            f"/api/hyperframes/composition/{job_id}/"
            f"{compiled.compile_id}/index.html"),
        "preview_identity": compiled.digest,
        "manifest": compiled.manifest,
    }


@router.get("/api/hyperframes/composition/{job_id}/{compile_id}/{asset}")
async def hyperframes_composition_asset(
        job_id: str, compile_id: str, asset: str, request: Request,
        user: AuthUser = Depends(get_current_user)):
    """Serve the compiled composition + allowlisted local assets (sandboxed)."""
    job_dir = _guard_job(job_id, user)
    path = _resolve_hf_asset(job_dir, compile_id, asset)
    media_type, _ = mimetypes.guess_type(path)
    if asset.endswith(".html"):
        media_type = "text/html"
    elif asset.endswith(".ttf"):
        media_type = "font/ttf"
    # No remote fetches from generated compositions: the composition dir only
    # ever contains compiler-staged local files (validated at resolve time).
    return FileResponse(path, media_type=media_type or "application/octet-stream")


# ---------------------------------------------------------------------------
# Render (export)
# ---------------------------------------------------------------------------

def _output_name(clip_index: int, version: int, template_id: str) -> str:
    safe = re.sub(r"[^a-z0-9-]", "", template_id or "template")
    return f"hf_clip_{clip_index}_v{version}_{safe}.mp4"


@router.post("/api/hyperframes/render/{job_id}/{clip_index}")
async def hyperframes_render(job_id: str, clip_index: int,
                             body: RenderRequest, request: Request,
                             user: AuthUser = Depends(get_current_user)):
    """Dispatch ONE bounded local worker render of the accepted snapshot.

    Consumes the frozen accepted project snapshot (same version contract as
    the compose route). When the project does not opt into the hyperframes
    renderer, refuses honestly — never silently publishes legacy output.
    """
    enforce_rate_limit(request, "hyperframes_render",
                       capacity=4, refill_per_sec=4 / 600)
    job_dir = _guard_job(job_id, user)
    if clip_index < 0 or clip_index > 9999:
        raise HTTPException(status_code=400, detail="Invalid clip index")

    from clippyme.finishing import compile as _c
    from clippyme.finishing import cache as _cache
    from clippyme.finishing import worker_client as _wc

    # Resolve the composition: an explicit compile_id reuses the exact
    # compiled dir the preview showed (SAME composition + clock); otherwise
    # compile the acknowledged snapshot now.
    compile_id = body.compile_id
    compiled = None
    if compile_id:
        if not _compile_id_ok(compile_id):
            raise HTTPException(status_code=400, detail="Invalid compile ID")
        crec = os.path.join(_hf_dir(job_id), "compiles", compile_id + ".json")
        if not os.path.isfile(crec):
            raise HTTPException(status_code=404, detail="Compile not found")
        try:
            rec = json.load(open(crec, encoding="utf-8"))
        except (OSError, ValueError):
            raise HTTPException(status_code=400, detail="Corrupt compile record")
        if rec.get("job_id") != job_id or int(rec.get("clip_index", -1)) != clip_index:
            raise HTTPException(
                status_code=400,
                detail="Compile belongs to a different clip/job")
        if (body.expected_version is not None
                and int(rec.get("project_version", -1)) != body.expected_version):
            raise HTTPException(
                status_code=409,
                detail="Compile was built from a different project version")
        comp_dir = os.path.join(_hf_dir(job_id), compile_id)
        manifest_path = os.path.join(comp_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise HTTPException(status_code=410, detail="Compile expired")
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        preview_identity = rec.get("preview_identity")
        project_version = int(rec["project_version"])
        template_id = manifest["template"]["id"]
    else:
        # Export always renders the project's saved finishing selection:
        # the accepted snapshot, no per-render template overrides.
        project, project_version = _load_project_at_version(
            job_dir, clip_index, body.expected_version)
        finishing = getattr(project, "finishing", None)
        if not finishing or finishing.renderer != "hyperframes":
            raise HTTPException(
                status_code=400,
                detail="Hyperframes renderer is not enabled for this project "
                       "(default OFF). Apply a Hyperframes template first; the "
                       "accepted legacy renderer is unchanged.")
        if not finishing.template:
            raise HTTPException(
                status_code=400,
                detail="No template selected on this project.")
        try:
            compiled = await _c.compile_hyperframes_composition(
                project,
                template_id=finishing.template,
                params={"caption_size": finishing.caption_size,
                        "caption_position": finishing.caption_position},
                job_dir=job_dir,
                clip_index=clip_index,
                acknowledged_version=project_version,
                renderer_enabled=True,
            )
        except _c.FinishingDisabled as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except _c.CompileError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception:  # noqa: BLE001
            logger.exception("hyperframes render compile failed for %s clip %d",
                             job_id, clip_index)
            raise HTTPException(status_code=500, detail="Compile failed")
        comp_dir = compiled.composition_dir
        compile_id = compiled.compile_id
        manifest_path = os.path.join(comp_dir, "manifest.json")
        manifest = compiled.manifest
        preview_identity = compiled.digest
        template_id = manifest["template"]["id"]

    # Cache: key binds base + project digest/version + template/font/renderer.
    key = _cache.cache_key(
        base_sha256=manifest["base"]["sha256"],
        project_digest=manifest["project_digest"],
        project_version=project_version,
        template_id=manifest["template"]["id"],
        template_version=manifest["template"]["version"],
        params=manifest["params"],
        font_shas=[f["sha256"] for f in manifest["fonts"]],
        renderer=manifest["renderer"],
        width=manifest["render"]["width"],
        height=manifest["render"]["height"],
        fps=manifest["render"]["fps"],
    )
    cached = _cache.lookup(job_dir, key)
    output_name = _output_name(clip_index, project_version, template_id)
    output_path = os.path.join(job_dir, output_name)
    render_id = f"{job_id}_{uuid.uuid4().hex[:16]}"
    now = time.time()

    if cached:
        shutil.copy2(cached, output_path)
        record = {
            "render_id": render_id, "job_id": job_id, "clip_index": clip_index,
            "project_version": project_version, "compile_id": compile_id,
            "preview_identity": preview_identity, "template_id": template_id,
            "state": "rendered", "cache_hit": True,
            "output_file": output_name,
            "output_url": f"/videos/{job_id}/{output_name}",
            "created_at": now, "finished_at": now,
            "metrics": {"cache": "hit", "cache_key": key},
        }
        _write_record(job_dir, render_id, record)
        return {"render_id": render_id, "state": "rendered",
                "cache_hit": True, "output_url": record["output_url"],
                "preview_identity": preview_identity}

    # Dispatch the bounded worker in the background; state via GET.
    workdir = os.path.join(_hf_dir(job_id), "work", render_id)
    os.makedirs(workdir, exist_ok=True)
    worker_manifest = {
        "composition_dir": comp_dir,
        "width": manifest["render"]["width"],
        "height": manifest["render"]["height"],
        "fps": manifest["render"]["fps"],
        "duration": manifest["duration"],
        "frames": manifest["frames"],
        "audio_source": os.path.join(comp_dir, "base.mp4"),
        "format": "png",
        "deadline_sec": body.deadline_sec,
    }
    worker_manifest_path = os.path.join(workdir, "worker-manifest.json")
    with open(worker_manifest_path, "w", encoding="utf-8") as f:
        json.dump(worker_manifest, f)
    worker_out = os.path.join(workdir, "rendered.mp4")

    record = {
        "render_id": render_id, "job_id": job_id, "clip_index": clip_index,
        "project_version": project_version, "compile_id": compile_id,
        "preview_identity": preview_identity, "template_id": template_id,
        "state": "rendering", "cache_hit": False, "cache_key": key,
        "output_file": output_name,
        "output_url": f"/videos/{job_id}/{output_name}",
        "created_at": now, "finished_at": None,
        "workdir": workdir, "metrics": {},
    }
    _write_record(job_dir, render_id, record)

    async def _run():
        from clippyme.finishing import worker_client as _w
        rec = _read_record(job_dir, render_id) or {}
        try:
            result = await asyncio.to_thread(
                _w.run_worker_render,
                manifest_path=worker_manifest_path,
                out_path=worker_out,
                workdir=workdir,
                deadline_sec=body.deadline_sec,
            )
        except _w.WorkerBusy as exc:
            rec.update(state="failed", finished_at=time.time(),
                       error=f"worker busy: {exc}")
            _write_record(job_dir, render_id, rec)
            return
        except _w.WorkerCancelled as exc:
            rec.update(state="cancelled", finished_at=time.time(),
                       error=str(exc))
            _write_record(job_dir, render_id, rec)
            return
        except _w.WorkerDeadline as exc:
            rec.update(state="failed", finished_at=time.time(),
                       error=f"deadline: {exc}")
            _write_record(job_dir, render_id, rec)
            return
        except Exception as exc:  # noqa: BLE001
            rec.update(state="failed", finished_at=time.time(),
                       error=f"worker error: {exc}")
            _write_record(job_dir, render_id, rec)
            return
        # Atomic publish: cache first, then the versioned output name.
        # A concurrent cancel wins over a late publish: never install output
        # for a render that is no longer "rendering".
        try:
            rec_now = _read_record(job_dir, render_id) or {}
            if rec_now.get("state") != "rendering":
                return
            _cache.store(job_dir, key, worker_out)
            shutil.copy2(worker_out, output_path)
        except OSError as exc:
            rec.update(state="failed", finished_at=time.time(),
                       error=f"publish failed: {exc}")
            _write_record(job_dir, render_id, rec)
            return
        rec.update(state="rendered", finished_at=time.time(),
                   metrics=result.metrics)
        _write_record(job_dir, render_id, rec)

    task = asyncio.create_task(_run())
    _RENDER_TASKS[render_id] = task
    task.add_done_callback(lambda t: _RENDER_TASKS.pop(render_id, None))

    return {"render_id": render_id, "state": "rendering",
            "cache_hit": False, "preview_identity": preview_identity,
            "output_url": record["output_url"]}


@router.get("/api/hyperframes/render/{render_id}")
async def hyperframes_render_status(render_id: str, request: Request,
                                    user: AuthUser = Depends(get_current_user)):
    """Render job state. Binds job/clip/version/digest (contract 8)."""
    if not _render_id_ok(render_id):
        raise HTTPException(status_code=400, detail="Invalid render ID")
    job_id = render_id.rsplit("_", 1)[0]
    job_dir = _guard_job(job_id, user)
    rec = _read_record(job_dir, render_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Render not found")
    # Reconcile: a "rendering" record with no live task and no result is dead
    # (e.g. process restart) — report honestly instead of hanging.
    if rec.get("state") == "rendering" and render_id not in _RENDER_TASKS:
        workdir = rec.get("workdir") or ""
        result_path = os.path.join(workdir, "result.json")
        if os.path.isfile(result_path):
            try:
                res = json.load(open(result_path, encoding="utf-8"))
            except (OSError, ValueError):
                res = {}
            status = res.get("status", "failed")
            rec.update(state=status if status in (
                "rendered", "failed", "cancelled", "deadline") else "failed",
                finished_at=time.time(),
                error=res.get("error") or "worker did not report back",
                metrics=res)
            _write_record(job_dir, render_id, rec)
        else:
            rec.update(state="failed", finished_at=time.time(),
                       error="render worker did not survive (no result)")
            _write_record(job_dir, render_id, rec)
    return {
        "render_id": render_id,
        "state": rec.get("state"),
        "job_id": rec.get("job_id"),
        "clip_index": rec.get("clip_index"),
        "project_version": rec.get("project_version"),
        "preview_identity": rec.get("preview_identity"),
        "template_id": rec.get("template_id"),
        "cache_hit": rec.get("cache_hit", False),
        "output_url": rec.get("output_url"),
        "error": rec.get("error"),
        "metrics": rec.get("metrics", {}),
    }


@router.post("/api/hyperframes/render/{render_id}/cancel")
async def hyperframes_render_cancel(render_id: str, request: Request,
                                    user: AuthUser = Depends(get_current_user)):
    """Cooperative cancellation of a running render."""
    if not _render_id_ok(render_id):
        raise HTTPException(status_code=400, detail="Invalid render ID")
    job_id = render_id.rsplit("_", 1)[0]
    job_dir = _guard_job(job_id, user)
    rec = _read_record(job_dir, render_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Render not found")
    if rec.get("state") not in ("rendering",):
        return {"render_id": render_id, "state": rec.get("state"),
                "note": "not running; nothing to cancel"}
    from clippyme.finishing import worker_client as _w
    workdir = rec.get("workdir") or ""
    if workdir:
        _w.request_cancel(workdir)
    task = _RENDER_TASKS.get(render_id)
    if task is not None:
        task.cancel()
    rec.update(state="cancelled", finished_at=time.time(),
               error="cancel requested")
    _write_record(job_dir, render_id, rec)
    return {"render_id": render_id, "state": "cancelled"}
