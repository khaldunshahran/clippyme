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
- POST /api/hyperframes/preview-token/{job_id}/{compile_id}
    Mint a short-lived HMAC preview token (single-compile scope, ~15 min)
    so a browser <iframe> can load the composition without the apiFetch
    Bearer <redacted> Authed via get_current_user, job ownership checked.
- GET  /api/hyperframes/composition/{job_id}/{compile_id}/{asset}
    Serve the compiled composition and its allowlisted local assets
    (index.html, base.mp4, template fonts). Traversal/symlink-safe.
    Accepts ?token= (a valid preview token) instead of get_current_user;
    the Bearer <redacted> is unchanged when no token is given.
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
import inspect
import json
import logging
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid

from urllib.parse import quote as _url_quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


def _composition_asset_allowlist() -> set[str]:
    """Closed allowlist of composition-dependent assets (Task 18 review-02).

    index.html plus the compiler-staged video and template font files.
    Shared by the asset resolver and the token injector so the two can
    never disagree about which relative resources are authorized.
    """
    allowed = {"index.html", "base.mp4"}
    from clippyme.finishing import templates as _t
    for spec in _t.TEMPLATES.values():
        allowed.add(spec.font_file)
        if spec.hook_font_file:
            allowed.add(spec.hook_font_file)
    return allowed


def _inject_preview_token(html: str, token: str) -> str:
    """Rewrite allowlisted relative asset references to carry ?token=.

    Relative resources do NOT inherit the navigation query, so without
    this the token-granted index.html loads a base.mp4/font that 401s.
    Only the closed allowlist above is rewritten (exact src=/url() shapes
    the compiler emits); the scoped preview token is the same grant, not
    a weakened one, and no JWT ever enters a URL.
    """
    q = "?token=" + _url_quote(token, safe="")
    for name in sorted(_composition_asset_allowlist()):
        if name == "index.html":
            continue
        tagged = name + q
        html = html.replace(f'src="{name}"', f'src="{tagged}"')
        html = html.replace(f"src='{name}'", f"src='{tagged}'")
        html = html.replace(f'url("{name}")', f'url("{tagged}")')
        html = html.replace(f"url('{name}')", f"url('{tagged}')")
        html = html.replace(f"url({name})", f"url({tagged})")
    return html


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
# Atomic publication (review-01 group 4)
# ---------------------------------------------------------------------------

class PublishValidationError(Exception):
    """A staged output file failed media validation."""


class PublishSkipped(Exception):
    """The render left the 'rendering' state (e.g. cancelled) before publish:
    output is deliberately NOT installed. The record already carries an
    honest note; this is not an error."""


def _validate_staged_output(path: str, *, width: int, height: int) -> None:
    """Probe a staged output file before it may replace the final export.

    Raises PublishValidationError unless the file has a video stream, a
    positive duration, and the manifest's render dimensions.
    """
    from clippyme.domain import project_render as pr
    probe = pr.probe_media(path)
    if probe.get("has_video") is not True:
        raise PublishValidationError(
            f"staged output has no video stream: {os.path.basename(path)}")
    dur = probe.get("duration")
    if not (isinstance(dur, (int, float)) and dur > 0):
        raise PublishValidationError(
            f"staged output has non-positive duration: {dur!r}")
    if probe.get("width") != int(width) or probe.get("height") != int(height):
        raise PublishValidationError(
            f"staged output dims {probe.get('width')}x{probe.get('height')} "
            f"!= render {width}x{height}")


def _stage_and_install(src_path: str, final_path: str, *,
                       width: int, height: int) -> None:
    """Atomically install src_path as final_path.

    Stages into a destination-local unique temp file (same directory, so the
    final os.replace is atomic), validates the staged file, then replaces.
    On ANY failure the previous final_path is left untouched and only our
    own temp file is removed.
    """
    dest_dir = os.path.dirname(os.path.abspath(final_path))
    tmp = os.path.join(
        dest_dir,
        os.path.basename(final_path) + f".part-{uuid.uuid4().hex[:8]}")
    try:
        shutil.copy2(src_path, tmp)
        _validate_staged_output(tmp, width=width, height=height)
        os.replace(tmp, final_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


async def _publish_worker_output(*, job_dir: str, render_id: str,
                                 src_path: str, cache_key: str,
                                 output_path: str, width: int, height: int) -> None:
    """Publish a finished worker render: install into the cache, then
    atomically install the versioned output file.

    Commit is coordinated with the terminal state: the prior export is
    snapshotted BEFORE the staging/install thread, and the record is
    re-read AFTER the thread settles. If the render left 'rendering' in the
    meantime (cancel during copy/validation), the prior export is restored
    and PublishSkipped is raised. When the commit gate passes, the terminal
    'rendered' state is recorded BEFORE the cancellable backup cleanup, so
    a cancel landing in cleanup sees a terminal record and reports honestly
    instead of overwriting a committed export with 'cancelled'. When
    cancellation wins instead (during staging), the cancelled record stays
    truthful about the bytes on disk. A staging/install
    failure propagates normally (atomic _stage_and_install leaves the prior
    output untouched). The early state check below is only a fast path; the
    commit gate after settle is the actual decision point.
    """
    from clippyme.finishing import cache as _cache
    rec = _read_record(job_dir, render_id) or {}
    if rec.get("state") != "rendering":
        note = (f"worker finished but the record left 'rendering' "
                f"(state={rec.get('state')!r}); output not published")
        rec["publish_note"] = note
        _write_record(job_dir, render_id, rec)
        raise PublishSkipped(note)
    _cache.store(job_dir, cache_key, src_path)
    backup_path = None
    if os.path.isfile(output_path):
        backup_path = f"{output_path}.prev-{uuid.uuid4().hex[:8]}"
        await asyncio.to_thread(shutil.copy2, output_path, backup_path)
    settled = threading.Event()
    box: dict = {}

    def _run_stage() -> None:
        try:
            _stage_and_install(src_path, output_path,
                               width=width, height=height)
        except BaseException as exc:  # staging failure: propagate below
            box["error"] = exc
        finally:
            settled.set()

    cancelled = False
    try:
        await asyncio.to_thread(_run_stage)
    except asyncio.CancelledError:
        # The staging thread is NOT stopped by the task cancel; wait for it
        # (shielded) before the commit gate so restore can never be followed
        # by a late os.replace.
        cancelled = True
    if not settled.is_set():
        await asyncio.shield(asyncio.to_thread(settled.wait))
    if "error" in box:
        if backup_path:
            try:
                await asyncio.to_thread(os.remove, backup_path)
            except OSError:
                pass
        err = box["error"]
        if cancelled:
            raise asyncio.CancelledError() from err
        raise err
    # Commit gate: decide AFTER staging settled, on the CURRENT record.
    rec2 = _read_record(job_dir, render_id) or {}
    if cancelled or rec2.get("state") != "rendering":
        if backup_path and os.path.isfile(backup_path):
            await asyncio.to_thread(os.replace, backup_path, output_path)
        elif backup_path is None and os.path.isfile(output_path):
            # No prior export existed: never leave a fresh install behind a
            # non-rendering record.
            try:
                await asyncio.to_thread(os.remove, output_path)
            except OSError:
                pass
        note = (f"publish superseded: record left 'rendering' "
                f"(state={rec2.get('state')!r}) during staging; prior "
                f"output preserved")
        rec2["publish_note"] = note
        _write_record(job_dir, render_id, rec2)
        if cancelled:
            raise asyncio.CancelledError()
        raise PublishSkipped(note)
    # Commit stands: record the terminal state BEFORE any cancellable
    # cleanup. The staging thread already installed the new output
    # atomically; once 'rendered' is recorded, a cancel landing in cleanup
    # sees a terminal record and reports honestly instead of overwriting a
    # committed export with 'cancelled'. There is no await between the
    # commit gate above and this write, so the cancel route cannot
    # interleave between the decision and its recording.
    rec2["state"] = "rendered"
    rec2["finished_at"] = time.time()
    _write_record(job_dir, render_id, rec2)
    # Backup cleanup is best-effort and must not change the terminal
    # decision: shield it so a late cancel cannot turn a committed render
    # into a cancelled task. The backup is ours alone (unique .prev- name).
    if backup_path:
        try:
            await asyncio.shield(asyncio.to_thread(os.remove, backup_path))
        except OSError:
            pass
        except asyncio.CancelledError:
            # Terminal state already committed above; the orphaned backup is
            # harmless. Re-raise so a genuinely cancelled task still reports
            # cancellation instead of swallowing it.
            raise


async def _route_current_user(request: Request) -> AuthUser:
    """Resolve the current user for the composition asset route.

    Honors the FastAPI dependency override used by the test harness; in
    production this is exactly get_current_user(request). Separated out
    because the route must skip session auth entirely when a valid preview
    token is presented (a Depends would raise 401 before the route body
    runs); the Bearer <redacted> is otherwise unchanged.
    """
    override = request.app.dependency_overrides.get(get_current_user)
    if override is not None:
        result = override()
        if inspect.isawaitable(result):
            result = await result
        return result
    return get_current_user(request)


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
    except _c.CleanBaseError as exc:
        # Honest refusal (framing trajectory, reframed_base lineage,
        # configured music): the pilot cannot render this project.
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


@router.post("/api/hyperframes/preview-token/{job_id}/{compile_id}")
async def hyperframes_preview_token(job_id: str, compile_id: str,
                                    request: Request,
                                    user: AuthUser = Depends(get_current_user)):
    """Mint a short-lived, single-compile preview token for the composition.

    Lets a browser <iframe> load the authed composition assets without
    attaching the apiFetch Bearer <redacted> The token is HMAC-signed with
    the app's existing shared secret (never a Supabase JWT), scoped to this
    job+compile, ~15 min expiry. Authed via get_current_user; job ownership
    checked like every other endpoint.
    """
    enforce_rate_limit(request, "hyperframes_preview_token",
                       capacity=20, refill_per_sec=20 / 60)
    job_dir = _guard_job(job_id, user)
    if not _compile_id_ok(compile_id):
        raise HTTPException(status_code=400, detail="Invalid compile ID")
    crec = os.path.join(_hf_dir(job_id), "compiles", compile_id + ".json")
    if not os.path.isfile(crec):
        raise HTTPException(status_code=404, detail="Unknown compile ID")
    from clippyme.finishing import preview_tokens as _pt
    token, expires_in = _pt.mint_preview_token(job_id, compile_id)
    return {"token": token, "expires_in": expires_in}


@router.get("/api/hyperframes/composition/{job_id}/{compile_id}/{asset}")
async def hyperframes_composition_asset(
        job_id: str, compile_id: str, asset: str, request: Request,
        token: str | None = Query(default=None, max_length=2048)):
    """Serve the compiled composition + allowlisted local assets (sandboxed).

    Two auth modes (ownership never weakened):
    - ?token=: a valid HMAC preview token (minted authed above) scoped to
      this job+compile — served WITHOUT get_current_user so browser iframe
      navigation works. Bad/expired/mis-scoped tokens -> 401; unknown
      compile -> 404. When index.html is served under a valid token, its
      allowlisted relative asset references (base.mp4, template fonts) are
      rewritten to carry the SAME token, because relative resources do not
      inherit the navigation query and would otherwise 401.
    - no token: the existing Bearer <redacted> (unchanged).

    Fonts are served with Access-Control-Allow-Origin: * (token- or Bearer
    <redacted>): the composition iframe is sandbox=allow-scripts, i.e. an
    opaque origin, and fonts are CORS-checked resources that would stay
    blocked without it. The grant remains the token/Bearer session, so the
    wildcard does not weaken ownership.
    """
    from clippyme.finishing import preview_tokens as _pt
    if token:
        try:
            _pt.verify_preview_token(token, job_id=job_id,
                                     compile_id=compile_id)
        except _pt.PreviewTokenError as exc:
            raise HTTPException(status_code=401, detail=str(exc))
        if not is_valid_job_id(job_id):
            raise HTTPException(status_code=400, detail="Invalid job ID")
        job_dir = _job_dir(job_id)
        if not os.path.isdir(job_dir):
            raise HTTPException(status_code=404, detail="Job not found on disk")
        crec = os.path.join(job_dir, _HF_DIRNAME, "compiles",
                            compile_id + ".json")
        if not os.path.isfile(crec):
            raise HTTPException(status_code=404, detail="Unknown compile ID")
    else:
        user = await _route_current_user(request)
        job_dir = _guard_job(job_id, user)
    path = _resolve_hf_asset(job_dir, compile_id, asset)
    media_type, _ = mimetypes.guess_type(path)
    if asset.endswith(".html"):
        media_type = "text/html"
    elif asset.endswith(".ttf"):
        media_type = "font/ttf"
    headers = {}
    if asset.endswith((".ttf", ".woff", ".woff2")) or token:
        headers["Access-Control-Allow-Origin"] = "*"
    # Token-granted index.html: authorize its allowlisted dependent assets
    # with the same scoped grant (relative URLs carry no query).
    if asset == "index.html" and token:
        from fastapi.responses import Response
        html = _inject_preview_token(
            open(path, "r", encoding="utf-8").read(), token)
        return Response(content=html, media_type="text/html",
                        headers=headers)
    # No remote fetches from generated compositions: the composition dir only
    # ever contains compiler-staged local files (validated at resolve time).
    return FileResponse(path, media_type=media_type or "application/octet-stream",
                        headers=headers)


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
        # Snapshot binding (review-01 group 3): passing compile_id must not
        # bypass snapshot validation. The compile must be for the CURRENT
        # saved project — same version AND same digest, recomputed with the
        # compiler's digest function. A newer save or any project change
        # after the preview makes the preview stale: re-preview to export.
        project, latest = _load_project_at_version(job_dir, clip_index, None)
        rec_version = int(rec.get("project_version", -1))
        if rec_version != latest:
            raise HTTPException(
                status_code=409,
                detail=f"Preview is stale: compiled from project v{rec_version}, "
                       f"but the project is now v{latest}. Re-preview to export.")
        if rec.get("project_digest") != _c.project_digest(project):
            raise HTTPException(
                status_code=409,
                detail="Preview is stale: the project changed after this "
                       "preview was compiled. Re-preview to export.")
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
        except _c.CleanBaseError as exc:
            # Honest refusal (framing trajectory, reframed_base lineage,
            # configured music): the pilot cannot render this project.
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
        # Atomic publish: stage into a destination-local temp file, validate
        # the staged media, then replace. A failed publish never leaves a
        # partial file behind and the previous complete export is preserved.
        try:
            _stage_and_install(cached, output_path,
                               width=manifest["render"]["width"],
                               height=manifest["render"]["height"])
        except Exception as exc:  # noqa: BLE001 - honest failure, keep old output
            record = {
                "render_id": render_id, "job_id": job_id,
                "clip_index": clip_index,
                "project_version": project_version, "compile_id": compile_id,
                "preview_identity": preview_identity,
                "template_id": template_id,
                "state": "failed", "cache_hit": True,
                "output_file": output_name,
                "output_url": f"/videos/{job_id}/{output_name}",
                "created_at": now, "finished_at": time.time(),
                "error": f"publish failed: {exc}; previous output preserved",
                "metrics": {"cache": "hit", "cache_key": key},
            }
            _write_record(job_dir, render_id, record)
            raise HTTPException(status_code=500, detail=record["error"])
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
        # Atomic publish: install into the cache, then stage + validate +
        # atomically replace the versioned output name. A concurrent cancel
        # wins over a late publish: never install output for a render that
        # is no longer "rendering" (the record keeps its terminal state with
        # an honest note).
        try:
            await _publish_worker_output(
                job_dir=job_dir, render_id=render_id, src_path=worker_out,
                cache_key=key, output_path=output_path,
                width=manifest["render"]["width"],
                height=manifest["render"]["height"])
        except PublishSkipped:
            # Terminal-state race: the record already carries the honest
            # note and keeps its state (e.g. cancelled). Nothing to do.
            return
        except Exception as exc:  # noqa: BLE001
            rec = _read_record(job_dir, render_id) or {}
            rec.update(state="failed", finished_at=time.time(),
                       error=f"publish failed: {exc}")
            _write_record(job_dir, render_id, rec)
            return
        rec = _read_record(job_dir, render_id) or {}
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
        "note": rec.get("publish_note"),
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
