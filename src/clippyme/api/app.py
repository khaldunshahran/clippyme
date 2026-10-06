import json
import os
import sys
import uuid
import shutil
import glob
import asyncio
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)
logger = logging.getLogger("clippyme")

# The pinned dependency set (faster-whisper, mediapipe, etc.) is only tested on
# Python 3.11+. Warn loudly rather than failing with a cryptic import error on
# an older interpreter.
if sys.version_info < (3, 11):
    logger.warning(
        "ClippyMe requires Python 3.11+. Detected %s — imports may fail or behave unexpectedly.",
        ".".join(map(str, sys.version_info[:3])),
    )

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from clippyme.api.auth import AuthUser, get_current_user

from clippyme.domain.job_results import build_main_cmd, canonical_reframe_mode
from clippyme.domain.runtime_state import (
    STAGE_ORDER,
    STAGE_PROGRESS,
    load_runtime_state,
    worker_python,
)
from clippyme.domain.compose import compose_layers
from clippyme.domain.reframe_service import run_reframe
from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.uploads import stream_upload_within_limit, FileTooLarge
from clippyme.domain.clip_endpoints import run_smart_cut, restore_job_from_disk
from clippyme.domain.clip_resolve import resolve_clip
from clippyme.domain import clip_tools
from clippyme.domain import job_control
from clippyme.domain.job_actions import cancel_job_action, stop_job_action
from clippyme.domain.job_journal import JOURNAL_FILENAME, make_journal_writer, recover_jobs
from clippyme.domain.job_runner import make_run_job
from clippyme.domain.job_submission import QueueFullError, submit_job
from clippyme.domain.publish_service import publish_clip_flow
from clippyme.api.schemas import (
    BatchRequest,
    ComposeRequest,
    EditAIRequest,
    GenerateMetadataRequest,
    LiveMonitorPublishingRequest,
    LiveMonitorStartRequest,
    LiveMonitorStopRequest,
    ProcessRequest,
    PublishRequest,
    ReframeRequest,
    ScheduleClipRequest,
    SteerRequest,
    ValidateUrlRequest,
    _validate_drop_ranges,
    validate_public_url,
)
from clippyme.api.security import (
    ALLOWED_ORIGINS,
    enforce_api_token,
    enforce_rate_limit,
    require_trusted_config_request,
)
from clippyme.storage.config_store import (
    load_persistent_config,
    save_persistent_config,
    load_zernio_config,
)
from clippyme.domain.job_worker import make_workers
from clippyme.domain.history_service import scan_history, is_valid_job_id
from clippyme.api.config_routes import router as config_router
from clippyme.api.dubbing_routes import router as dubbing_router
from clippyme.api.studio_routes import router as studio_router
from clippyme.api.ugc_routes import router as ugc_router
from clippyme.api.highlight_routes import router as highlight_router
from clippyme.api.billing_routes import router as billing_router
from clippyme.api.trend_routes import make_trend_router
from clippyme.api.channel_routes import router as channel_router

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
DATA_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Initial load to env
save_persistent_config(load_persistent_config())

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "16384"))
# Default retention is 7 days — the frontend History tab is the
# authoritative source of truth for what the user considers "done".
# Aggressive auto-purge was destroying clips behind the user's back
# (jobs older than 1 hour vanished on the next cleanup tick, meaning
# every docker restart + 5 min wait blew away yesterday's work).
# Override via env: JOB_RETENTION_SECONDS (0 disables auto-purge).
JOB_RETENTION_SECONDS = int(os.environ.get("JOB_RETENTION_SECONDS", str(7 * 86400)))

# Application State
job_queue = asyncio.Queue(maxsize=50)
jobs: Dict[str, Dict] = {}
# Semaphore to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

# Job journal: persists the ACTIVE jobs to data/jobs_journal.json on every
# status transition so a restart can re-enqueue queued jobs and fail (or
# restore) interrupted ones instead of silently forgetting them.
JOURNAL_PATH = os.path.join(DATA_DIR, JOURNAL_FILENAME)
persist_jobs = make_journal_writer(jobs=jobs, path=JOURNAL_PATH)

# The per-job subprocess runner, bound to the shared jobs dict (thin-handler
# rule: the body lives in clippyme.domain.job_runner).
run_job = make_run_job(jobs=jobs, output_root=OUTPUT_DIR, on_change=persist_jobs)

# Multi-platform content monitor registry: concurrent asyncio tasks (one per
# platform:channel) that detect live streams / new VODs, submit them as normal
# jobs, and auto-publish clips with GLOBAL publish spacing. Bound to the same
# shared job state (thin-handler rule: logic lives in domain.live_monitor).
from clippyme.domain.live_monitor import LiveMonitorRegistry
live_monitor = LiveMonitorRegistry(
    jobs=jobs, job_queue=job_queue, output_dir=OUTPUT_DIR,
    upload_dir=UPLOAD_DIR, on_job_change=persist_jobs,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("pipeline worker interpreter: %s", worker_python())
    # Recover journalled jobs from the previous server life BEFORE the
    # dispatcher starts: queued jobs are re-enqueued, interrupted ones are
    # marked failed (or restored as completed when their result is on disk).
    # Runs on the event loop (not to_thread): asyncio.Queue.put_nowait is not
    # thread-safe, and the journal is small so the startup pause is negligible.
    try:
        recover_jobs(journal_path=JOURNAL_PATH, jobs=jobs,
                     job_queue=job_queue, output_root=OUTPUT_DIR)
    except Exception:
        logger.exception("Job journal recovery failed — starting with an empty queue")

    cleanup_jobs, process_queue, _run_job_wrapper = make_workers(
        jobs=jobs,
        job_queue=job_queue,
        concurrency_semaphore=concurrency_semaphore,
        run_job=run_job,
        output_dir=OUTPUT_DIR,
        upload_dir=UPLOAD_DIR,
        data_dir=DATA_DIR,
        job_retention_seconds=JOB_RETENTION_SECONDS,
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
    )
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    # Background auto-update for the auto-editor binary used by smartcut.py.
    # Failures are non-fatal — smartcut has an FFmpeg fallback path.
    from clippyme.integrations.auto_editor_updater import background_updater_loop
    ae_updater_task = asyncio.create_task(background_updater_loop())

    # Start Telegram AI Assistant & Remote Control listener
    from clippyme.domain.telegram_bot import TelegramBotListener
    telegram_listener = TelegramBotListener(
        jobs=jobs,
        job_queue=job_queue,
        output_dir=OUTPUT_DIR,
        upload_dir=UPLOAD_DIR,
        data_dir=DATA_DIR,
        run_job_fn=run_job,
    )
    telegram_task = asyncio.create_task(telegram_listener.start())

    # Trend Radar background poller: periodic AI trend research for US topics
    async def trend_radar_poller():
        from clippyme.api.trend_routes import load_trend_config
        from clippyme.domain.trend_discovery import run_trend_discovery, load_trend_radar
        while True:
            try:
                cfg = load_trend_config()
                if cfg.get("auto_scan", True):
                    radar = load_trend_radar()
                    last_scanned = radar.get("last_scanned")
                    interval_sec = cfg.get("interval_hours", 3) * 3600
                    should_scan = False
                    if not last_scanned:
                        should_scan = True
                    else:
                        try:
                            last_dt = datetime.fromisoformat(last_scanned)
                            if (datetime.now(timezone.utc) - last_dt).total_seconds() >= interval_sec:
                                should_scan = True
                        except Exception:
                            should_scan = True
                    if should_scan:
                        persisted = await asyncio.to_thread(load_persistent_config)
                        api_key = (persisted.get("GEMINI_API_KEY") if persisted else None) or os.environ.get("GEMINI_API_KEY")
                        await asyncio.to_thread(run_trend_discovery, api_key=api_key, categories=cfg.get("categories"))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Trend radar background poller encountered error: %s", exc)
            await asyncio.sleep(600)  # Check every 10 minutes

    trend_poller_task = asyncio.create_task(trend_radar_poller())

    # Bring back every monitor that was still marked resume_on_start when the
    # process last went down (durable auto-resume). Never fatal to startup —
    # a per-monitor failure stays visible via its status() instead.
    try:
        await live_monitor.auto_resume()
    except Exception:
        logger.exception("live monitor auto-resume failed")
    yield
    telegram_listener.stop()
    # Stop the live monitor first so its in-flight capture/publish tasks unwind
    # cleanly before we tear down the worker loops they depend on. shutdown()
    # (not stop()) so resume_on_start survives for the next auto-resume.
    try:
        await live_monitor.shutdown()
    except Exception:
        logger.exception("live monitor failed to stop cleanly")
    # Cancel ALL background tasks on shutdown — not just the updater. Leaving
    # the worker/cleanup loops pending blocks uvicorn's graceful exit and logs
    # "Task was destroyed but it is pending!" tracebacks.
    _bg_tasks = (worker_task, cleanup_task, ae_updater_task, telegram_task, trend_poller_task)
    for _t in _bg_tasks:
        _t.cancel()
    for _t in _bg_tasks:
        try:
            await _t
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("background task raised during shutdown")

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def _api_token_gate(request: Request, call_next):
    """Optional shared-secret auth for deliberate LAN deployments.

    Active only when CLIPPYME_API_TOKEN is set (default unset = no-op). Guards
    every /api route; the static media mounts (/videos, /thumbnails, /fonts)
    stay IP-open because <video>/<img>/FontFace requests can't attach custom
    headers. HTTPException is converted here because raise inside middleware
    bypasses FastAPI's exception handlers.
    """
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        try:
            enforce_api_token(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    """Add OWASP-recommended hardening headers to every response.

    - nosniff: block MIME-confusion attacks on served media/JSON.
    - frame-ancestors/X-Frame-Options: clickjacking defence.
    - Referrer-Policy: don't leak full URLs (job ids) to third parties.
    - CSP default-src 'none': the API serves JSON + media consumed by the
      separate Vite frontend; it should never itself be a script/HTML host.
    Ref: OWASP Secure Headers Project.
    """
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
    return response


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(request: Request, exc: RequestValidationError):
    from fastapi.encoders import jsonable_encoder
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc.errors())
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.exception_handler(ClippyMeError)
async def _clippyme_error_handler(request: Request, exc: ClippyMeError):
    """Map domain exceptions to HTTP responses so domain modules don't need
    to import FastAPI's HTTPException."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception):
    """Catch-all so a stray exception never leaks a traceback / internal path
    to the client. FastAPI's HTTPException is handled separately and is not
    affected by this."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Gemini-Key", "X-API-Token", "Idempotency-Key"],
)

# Mount static files for serving videos.
# The output directory also holds *_metadata.json (full transcripts + AI
# analysis) and source_*.mp4 (the raw 16:9 slices). Those are internal
# artifacts and must NOT be publicly downloadable — only the rendered clips,
# composed clips, covers and thumbnails are user-facing. SafeStaticFiles
# 404s the sensitive patterns while serving everything else as before.
class SafeStaticFiles(StaticFiles):
    # Only user-facing media belongs under /videos. Metadata, transcript
    # sidecars and temporary renderer files can contain full transcripts,
    # prompts, local paths or source footage and must never be served.
    _BLOCKED_SUFFIXES = (".json", ".ass", ".srt", ".tmp", ".part")
    _BLOCKED_PREFIXES = ("source_", ".")

    async def get_response(self, path, scope):
        leaf = os.path.basename(path.replace("\\", "/"))
        if leaf.endswith(self._BLOCKED_SUFFIXES) or leaf.startswith(self._BLOCKED_PREFIXES):
            raise HTTPException(status_code=404, detail="Not found")
        return await super().get_response(path, scope)


app.mount("/videos", SafeStaticFiles(directory=OUTPUT_DIR), name="videos")

# Mount static files for serving thumbnails
THUMBNAILS_DIR = os.path.join(OUTPUT_DIR, "thumbnails")
os.makedirs(THUMBNAILS_DIR, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=THUMBNAILS_DIR), name="thumbnails")

# Mount static files for serving fonts (used by subtitle preview in frontend)
app.mount("/fonts", StaticFiles(directory="fonts"), name="fonts")

# Config-family routes (keys, cookies, fonts, logo, zernio) live in their own
# router — they touch none of the job runtime state, so keeping them out of
# app.py lets this module stay focused on the job lifecycle.
app.include_router(config_router)
app.include_router(dubbing_router)
app.include_router(studio_router)
app.include_router(ugc_router)
app.include_router(highlight_router)
app.include_router(billing_router)
trend_router = make_trend_router(
    jobs=jobs,
    job_queue=job_queue,
    output_dir=OUTPUT_DIR,
    on_change=persist_jobs,
)
app.include_router(trend_router)
app.include_router(channel_router)


@app.get("/")
async def root():
    return {"status": "online", "message": "ClippyMe API is running"}

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    user: AuthUser = Depends(get_current_user),
):
    # CSRF/origin gate so a malicious page can't trigger compute jobs against
    # a locally-running backend. Same trust model as the config endpoints.
    require_trusted_config_request(request)
    # ~20 single-job submissions/min per client; compute-heavy, so throttle.
    enforce_rate_limit(request, "process", capacity=20, refill_per_sec=20 / 60)
    persisted = await asyncio.to_thread(load_persistent_config)
    api_key = (
        request.headers.get("X-Gemini-Key")
        or (persisted.get("GEMINI_API_KEY") if persisted else None)
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    # Handle JSON body via ProcessRequest for URL payloads. Pydantic
    # enforces the reframe_mode regex and the instructions length cap
    # before we hand anything to build_main_cmd. Multipart uploads keep
    # the manual form extraction because the file streaming path is
    # already using FastAPI's File/Form dependencies.
    instructions = None
    reframe_mode = None
    letterbox_zoom = None
    aspect = None
    language = None
    no_zoom = False
    skip_analysis = False
    model = None
    min_duration = None
    max_duration = None
    min_clips = None
    max_clips = None
    clip_type = None
    duration_mode = None
    highlights = False
    caption_style_default = None
    source_timeframe = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            validated = ProcessRequest.model_validate(body or {})
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        url = validated.url
        instructions = validated.instructions
        reframe_mode = validated.reframe_mode
        letterbox_zoom = validated.letterbox_zoom
        aspect = validated.aspect
        language = validated.language
        no_zoom = bool(validated.no_zoom)
        skip_analysis = bool(validated.skip_analysis)
        model = validated.model
        min_duration = validated.min_duration
        max_duration = validated.max_duration
        min_clips = validated.min_clips
        max_clips = validated.max_clips
        clip_type = validated.clip_type
        duration_mode = validated.duration_mode
        highlights = bool(validated.highlights)
        caption_style_default = validated.caption_style_default
        _tf = validated.source_timeframe
        source_timeframe = (_tf.start, _tf.end) if _tf else None

    # For multipart/form-data uploads, extract reframe_mode + language from form fields
    if "multipart/form-data" in content_type:
        form = await request.form()
        reframe_mode = form.get("reframe_mode", reframe_mode)
        letterbox_zoom = form.get("letterbox_zoom", letterbox_zoom)
        aspect = form.get("aspect", aspect)
        language = form.get("language", language)
        # Also honour the optional instructions field in multipart mode
        # so drag-and-drop uploads can pass AI directives just like URL
        # submissions (was previously ignored).
        instructions = form.get("instructions", instructions)
        no_zoom = str(form.get("no_zoom", "")).lower() in {"1", "true", "yes"} or no_zoom
        skip_analysis = str(form.get("skip_analysis", "")).lower() in {"1", "true", "yes"} or skip_analysis
        model = form.get("model", model) or None
        min_duration = float(form["min_duration"]) if form.get("min_duration") else min_duration
        max_duration = float(form["max_duration"]) if form.get("max_duration") else max_duration
        min_clips = int(form["min_clips"]) if form.get("min_clips") else min_clips
        max_clips = int(form["max_clips"]) if form.get("max_clips") else max_clips
        clip_type = form.get("clip_type", clip_type) or None
        duration_mode = form.get("duration_mode", duration_mode) or None
        highlights = str(form.get("highlights", "")).lower() in {"1", "true", "yes"} or highlights
        caption_style_default = form.get("caption_style_default", caption_style_default) or None
        _tf_raw = (form.get("source_timeframe", "") or "").strip()
        if _tf_raw:
            try:
                _tf_s, _tf_e = _tf_raw.split(",", 1)
                source_timeframe = (float(_tf_s), float(_tf_e))
            except ValueError:
                raise HTTPException(status_code=400, detail="source_timeframe must be 'start,end' seconds")
        # Validate the multipart values through the same schema for
        # consistency — we drop the url requirement since we're using
        # an uploaded file path.
        try:
            ProcessRequest.model_validate({
                "url": "https://upload.invalid/local",
                "reframe_mode": reframe_mode or None,
                "letterbox_zoom": letterbox_zoom or None,
                "aspect": aspect or None,
                "language": language or None,
                "instructions": instructions or None,
                "no_zoom": no_zoom,
                "skip_analysis": skip_analysis,
                "model": model or None,
                "min_duration": min_duration,
                "max_duration": max_duration,
                "min_clips": min_clips,
                "max_clips": max_clips,
                "clip_type": clip_type,
                "duration_mode": duration_mode,
                "highlights": highlights,
                "caption_style_default": caption_style_default,
                "source_timeframe": (
                    {"start": source_timeframe[0], "end": source_timeframe[1]}
                    if source_timeframe else None
                ),
            })
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors())

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    from clippyme.domain.quota_service import check_user_quota
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key

    input_path = None
    if not url:
        # Save uploaded file with a server-generated name. We deliberately
        # discard the client-supplied filename (path traversal risk) and only
        # preserve a sanitized extension whitelisted to known media formats.
        raw_ext = os.path.splitext(file.filename or "")[1].lower()
        allowed_ext = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
        if raw_ext not in allowed_ext:
            # Reject unknown extensions explicitly instead of silently
            # treating them as .mp4 (which produced confusing downstream
            # ffmpeg failures on non-video uploads).
            shutil.rmtree(job_output_dir, ignore_errors=True)
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Allowed: .mp4, .mov, .mkv, .webm, .m4v, .avi",
            )
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}{raw_ext}")
        try:
            upload_size = await stream_upload_within_limit(
                file, input_path, MAX_FILE_SIZE_MB * 1024 * 1024
            )
        except FileTooLarge as exc:
            await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
            raise
        if upload_size == 0:
            await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
            try:
                await asyncio.to_thread(os.remove, input_path)
            except FileNotFoundError:
                pass
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        # Keep uploaded file available for highlights extraction
        if highlights:
            dest_input = os.path.join(job_output_dir, f"source_{job_id}{raw_ext}")
            shutil.copyfile(input_path, dest_input)

    try:
        cmd = build_main_cmd(
            url=url,
            input_path=input_path,
            output_dir=job_output_dir,
            instructions=instructions,
            reframe_mode=reframe_mode,
            letterbox_zoom=letterbox_zoom,
            aspect=aspect,
            cookies_path=os.path.join("data", "cookies.txt"),
            language=language,
            no_zoom=no_zoom,
            skip_analysis=skip_analysis,
            model=model,
            min_duration=min_duration,
            max_duration=max_duration,
            min_clips=min_clips,
            max_clips=max_clips,
            clip_type=clip_type,
            duration_mode=duration_mode,
            highlights=highlights,
            caption_style_default=caption_style_default,
            source_timeframe=source_timeframe,
        )
    except ValueError as exc:
        await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
        if input_path:
            try:
                await asyncio.to_thread(os.remove, input_path)
            except FileNotFoundError:
                pass
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Queue-full rollback removes both the output directory and uploaded input.
    await submit_job(
        jobs=jobs, job_queue=job_queue, job_id=job_id,
        cmd=cmd, env=env, job_output_dir=job_output_dir,
        on_change=persist_jobs, cleanup_paths=(input_path,), input_path=input_path,
        user_id=user.id,
    )

    return {"job_id": job_id, "status": "queued"}


@app.post("/api/batch")
async def batch_process(
    req: BatchRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    """Submit multiple URLs for batch processing. Each URL becomes a separate job."""
    require_trusted_config_request(request)
    # Each batch can enqueue up to 20 jobs, so limit batch calls more tightly.
    enforce_rate_limit(request, "batch", capacity=10, refill_per_sec=10 / 60)
    persisted = await asyncio.to_thread(load_persistent_config)
    api_key = (
        request.headers.get("X-Gemini-Key")
        or (persisted.get("GEMINI_API_KEY") if persisted else None)
        or os.environ.get("GEMINI_API_KEY")
    )
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")

    from clippyme.domain.quota_service import check_user_quota
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    batch_jobs = []

    for url in req.urls:
        url = url.strip()
        if not url:
            continue

        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        try:
            cmd = build_main_cmd(
                url=url,
                output_dir=job_output_dir,
                instructions=req.instructions,
                reframe_mode=req.reframe_mode,
                letterbox_zoom=req.letterbox_zoom,
                aspect=getattr(req, "aspect", None),
                cookies_path=os.path.join("data", "cookies.txt"),
                language=getattr(req, "language", None),
                no_zoom=bool(getattr(req, "no_zoom", False)),
                skip_analysis=bool(getattr(req, "skip_analysis", False)),
                model=getattr(req, "model", None),
                min_duration=getattr(req, "min_duration", None),
                max_duration=getattr(req, "max_duration", None),
                min_clips=getattr(req, "min_clips", None),
                max_clips=getattr(req, "max_clips", None),
                clip_type=getattr(req, "clip_type", None),
                duration_mode=getattr(req, "duration_mode", None),
            )
        except ValueError as exc:
            # This item's output dir was already created above but it never
            # made it into `jobs` — clean it up so a bad URL can't orphan a dir.
            await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
            raise HTTPException(status_code=400, detail=str(exc))

        env = os.environ.copy()
        env["GEMINI_API_KEY"] = api_key

        try:
            await submit_job(
                jobs=jobs, job_queue=job_queue, job_id=job_id,
                cmd=cmd, env=env, job_output_dir=job_output_dir, batch=True,
                on_change=persist_jobs, user_id=user.id,
            )
            batch_jobs.append({"url": url, "job_id": job_id})
        except QueueFullError:
            # Only the item that failed to enqueue was cleaned up (by
            # submit_job) — already enqueued jobs stay running. Stop adding
            # more; the queue is full. Mirrors the single /api/process path.
            break

    if not batch_jobs:
        raise HTTPException(status_code=400, detail="No valid URLs provided or queue is full.")

    return {"jobs": batch_jobs, "total": len(batch_jobs)}


def _check_job_ownership(job_dict: dict, user: AuthUser) -> None:
    if user.is_admin:
        return
    job_user = job_dict.get("user_id", "default_user")
    if job_user != user.id:
        raise HTTPException(status_code=404, detail="Job not found")


def _verify_job_ownership(job_id: str, user: AuthUser) -> None:
    """Verify that user owns the job (or is an admin).

    Checks both in-memory job state and on-disk runtime metadata.
    Raises HTTPException(404) if job does not exist or belongs to another user.
    """
    if user.is_admin:
        return
    if job_id in jobs:
        _check_job_ownership(jobs[job_id], user)
        return

    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")

    job_user = None
    runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
    if os.path.isfile(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                rt_data = json.load(f)
            job_user = rt_data.get("user_id")
        except Exception:
            pass

    if not job_user:
        meta_files = glob.glob(os.path.join(job_dir, "*_metadata.json"))
        if meta_files:
            try:
                with open(meta_files[0], "r", encoding="utf-8") as f:
                    meta_data = json.load(f)
                job_user = meta_data.get("user_id")
            except Exception:
                pass

    job_user = job_user or "default_user"
    if job_user != user.id:
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/jobs/active")
async def get_active_jobs(user: AuthUser = Depends(get_current_user)):
    active = []
    for j_id, j in jobs.items():
        if not user.is_admin and j.get("user_id", "default_user") != user.id:
            continue
        if j.get("status") in ("queued", "processing", "paused"):
            cmd = j.get("cmd") or []
            source = j.get("input_path")
            if not source and "-u" in cmd:
                try:
                    source = cmd[cmd.index("-u") + 1]
                except (ValueError, IndexError):
                    source = None
            active.append({
                "jobId": j_id,
                "status": j["status"],
                "source": source,
            })
    return {"jobs": active}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str, user: AuthUser = Depends(get_current_user)):
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    if job_id not in jobs:
        # Resilient fallback: look up state from output directory on disk
        job_dir = os.path.join(OUTPUT_DIR, job_id)
        if os.path.isdir(job_dir):
            runtime_path = os.path.join(job_dir, ".clippyme_runtime.json")
            if os.path.isfile(runtime_path):
                try:
                    with open(runtime_path, "r", encoding="utf-8") as f:
                        rt_data = json.load(f)
                    if not user.is_admin:
                        rt_user = rt_data.get("user_id", "default_user")
                        if rt_user != user.id:
                            raise HTTPException(status_code=404, detail="Job not found")
                    status = "complete" if rt_data.get("completed_at") else (
                        "failed" if rt_data.get("failed_at") else "processing"
                    )
                    return {
                        "status": status,
                        "logs": rt_data.get("logs", []),
                        "result": rt_data.get("artifacts"),
                    }
                except HTTPException:
                    raise
                except Exception:
                    pass
            try:
                files = os.listdir(job_dir)
                if files and user.is_admin:
                    return {
                        "status": "complete",
                        "logs": ["Job loaded from storage"],
                        "result": {},
                    }
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    _check_job_ownership(job, user)
    return {
        "status": job['status'],
        "logs": job.get('logs', [])[-500:],
        "result": job.get('result')
    }


def _estimate_eta_from_stages(state: dict) -> int | None:
    """ETA in seconds, estimated from completed stage durations.

    Method: for each completed stage with a recorded duration, take the
    progress it covered (STAGE_PROGRESS[stage] minus STAGE_PROGRESS of the
    previous stage in STAGE_ORDER) and divide its seconds by that coverage
    to get a seconds-per-progress-point rate. The mean rate over completed
    stages is extrapolated over the remaining progress (100 - current
    progress). Returns None when there is insufficient data (no completed
    stage durations, fewer than 5 progress points covered, or progress
    already at/above 100) rather than guessing.
    """
    durations = state.get("stage_durations") or {}
    completed = state.get("completed_stages") or []
    try:
        progress = float(state.get("progress") or 0)
    except (TypeError, ValueError):
        progress = 0.0
    if progress >= 100:
        return None
    total_points = 0.0
    total_seconds = 0.0
    for stage in completed:
        if stage not in STAGE_PROGRESS or stage not in durations:
            continue
        try:
            idx = STAGE_ORDER.index(stage)
        except ValueError:
            continue
        prev = STAGE_ORDER[idx - 1] if idx > 0 else None
        prev_points = STAGE_PROGRESS.get(prev, 0) if prev else 0
        points = STAGE_PROGRESS[stage] - prev_points
        try:
            secs = float(durations[stage] or 0)
        except (TypeError, ValueError):
            continue
        if points > 0 and secs > 0:
            total_points += points
            total_seconds += secs
    if total_points < 5 or total_seconds <= 0:
        return None
    remaining = 100.0 - progress
    if remaining <= 0:
        return None
    return max(0, int(remaining * (total_seconds / total_points)))


@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str, user: AuthUser = Depends(get_current_user)):
    """Live progress-pill data: percent, stage, ETA. Read-only.

    Sources the job's ``.clippyme_runtime.json`` (written atomically by the
    orchestrator); ``/api/status`` behavior is untouched.
    """
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    state = load_runtime_state(job_dir)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        progress = int(state.get("progress") or 0)
    except (TypeError, ValueError):
        progress = 0
    return {
        "progress": max(0, min(100, progress)),
        "stage": state.get("stage") or "unknown",
        "eta_seconds": _estimate_eta_from_stages(state),
        "detail": state.get("detail"),
    }


_validate_probe_semaphore = asyncio.Semaphore(3)


def _probe_url_metadata(url: str) -> dict:
    """Blocking yt-dlp metadata probe. Runs in a thread; never downloads.

    Uses the downloader's established client strategy (web_embedded player,
    no cookies -- the proven bot-check bypass) with metadata-only
    extraction. Raises on any failure; callers translate that into
    ``downloadable: false`` with a reason.
    """
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": 10,
        "retries": 1,
        "fragment_retries": 1,
        "force_ipv4": True,
        "cachedir": False,
        "remote_components": ["ejs:github"],
        "extractor_args": {"youtube": {"player_client": ["web_embedded"]}},
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError("probe returned no metadata")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if not entries:
            raise RuntimeError("playlist has no playable entries")
        info = entries[0]
    return {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
    }


@app.post("/api/validate-url")
async def validate_url(
    req: ValidateUrlRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    """Check a video URL is valid and downloadable. Read-only, never downloads.

    Fast probe with a hard 15 s cap: bot-walls and timeouts report
    ``downloadable: false`` with a reason instead of hanging the request.
    """
    require_trusted_config_request(request)
    try:
        url = validate_public_url(req.url)
    except ValueError as exc:
        return {
            "valid": False,
            "downloadable": False,
            "reason": str(exc),
            "title": None,
            "duration": None,
            "thumbnail": None,
        }
    async with _validate_probe_semaphore:
        try:
            info = await asyncio.wait_for(
                asyncio.to_thread(_probe_url_metadata, url), timeout=15.0
            )
        except asyncio.TimeoutError:
            return {
                "valid": True,
                "downloadable": False,
                "reason": "probe timed out after 15s (site bot protection likely)",
                "title": None,
                "duration": None,
                "thumbnail": None,
            }
        except Exception as exc:  # noqa: BLE001 - probe surface is intentionally broad
            return {
                "valid": True,
                "downloadable": False,
                "reason": (str(exc) or "probe failed")[-300:],
                "title": None,
                "duration": None,
                "thumbnail": None,
            }
    return {
        "valid": True,
        "downloadable": True,
        "reason": None,
        "title": info.get("title"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
    }

@app.post("/api/cancel/{job_id}")
async def cancel_job(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Cancel a running job by killing its subprocess."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    _check_job_ownership(jobs[job_id], user)
    try:
        return await cancel_job_action(job_id, jobs[job_id])
    finally:
        persist_jobs()


@app.post("/api/pause/{job_id}")
async def pause_job(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Suspend a running job's process tree (SIGSTOP/SuspendThread via psutil)."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    _check_job_ownership(job, user)
    if not job_control.can_pause(job['status']):
        raise HTTPException(status_code=400, detail="Job cannot be paused")

    proc = job.get('process')
    if not (proc and proc.poll() is None):
        raise HTTPException(status_code=409, detail="Job has no running process")

    n = await asyncio.to_thread(job_control.suspend_tree, proc.pid)
    job['status'] = 'paused'
    job['logs'].append(f"Job paused by user ({n} process(es) suspended).")
    logger.info("Job %s paused (%d procs)", job_id, n)
    persist_jobs()
    return {"success": True, "status": "paused"}


@app.post("/api/resume/{job_id}")
async def resume_job(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Resume a paused job's process tree."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    _check_job_ownership(job, user)
    if not job_control.can_resume(job['status']):
        raise HTTPException(status_code=400, detail="Job is not paused")

    proc = job.get('process')
    if not (proc and proc.poll() is None):
        raise HTTPException(status_code=409, detail="Job has no running process")

    n = await asyncio.to_thread(job_control.resume_tree, proc.pid)
    job['status'] = 'processing'
    job['logs'].append(f"Job resumed by user ({n} process(es) resumed).")
    logger.info("Job %s resumed (%d procs)", job_id, n)
    persist_jobs()
    return {"success": True, "status": "processing"}


@app.post("/api/stop/{job_id}")
async def stop_job(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Graceful stop: kill the subprocess but KEEP finished clips.

    Unlike ``/api/cancel`` (hard discard), this promotes the partial result to
    final so the user can still view/edit/publish the clips already rendered.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    _check_job_ownership(jobs[job_id], user)
    try:
        return await stop_job_action(job_id, jobs[job_id])
    finally:
        persist_jobs()


@app.post("/api/retry/{job_id}")
@app.post("/api/jobs/{job_id}/retry")
async def retry_job_endpoint(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Resume and retry a failed or interrupted job directly from its latest checkpoints."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    persisted = await asyncio.to_thread(load_persistent_config)
    api_key = (
        request.headers.get("X-Gemini-Key")
        or (persisted.get("GEMINI_API_KEY") if persisted else None)
        or os.environ.get("GEMINI_API_KEY")
    )

    from clippyme.domain.job_submission import retry_job_action
    try:
        return await retry_job_action(
            jobs=jobs,
            job_queue=job_queue,
            job_id=job_id,
            output_root=OUTPUT_DIR,
            api_key=api_key,
            on_change=persist_jobs,
        )
    except (NotFoundError, ValidationError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@app.post("/api/jobs/{job_id}/rescore")
async def rescore_job_endpoint(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Rescore clips in an existing job with AI virality scores and titles."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    persisted = await asyncio.to_thread(load_persistent_config)
    api_key = (
        request.headers.get("X-Gemini-Key")
        or (persisted.get("GEMINI_API_KEY") if persisted else None)
        or os.environ.get("GEMINI_API_KEY")
    )

    from clippyme.domain.rescore_service import rescore_job
    try:
        return await asyncio.to_thread(
            rescore_job,
            job_id=job_id,
            output_dir=OUTPUT_DIR,
            api_key=api_key,
        )
    except ClippyMeError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    except Exception as exc:
        logger.error("Unexpected error rescoring job %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


_STEER_PAST_ANALYSIS_STAGES = frozenset({
    "cutting", "reframing", "quality", "finalizing",
    "completed", "failed", "cancelled", "stopped",
})


@app.post("/api/jobs/{job_id}/steer")
async def steer_job(job_id: str, req: SteerRequest, request: Request,
                   user: AuthUser = Depends(get_current_user)):
    """Phase B1: steer a running job's moment selection with a prompt.

    The prompt is persisted on the job's runtime state; the orchestrator
    consumes it at the analysis/moment-selection checkpoint and appends it
    to the Gemini instructions. If the job already passed analysis, the
    steer is REJECTED (409) with a clear message -- never silently ignored.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    state = load_runtime_state(job_dir)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    from clippyme.domain.runtime_state import RuntimeState

    stage = state.get("stage") or "unknown"
    completed = set(state.get("completed_stages") or [])
    past_checkpoint = bool(state.get("steering_checkpoint_passed"))
    if past_checkpoint or "analyzing" in completed or stage in _STEER_PAST_ANALYSIS_STAGES:
        missed = state.get("steering_missed") or []
        return JSONResponse(
            status_code=409,
            content={
                "accepted": False,
                "stage": stage,
                "reason": (
                    "job is already past the steering point (moment selection "
                    "has run) -- this prompt was NOT applied; it will apply "
                    "to your next job"
                ),
                "missed_steering": len(missed),
            },
        )
    try:
        pending = await asyncio.to_thread(
            _steer_add_prompt, job_dir, job_id, req.prompt)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "accepted": True,
        "stage": stage,
        "steering_pending": pending,
        "note": "prompt will be applied at the moment-selection checkpoint",
    }


def _steer_add_prompt(job_dir: str, job_id: str, prompt: str) -> int:
    """Blocking helper: append a steering prompt to the job runtime state."""
    from clippyme.domain.runtime_state import RuntimeState

    rs = RuntimeState(job_dir, job_id=job_id)
    return rs.add_steering(prompt)


@app.post("/api/smartcut/{job_id}/{clip_index}")
async def smart_cut_clip(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Generate a smart-cut version of a clip (silences + filler words removed)."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "smartcut", capacity=20, refill_per_sec=20 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(resolve_clip, job_id, clip_index, OUTPUT_DIR)
    # Optional manual-trim spans (flycut-style interactive cut). Legacy callers
    # POST no body — tolerate that and fall back to pure auto Smart Cut.
    drop_ranges = None
    raw_body = await request.body()
    if raw_body:
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Malformed JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=422, detail="Request body must be a JSON object")
        drop_ranges = body.get("drop_ranges")
    # This raw-body path bypasses Pydantic, so apply the same bound check the
    # ComposeRequest/PublishRequest schemas use — rejects an oversized or
    # malformed list before the engine iterates it (DoS gate).
    try:
        _validate_drop_ranges(drop_ranges)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid drop_ranges: {exc}")
    return await run_smart_cut(
        job_id=job_id,
        clip_index=clip_index,
        resolved=resolved,
        drop_ranges=drop_ranges,
    )


@app.get("/api/transcript/{job_id}/{clip_index}")
async def clip_transcript(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Per-clip transcript segments (clip-relative seconds) for the manual-trim
    UI. Each segment is {index, text, start, end}; the frontend lets the user
    mark segments to drop and posts the resulting spans as `drop_ranges`."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    transcript = resolved.metadata.get("transcript") or {}
    clip = resolved.clip_info
    start, end = clip.get("start", 0), clip.get("end", 0)
    from clippyme.domain.smartcut import clip_transcript_segments
    segments = clip_transcript_segments(transcript, start, end)
    return {
        "segments": segments,
        "duration": round(max(0.0, end - start), 3),
        "language": transcript.get("language", "en") if isinstance(transcript, dict) else "en",
    }


@app.post("/api/edit-ai/{job_id}/{clip_index}")
async def edit_clip_ai(
    job_id: str,
    clip_index: int,
    req: EditAIRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
    api_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    """Conversational clip editing.

    mode="trim" (default, unchanged): plain-English instruction -> Gemini ->
    clip-relative spans to remove; the returned `drop_ranges` feed the SAME
    manual-trim machinery as the tap-to-cut UI.

    mode="patch" (Phase B2 copilot): instruction -> Gemini -> a validated
    ClipProject patch (hook text, caption style/position, word corrections,
    crop nudge, grade preset). The patch passes `validate_project()` +
    layer-order checks; invalid patches 400 and the stored project is never
    touched. The patched project is RETURNED, not saved -- the client saves
    it explicitly via PUT /api/project.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    clip = resolved.clip_info
    start, end = clip.get("start", 0), clip.get("end", 0)
    duration = round(max(0.0, end - start), 3)

    transcript = resolved.metadata.get("transcript") or {}
    from clippyme.domain.smartcut import clip_transcript_segments
    segments = clip_transcript_segments(transcript, start, end)

    cfg = load_persistent_config() or {}
    key = api_key or os.environ.get("GEMINI_API_KEY") or cfg.get("GEMINI_API_KEY")
    model = req.model or cfg.get("GEMINI_MODEL") or "gemini-3.5-flash"
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API key not configured")

    if req.mode == "patch":
        return await _edit_ai_patch(
            job_id=job_id, clip_index=clip_index, resolved=resolved,
            segments=segments, instruction=req.instruction,
            api_key=key, model=model)

    from clippyme.domain.clip_edit_ai import suggest_drops
    result = await asyncio.to_thread(
        suggest_drops,
        api_key=key,
        model=model,
        segments=segments,
        instruction=req.instruction,
        clip_duration=duration,
    )
    return {"drop_ranges": result["drops"], "explanation": result["explanation"]}


async def _edit_ai_patch(*, job_id, clip_index, resolved, segments,
                         instruction, api_key, model):
    """Phase B2: NL instruction -> validated ClipProject patch (not saved)."""
    from pydantic import ValidationError as PydanticValidationError

    from clippyme.domain.clip_project import validate_project
    from clippyme.domain.copilot_patch import (
        apply_patch_to_project_data,
        project_summary_for_prompt,
        suggest_project_patch,
    )
    from clippyme.domain.compose import ComposeOrderError
    from clippyme.domain.project_render import get_project_path

    path = await asyncio.to_thread(
        get_project_path, resolved.job_dir, clip_index, None)
    if not path:
        raise HTTPException(
            status_code=404, detail="No project found for this clip")
    try:
        with open(path, "r", encoding="utf-8") as f:
            project_data = json.load(f)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read clip project: {exc}")
    try:
        project = validate_project(project_data)
    except PydanticValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Stored clip project is invalid: {exc.errors()}")

    summary = project_summary_for_prompt(project)
    try:
        result = await asyncio.to_thread(
            suggest_project_patch,
            api_key=api_key,
            model=model,
            project_summary=summary,
            segments=segments,
            instruction=instruction,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not understand the AI edit: {exc}")

    patch = result["patch"]
    if not patch:
        return {
            "applied": False,
            "patch": {},
            "project": project.model_dump(),
            "explanation": result["explanation"],
        }
    try:
        patched = await asyncio.to_thread(
            apply_patch_to_project_data, project.model_dump(), patch)
    except (ValueError, PydanticValidationError, ComposeOrderError) as exc:
        detail = getattr(exc, "detail", None) or str(exc)
        raise HTTPException(status_code=400, detail=f"Invalid patch: {detail}")
    return {
        "applied": True,
        "patch": patch,
        "project": patched,
        "explanation": result["explanation"],
    }


@app.post("/api/generate-metadata/{job_id}")
async def generate_all_metadata_endpoint(
    job_id: str,
    request: Request,
    user: AuthUser = Depends(get_current_user),
    api_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    """Batch generate situational platform metadata and captions for all clips in a job."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    cfg = load_persistent_config() or {}
    key = api_key or os.environ.get("GEMINI_API_KEY") or cfg.get("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API key not configured")

    from clippyme.domain.metadata_generator import generate_all_clips_metadata
    result = await asyncio.to_thread(
        generate_all_clips_metadata,
        job_id=job_id,
        output_dir=OUTPUT_DIR,
        api_key=key,
        force=True,
    )
    return result


@app.post("/api/generate-metadata/{job_id}/{clip_index}")
async def generate_clip_metadata_endpoint(
    job_id: str,
    clip_index: int,
    req: GenerateMetadataRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
    api_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
):
    """AI speaker identification, trending hashtags, title, and viral caption generation."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    clip = resolved.clip_info
    start, end = clip.get("start", 0), clip.get("end", 0)

    transcript = resolved.metadata.get("transcript") or {}
    from clippyme.domain.smartcut import clip_transcript_segments
    segments = clip_transcript_segments(transcript, start, end)
    clip_transcript_text = " ".join(s.get("text", "") for s in segments if s.get("text"))

    video_title = resolved.metadata.get("title") or clip.get("video_title_for_youtube_short") or ""
    uploader = resolved.metadata.get("uploader") or ""

    from clippyme.pipeline.audience_intel import load_audience_intel
    audience_intel = load_audience_intel(resolved.job_dir)
    source_info = resolved.metadata.get("source_info") or {}

    cfg = load_persistent_config() or {}
    key = api_key or os.environ.get("GEMINI_API_KEY") or cfg.get("GEMINI_API_KEY")
    model = req.model or cfg.get("GEMINI_LITE_MODEL") or "gemini-3.5-flash-lite"
    if not key:
        raise HTTPException(status_code=400, detail="Gemini API key not configured")

    from clippyme.domain.metadata_generator import generate_clip_metadata
    result = await asyncio.to_thread(
        generate_clip_metadata,
        api_key=key,
        model=model,
        clip_transcript=clip_transcript_text,
        start=start,
        end=end,
        video_title=video_title,
        uploader=uploader,
        instructions=req.instruction or "",
        audience_intel=audience_intel,
        source_info=source_info,
    )

    # Persist the newly generated metadata back to disk
    if isinstance(result, dict) and result.get("platforms"):
        clip["platforms"] = result["platforms"]
        clip["speaker_name"] = result.get("speaker_name") or clip.get("speaker_name", "")
        clip["hashtags"] = result.get("hashtags") or clip.get("hashtags", [])
        clip["caption"] = result.get("caption") or clip.get("caption", "")
        clip["video_description_for_tiktok"] = result["platforms"]["tiktok"]["caption"]
        clip["video_description_for_instagram"] = result["platforms"]["instagram"]["caption"]
        clip["video_description"] = result["platforms"]["youtube"]["description"]
        if result.get("title") and not clip.get("video_title_for_youtube_short"):
            clip["video_title_for_youtube_short"] = result["title"]
        try:
            from clippyme.domain.job_artifacts import save_job_metadata
            save_job_metadata(resolved.metadata_path, resolved.metadata)
        except Exception as exc:
            logger.warning("Failed to persist updated clip metadata: %s", exc)

    return result


@app.post("/api/reframe/{job_id}/{clip_index}")
async def reframe_clip(
    job_id: str,
    clip_index: int,
    req: ReframeRequest,
    request: Request,
    user: AuthUser = Depends(get_current_user),
):
    """Switch a clip between reframe modes (auto / subject / disabled) after generation.

    Requires the per-clip 16:9 source slice (``source_<clip>.mp4``) to still
    exist on disk. Spawns ``main.py --reframe-only`` as a subprocess to reuse
    the exact same reframing / zoom / normalize / cover pipeline the initial
    run used. Updates metadata.json and the in-memory job state so the
    dashboard picks up the new video URL on the next poll.
    """
    require_trusted_config_request(request)
    enforce_rate_limit(request, "reframe", capacity=20, refill_per_sec=20 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    _verify_job_ownership(job_id, user)
    mode = (req.reframe_mode or "auto").strip().lower()
    if mode not in ("auto", "disabled", "subject", "object", "split", "screencast"):
        raise HTTPException(status_code=400, detail="reframe_mode must be 'auto', 'subject', 'disabled', 'split', or 'screencast'")
    # 'object' is the legacy name for 'subject' — normalize so the subprocess
    # argv + metadata are written with the canonical value.
    mode = canonical_reframe_mode(mode)

    # Everything from metadata resolution through the subprocess run lives in
    # the domain helper (thin-handler rule); ClippyMeError subclasses raised
    # there are mapped to HTTP responses by the app-level exception handler.
    return await run_reframe(
        job_id=job_id, clip_index=clip_index, mode=mode,
        letterbox_zoom=req.letterbox_zoom,
        output_root=OUTPUT_DIR, jobs=jobs,
    )


@app.get("/api/history")
async def list_history(request: Request, user: AuthUser = Depends(get_current_user)):
    """Scan output/ for past jobs with metadata files."""
    require_trusted_config_request(request)
    filter_user = None if user.is_admin else user.id
    return {"jobs": await asyncio.to_thread(scan_history, OUTPUT_DIR, user_id=filter_user)}

@app.get("/api/storage/breakdown")
async def get_storage_stats(request: Request):
    """Return categorized disk usage breakdown across output/ and uploads/."""
    require_trusted_config_request(request)
    from clippyme.domain.job_artifacts import get_storage_breakdown
    return await asyncio.to_thread(get_storage_breakdown, OUTPUT_DIR, UPLOAD_DIR)

@app.post("/api/storage/cleanup")
async def trigger_storage_cleanup(request: Request):
    """Run manual storage cleanup pass: purges partial downloads, ASR audio, uploads, and optionally source videos."""
    require_trusted_config_request(request)
    body = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
        except Exception:
            body = {}

    purge_raw_sources = bool(
        body.get("purge_raw_sources")
        or request.query_params.get("purge_raw_sources") in ("1", "true", "yes")
        or body.get("mode") == "deep"
    )
    source_max_days = body.get("source_max_days") or request.query_params.get("source_max_days")
    source_max_age_seconds = float(source_max_days) * 86400 if source_max_days is not None else None

    from clippyme.domain.job_artifacts import run_storage_cleanup
    from clippyme.domain.job_worker import active_input_paths
    from clippyme.domain.job_control import ACTIVE_STATES
    protected = active_input_paths(jobs)
    active_job_ids = {jid for jid, job in jobs.items() if job.get("status") in ACTIVE_STATES}

    res = await asyncio.to_thread(
        run_storage_cleanup,
        output_dir=OUTPUT_DIR,
        upload_dir=UPLOAD_DIR,
        purge_raw_sources=purge_raw_sources,
        source_max_age_seconds=source_max_age_seconds,
        active_paths=protected,
        protected_job_ids=active_job_ids,
        partial_min_age_seconds=1800.0,
    )
    return res

@app.delete("/api/history/{job_id}")
async def delete_history(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Delete a job's output directory and all its files."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    job = jobs.get(job_id)
    if job is not None and not job_control.can_purge(job.get("status")):
        raise HTTPException(
            status_code=409,
            detail="Active jobs must be stopped or cancelled before deletion",
        )
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if os.path.islink(job_dir):
        raise HTTPException(status_code=409, detail="Refusing to delete a symbolic link")
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found on disk")
    try:
        await asyncio.to_thread(shutil.rmtree, job_dir)
    except OSError as exc:
        logger.error("Could not delete history directory %s", job_dir, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not delete job files") from exc
    if job_id in jobs:
        input_path = jobs[job_id].get("input_path")
        del jobs[job_id]
        if input_path:
            try:
                await asyncio.to_thread(os.remove, input_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.warning("Could not remove uploaded input %s", input_path, exc_info=True)
        persist_jobs()
    # Sweep any upload file matching job_id from uploads/
    for up_match in glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}*")):
        try:
            if os.path.isfile(up_match):
                await asyncio.to_thread(os.remove, up_match)
        except OSError:
            pass
    logger.info("Deleted job %s and all files", job_id)
    return {"success": True}

@app.post("/api/compose/{job_id}/{clip_index}")
async def compose_clip(job_id: str, clip_index: int, req: ComposeRequest, request: Request, user: AuthUser = Depends(get_current_user)):
    """Compose a final video from active toggle layers (Smart Cut → Hook → Subtitles)."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "compose", capacity=30, refill_per_sec=30 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    resolved = await asyncio.to_thread(resolve_clip, job_id, clip_index, OUTPUT_DIR)

    # Phase 2 (clip editor): a full ClipProject dict wins over legacy toggles.
    if req.project is not None:
        from pydantic import ValidationError as PydanticValidationError
        from clippyme.domain.clip_project import validate_project
        from clippyme.domain.compose import ComposeOrderError
        from clippyme.domain.project_render import (
            ProjectRenderError, render_project)
        try:
            project = validate_project(req.project)
        except PydanticValidationError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clip project: {exc.errors()}",
            )
        try:
            result = await render_project(
                job_id=job_id,
                clip_index=clip_index,
                project=project,
                job_dir=resolved.job_dir,
                metadata=resolved.metadata,
                clip_info=resolved.clip_info,
                naming="versioned",
            )
        except (ProjectRenderError, ComposeOrderError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except (HTTPException, ClippyMeError):
            raise
        except Exception as e:
            logger.error("Project render error for job %s clip %d: %s",
                         job_id, clip_index, e)
            raise HTTPException(status_code=500, detail="Compose pipeline failed")
        return {
            "composed_url": f"/videos/{job_id}/{result.output_basename}",
            "version": result.version,
            "deduped": result.deduped,
        }

    # Legacy toggles path: map to an equivalent ClipProject and render through
    # the SAME project-render path (byte-identical output, existing naming).
    try:
        from clippyme.domain.clip_project import toggles_to_project
        from clippyme.domain.project_render import (
            ProjectRenderError, render_project)
        from clippyme.pipeline.media_qa import probe_media as _probe
        _src_probe = _probe(resolved.clip_path)
        _src_dur = float(_src_probe.get("duration") or 0)
        legacy_project = toggles_to_project(
            job_id=job_id,
            clip_index=clip_index,
            base_clip=os.path.basename(resolved.clip_path),
            source_file=os.path.basename(resolved.clip_path),
            source_width=int(_src_probe.get("width") or 608),
            source_height=int(_src_probe.get("height") or 1080),
            source_duration=_src_dur,
            render_width=int(_src_probe.get("width") or 608),
            render_height=int(_src_probe.get("height") or 1080),
            fps=int(_src_probe.get("fps") or 30),
            toggles=req.toggles,
            hook_params=req.hook_params,
            subtitle_params=req.subtitle_params,
            logo_params=req.logo_params,
            grade_params=req.grade_params,
            banner_params=req.banner_params,
            drop_ranges=req.drop_ranges,
            metadata=resolved.metadata,
        )
        result = await render_project(
            job_id=job_id,
            clip_index=clip_index,
            project=legacy_project,
            job_dir=resolved.job_dir,
            metadata=resolved.metadata,
            clip_info=resolved.clip_info,
            drop_ranges=req.drop_ranges,
            naming="legacy",
        )
        return {"composed_url": f"/videos/{job_id}/{result.output_basename}"}
    except ProjectRenderError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (HTTPException, ClippyMeError):
        raise
    except Exception as e:
        logger.error("Compose error for job %s clip %d: %s", job_id, clip_index, e)
        raise HTTPException(status_code=500, detail="Compose pipeline failed")


# ---------------------------------------------------------------------------
# Clip editor project (Phase 2) endpoints
# ---------------------------------------------------------------------------

@app.get("/api/project/{job_id}/{clip_index}")
async def get_clip_project(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user), version: int | None = None):
    """Return the latest clip-project JSON for a clip (or ``?version=N``)."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    from clippyme.domain.project_render import get_project_path
    path = get_project_path(resolved.job_dir, clip_index, version)
    if not path:
        raise HTTPException(
            status_code=404, detail="No project found for this clip")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/project/{job_id}/{clip_index}/versions")
async def list_clip_project_versions(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """List recorded project versions for a clip (newest last). Read-only."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    from clippyme.domain.project_render import _load_version_map
    entry = _load_version_map(resolved.job_dir).get(str(clip_index), {})
    versions = []
    for ver_key in sorted((entry.get("versions") or {}).keys(), key=lambda k: int(k)):
        rec = (entry.get("versions") or {})[ver_key] or {}
        out = rec.get("output_file") or ""
        versions.append({
            "version": int(rec.get("version") or ver_key),
            "origin": rec.get("origin") or "auto",
            "created_at": rec.get("created_at") or "",
            "project_file": rec.get("project_file") or "",
            "output_file": out,
            "output_url": f"/videos/{job_id}/{out}" if out else "",
        })
    return {"versions": versions, "latest": int(entry.get("latest") or 0)}


@app.put("/api/project/{job_id}/{clip_index}")
async def save_clip_project(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Save a draft project version: validate, bump version, NO render."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    body = await request.json()
    project_data = body.get("project")
    if not isinstance(project_data, dict):
        raise HTTPException(
            status_code=400, detail="Missing 'project' dict in request body")
    from pydantic import ValidationError as PydanticValidationError
    from clippyme.domain.clip_project import validate_project
    from clippyme.domain.project_render import save_project_version
    try:
        project = validate_project(project_data)
    except PydanticValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid clip project: {exc.errors()}",
        )
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    # 3b: archive-and-bump lives in the shared helper (also used by
    # batch-apply) -- one implementation, no drift.
    version, project = save_project_version(
        job_dir=resolved.job_dir, clip_index=clip_index, project=project)
    return {"version": version, "project": project.model_dump()}


# ---------------------------------------------------------------------------
# Phase 3b: editor audio upload + batch-apply
# ---------------------------------------------------------------------------

AUDIO_UPLOAD_LIMIT = 20 * 1024 * 1024  # 20MB
AUDIO_DIRNAME = "audio"


def _sanitize_audio_filename(name: str) -> str:
    """Basename + safe chars + audio extension allowlist (400 otherwise)."""
    import re as _re
    base = os.path.basename((name or "").strip())
    safe = _re.sub(r"[^A-Za-z0-9._-]", "_", base)
    if not safe or safe in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    ext = os.path.splitext(safe)[1].lower()
    if ext not in (".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio type {ext!r} (mp3/wav/m4a/aac/ogg/flac)")
    return safe


@app.post("/api/audio/upload")
async def upload_audio(job_id: str, filename: str, request: Request,
                       user: AuthUser = Depends(get_current_user)):
    """Upload a music track for the clip editor (raw request body bytes).

    No multipart dependency: the client POSTs raw bytes with
    ``?job_id=...&filename=...``. Stored under ``<job_dir>/audio/`` and
    validated as real audio via ffprobe. Returns the project-relative path
    for ``audio[].file``.
    """
    require_trusted_config_request(request)
    enforce_rate_limit(request, "audio_upload", capacity=20, refill_per_sec=20 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    safe_name = _sanitize_audio_filename(filename)
    body = await request.body()
    if len(body) > AUDIO_UPLOAD_LIMIT:
        raise HTTPException(status_code=413, detail="File too large. Max 20MB")
    if len(body) < 1024:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    audio_dir = os.path.join(job_dir, AUDIO_DIRNAME)
    os.makedirs(audio_dir, exist_ok=True)
    dest = os.path.join(audio_dir, safe_name)
    tmp = dest + ".tmp"
    with open(tmp, "wb") as f:
        f.write(body)
    from clippyme.domain.project_render import _has_audio_stream
    if not await asyncio.to_thread(_has_audio_stream, tmp):
        os.remove(tmp)
        raise HTTPException(
            status_code=400, detail="Uploaded file has no audio stream")
    os.replace(tmp, dest)
    return {"file": f"{AUDIO_DIRNAME}/{safe_name}", "size": len(body)}


@app.post("/api/project/{job_id}/batch-apply")
async def batch_apply_projects(job_id: str, request: Request,
                               user: AuthUser = Depends(get_current_user)):
    """Apply one edit (patch) across several clips of a job.

    Body: {"clip_indices": [int, ...], "patch": {caption_style?,
    caption_position?, hook_text?, grade_preset?}}. Each clip's latest
    project is patched, re-validated, and saved as a new user version
    (the auto v1 record is never touched). Returns per-clip results.
    """
    require_trusted_config_request(request)
    enforce_rate_limit(request, "batch_apply", capacity=10, refill_per_sec=10 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    body = await request.json()
    clip_indices = body.get("clip_indices")
    patch = body.get("patch")
    if (not isinstance(clip_indices, list) or not clip_indices
            or any(not isinstance(x, int) or isinstance(x, bool) or x < 0
                   for x in clip_indices)):
        raise HTTPException(
            status_code=400,
            detail="clip_indices must be a non-empty list of clip indices")
    if len(clip_indices) > 50:
        raise HTTPException(status_code=400, detail="Too many clips (max 50)")
    if not isinstance(patch, dict) or not patch:
        raise HTTPException(
            status_code=400, detail="patch must be a non-empty dict")
    from clippyme.domain.clip_project import (
        apply_batch_patch, project_from_json, validate_project)
    from clippyme.domain.project_render import (
        get_project_path, save_project_version)
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    results: list[dict] = []
    for ci in clip_indices:
        try:
            path = await asyncio.to_thread(get_project_path, job_dir, ci)
            if not path:
                results.append({"clip_index": ci,
                                "error": "No project found for this clip"})
                continue
            with open(path, "r", encoding="utf-8") as f:
                project = project_from_json(f.read())
            try:
                apply_batch_patch(project, patch)
            except ValueError as exc:
                results.append({"clip_index": ci, "error": str(exc)[:200]})
                continue
            # Re-validate the mutated model: never silently corrupt.
            project = validate_project(project.model_dump())
            project.origin = "user"
            project.idempotency_key = str(uuid.uuid4())
            version, _ = save_project_version(
                job_dir=job_dir, clip_index=ci, project=project)
            results.append({"clip_index": ci, "version": version})
        except Exception as exc:  # per-clip isolation: one bad clip can't fail the batch
            results.append({"clip_index": ci, "error": str(exc)[:200]})
    return {"results": results}


def _find_backfill_source(job_dir: str, clip_filename: str | None):
    """Locate the source slice for backfill.

    Returns (path, is_final_fallback). The source slice is normally
    ``source_<clip_filename>``, but titles are sometimes translated between
    the slice cut and the final render (e.g. Italian slice, English final).
    The ``_clip_<N>`` number suffix is stable across translation, so match on
    that. As a last resort, fall back to the final clip file itself (already
    9:16 -- the emitter's center-crop math degrades to full-frame).
    """
    import glob as _glob
    import re as _re
    if clip_filename:
        cand = os.path.join(job_dir, f"source_{clip_filename}")
        if os.path.isfile(cand):
            return cand, False
        m = _re.search(r"_clip_(\d+)\.mp4$", clip_filename)
        if m:
            num = m.group(1)
            for p in sorted(_glob.glob(os.path.join(job_dir, "source_*.mp4"))):
                if _re.search(r"_clip_%s\.mp4$" % num, os.path.basename(p)):
                    return p, False
        final = os.path.join(job_dir, clip_filename)
        if os.path.isfile(final):
            return final, True
    return None, False


@app.post("/api/project/{job_id}/backfill")
async def backfill_clip_projects(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Generate clip-project.json for clips missing them. Never re-renders."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail="Job not found")
    from clippyme.domain.project_render import get_project_path
    from clippyme.pipeline.project_emit import emit_clip_project
    metadata: dict = {}
    import glob as _glob
    meta_candidates = sorted(_glob.glob(os.path.join(job_dir, "*_metadata.json")))
    meta_candidates += [os.path.join(job_dir, n)
                        for n in ("_metadata.json", "metadata.json")]
    for p in meta_candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                if isinstance(metadata, dict) and (metadata.get("clips") or metadata.get("shorts")):
                    break
                metadata = {}
            except (OSError, ValueError):
                pass
    clips = metadata.get("clips") or metadata.get("shorts") or []
    results: list[dict] = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        if get_project_path(job_dir, index):
            results.append({"clip_index": index, "status": "exists"})
            continue
        start, end = clip.get("start"), clip.get("end")
        clip_filename = clip.get("clip_filename")
        if not (isinstance(start, (int, float))
                and isinstance(end, (int, float))
                and end > start and clip_filename):
            results.append({"clip_index": index,
                            "status": "skipped_no_bounds"})
            continue
        clip_source, _src_is_final = _find_backfill_source(
            job_dir, clip_filename)
        if not clip_source:
            results.append({"clip_index": index,
                            "status": "skipped_no_source"})
            continue
        clip_final = os.path.join(job_dir, clip_filename)
        path = emit_clip_project(
            output_dir=job_dir,
            job_id=job_id,
            index=index,
            start=float(start),
            end=float(end),
            clip=clip,
            clips_data=metadata,
            clip_source=clip_source,
            clip_final=clip_final,
        )
        results.append({"clip_index": index,
                        "status": "created" if path else "failed"})
    return {"job_id": job_id, "results": results}


# ---------------------------------------------------------------------------
# Publish (Zernio) endpoints
# ---------------------------------------------------------------------------

@app.post("/api/publish/{job_id}/{clip_index}")
async def publish_clip_endpoint(job_id: str, clip_index: int, req: PublishRequest, request: Request, user: AuthUser = Depends(get_current_user)):
    """Upload a clip to Zernio and create a post on the requested platforms.

    If req.compose_first is True, the clip is freshly composed (Smart Cut →
    Hook → Subtitles) using req.toggles before upload — same flow as
    /api/compose. Otherwise we look for an existing composed_clip_{i}.mp4
    on disk and fall back to the base clip.
    """
    require_trusted_config_request(request)
    # Throttle uploads so a runaway "publish all" can't exhaust Zernio quota.
    enforce_rate_limit(request, "publish", capacity=30, refill_per_sec=30 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)

    # require_file=False: the base clip may be absent when a composed file
    # exists on disk — publish_clip_flow resolves the actual upload path.
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)

    zernio_cfg = await asyncio.to_thread(load_zernio_config)
    return await publish_clip_flow(
        job_id=job_id, clip_index=clip_index, resolved=resolved,
        req=req.model_dump(), zernio_cfg=zernio_cfg,
    )


# ---------------------------------------------------------------------------
# Clip popup actions (chat-UI): duplicate / schedule / upscale / export XML
# ---------------------------------------------------------------------------

@app.post("/api/clips/{job_id}/{clip_index}/duplicate")
async def duplicate_clip_endpoint(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Deep-copy a clip: new metadata entry, new rendered mp4, copied project.

    The source clip is never modified. Returns the new clip's index.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    result = await asyncio.to_thread(
        clip_tools.duplicate_clip, job_id, clip_index, OUTPUT_DIR)
    return {"ok": True, **result}


def _require_future_iso(value: str) -> str:
    """Validate scheduled_for is a future ISO 8601 timestamp (400 otherwise)."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400, detail="scheduled_for must be an ISO 8601 timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=400, detail="scheduled_for must be a future timestamp")
    return value


@app.post("/api/clips/{job_id}/{clip_index}/schedule")
async def schedule_clip_endpoint(job_id: str, clip_index: int, req: ScheduleClipRequest, request: Request, user: AuthUser = Depends(get_current_user)):
    """Schedule a clip for timed posting via Zernio (no local poller).

    Thin honest wrapper: builds a PublishRequest and calls the existing
    publish_clip_flow -- Zernio performs the actual timed post. Persists a
    clip-schedule-<index>.json record in the job dir.
    """
    require_trusted_config_request(request)
    # Same throttle bucket as immediate publishes.
    enforce_rate_limit(request, "publish", capacity=30, refill_per_sec=30 / 60)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    data = req.model_dump()
    scheduled_for = data.get("scheduled_for")
    if data["schedule_mode"] == "manual" and not scheduled_for:
        raise HTTPException(
            status_code=400,
            detail="schedule_mode='manual' requires scheduled_for")
    if scheduled_for:
        _require_future_iso(scheduled_for)

    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)

    # Reuse the full publish schema so platform/timezone validation is
    # identical to /api/publish.
    publish_req = PublishRequest(
        title=data["title"],
        caption=data["caption"],
        platforms=data["platforms"],
        schedule_mode=data["schedule_mode"],
        scheduled_for=scheduled_for,
        timezone=data["timezone"],
    )
    zernio_cfg = await asyncio.to_thread(load_zernio_config)
    result = await publish_clip_flow(
        job_id=job_id, clip_index=clip_index, resolved=resolved,
        req=publish_req.model_dump(), zernio_cfg=zernio_cfg,
    )

    record = {
        "job_id": job_id,
        "clip_index": clip_index,
        "schedule_mode": data["schedule_mode"],
        "scheduled_for": result.get("scheduled_for") or scheduled_for,
        "platforms": data["platforms"],
        "title": data["title"],
        "caption": data["caption"],
        "timezone": data["timezone"],
        "post_id": result.get("post_id"),
        "status": result.get("status") or "scheduled",
    }
    await asyncio.to_thread(
        clip_tools.write_schedule_record, resolved.job_dir, clip_index, record)

    response = {
        "ok": True,
        "scheduled_for": record["scheduled_for"],
        "status": "scheduled",
    }
    if record["post_id"]:
        response["post_id"] = record["post_id"]
    return response


@app.get("/api/clips/{job_id}/{clip_index}/schedule")
async def get_clip_schedule_endpoint(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Return the persisted schedule record, or {"scheduled": False}."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, OUTPUT_DIR, require_file=False)
    record = await asyncio.to_thread(
        clip_tools.read_schedule_record, resolved.job_dir, clip_index)
    return record if record else {"scheduled": False}


@app.post("/api/clips/{job_id}/{clip_index}/upscale")
async def upscale_clip_endpoint(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """ffmpeg 2x lanczos upscale (long edge capped at 2560px).

    Writes upscaled_<stem>.mp4 next to the original -- the source clip is
    never overwritten. NVENC when available, libx264 fallback.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    result = await asyncio.to_thread(
        clip_tools.upscale_clip, job_id, clip_index, OUTPUT_DIR)
    return {"ok": True, **result}


@app.get("/api/clips/{job_id}/{clip_index}/export-xml")
async def export_clip_xml_endpoint(job_id: str, clip_index: int, request: Request, user: AuthUser = Depends(get_current_user)):
    """Generate a minimal valid FCP7 (xmeml) timeline for the clip.

    Built from the clip-project JSON segments (falling back to the clip's
    start/end bounds). 400 when no timeline data exists -- never an empty
    timeline.
    """
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    _verify_job_ownership(job_id, user)
    path, filename = await asyncio.to_thread(
        clip_tools.export_timeline_xml, job_id, clip_index, OUTPUT_DIR)
    return FileResponse(path, media_type="application/xml", filename=filename)

# ---------------------------------------------------------------------------
# Published Performance Analytics & Feedback Loop endpoints
# ---------------------------------------------------------------------------

@app.get("/api/analytics/summary")
async def get_analytics_summary_endpoint():
    """Return aggregated stats, top-performing clips, and platform breakdowns."""
    from clippyme.domain.analytics_service import get_analytics_summary
    return await asyncio.to_thread(get_analytics_summary)


@app.post("/api/analytics/sync")
async def sync_analytics_endpoint(request: Request):
    """Sync live metrics from Zernio Analytics API."""
    require_trusted_config_request(request)
    from clippyme.domain.analytics_service import sync_zernio_analytics
    return await asyncio.to_thread(sync_zernio_analytics)


@app.post("/api/analytics/track")
async def track_analytics_endpoint(payload: dict, request: Request):
    """Record or update performance metrics for a specific clip."""
    require_trusted_config_request(request)
    clip_id = payload.get("clip_id")
    metrics = payload.get("metrics") or {}
    if not clip_id:
        raise HTTPException(status_code=400, detail="clip_id is required")
    from clippyme.domain.analytics_service import update_clip_metrics
    updated = await asyncio.to_thread(update_clip_metrics, clip_id, metrics)
    if not updated:
        raise HTTPException(status_code=404, detail="Clip not found in analytics registry")
    return {"success": True, "clip": updated}


@app.get("/api/analytics/insights")
async def get_analytics_insights_endpoint():
    """Return active learned performance rules and recommendations."""
    from clippyme.domain.performance_feedback import analyze_performance_patterns, get_learned_patterns_prompt
    patterns = await asyncio.to_thread(analyze_performance_patterns)
    prompt_snippet = await asyncio.to_thread(get_learned_patterns_prompt)
    return {
        "patterns": patterns,
        "prompt_snippet": prompt_snippet,
    }


# ---------------------------------------------------------------------------
# Content-monitor endpoints (multi-platform, multi-channel)
# ---------------------------------------------------------------------------

@app.post("/api/live-monitor/start")
async def live_monitor_start(req: LiveMonitorStartRequest, request: Request):
    """Start a monitor for one platform:channel. Returns that monitor's status
    (incl. its ``id``). Starting a duplicate (platform, channel) → 409."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    # start() raises ValidationError/ConflictError (ClippyMeError) → mapped to HTTP.
    return live_monitor.start(req.model_dump())


@app.post("/api/live-monitor/stop")
async def live_monitor_stop(request: Request):
    """Stop one monitor, or all monitors only when the body is truly absent."""
    require_trusted_config_request(request)
    raw = await request.body()
    req = None
    if raw:
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Malformed JSON body")
        try:
            req = LiveMonitorStopRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422, detail=exc.errors(include_context=False))
    return await live_monitor.stop(req.monitor_id if req else None)


@app.post("/api/live-monitor/{monitor_id}/config")
async def live_monitor_update_config(monitor_id: str, request: Request):
    """Patch selected settings on a running monitor (body: partial dict of
    updatable fields — see ``validate_monitor_partial_update``). Applies to
    FUTURE segments/publishes only, never retroactively."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    try:
        partial = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON body")
    return {"monitor": live_monitor.update_config(monitor_id, partial)}


@app.post("/api/live-monitor/{monitor_id}/publishing")
async def live_monitor_set_publishing(monitor_id: str, request: Request):
    """Pause/resume auto-publishing with a strict boolean request body."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON body")
    try:
        req = LiveMonitorPublishingRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=exc.errors(include_context=False))
    return {"monitor": live_monitor.set_publishing(monitor_id, req.enabled)}


@app.get("/api/live-monitor/status")
async def live_monitor_status(request: Request, monitor_id: Optional[str] = None):
    """One monitor's status (``?monitor_id=``) or ``{"monitors": [...]}`` for all."""
    require_trusted_config_request(request)
    return live_monitor.status(monitor_id)


@app.get("/api/live-monitor/{monitor_id}/pending-clips")
async def live_monitor_pending_clips(monitor_id: str, request: Request):
    """List pending preview clips awaiting review/approval for a monitor."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=20, refill_per_sec=20 / 60)
    return {"pending_clips": live_monitor.get_pending_clips(monitor_id)}


@app.post("/api/live-monitor/{monitor_id}/publish-clip/{clip_id}")
async def live_monitor_publish_pending_clip(monitor_id: str, clip_id: str, request: Request):
    """Approve and publish an individual previewed clip with optional overrides."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    overrides = None
    try:
        body = await request.body()
        if body:
            overrides = await request.json()
    except Exception:
        pass
    return await live_monitor.publish_pending_clip(monitor_id, clip_id, overrides)


@app.delete("/api/live-monitor/{monitor_id}/pending-clip/{clip_id}")
async def live_monitor_dismiss_pending_clip(monitor_id: str, clip_id: str, request: Request):
    """Dismiss/reject a pending preview clip, cleaning up on-disk files."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=20, refill_per_sec=20 / 60)
    return live_monitor.dismiss_pending_clip(monitor_id, clip_id)


@app.post("/api/live-monitor/{monitor_id}/publish-all")
async def live_monitor_publish_all(monitor_id: str, request: Request):
    """Approve and schedule publish for all pending clips of a monitor."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    return await live_monitor.publish_all_pending(monitor_id)


@app.post("/api/history/{job_id}/restore")
async def restore_job(job_id: str, request: Request):
    """Restore a past job into the in-memory jobs dict so edit/hook/subtitle endpoints work."""
    require_trusted_config_request(request)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    job_entry = restore_job_from_disk(job_id, OUTPUT_DIR, job_dir)
    jobs[job_id] = job_entry
    logger.info("Restored job %s into memory (%d clips)", job_id, len(job_entry["result"]["clips"]))
    return {"success": True, "status": "completed", "result": job_entry["result"]}
