import os
import sys
import time
import uuid
import json
import shutil
import glob
import asyncio
import logging
import threading
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
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from clippyme.api.auth import AuthUser, get_current_user

from clippyme.domain.job_results import build_main_cmd, canonical_reframe_mode
from clippyme.domain.compose import compose_layers
from clippyme.domain.reframe_service import run_reframe
from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.uploads import stream_upload_within_limit, FileTooLarge
from clippyme.domain.clip_endpoints import run_smart_cut, restore_job_from_disk
from clippyme.domain.clip_resolve import resolve_clip
from clippyme.domain import job_control
from clippyme.domain.job_actions import cancel_job_action, stop_job_action
from clippyme.domain.job_journal import JOURNAL_FILENAME, make_journal_writer, recover_jobs
from clippyme.domain.job_runner import make_run_job
from clippyme.domain.job_submission import QueueFullError, submit_job
from clippyme.domain.publish_service import publish_clip_flow
from clippyme.domain.publish_tasks import (
    get_publish_task,
    is_valid_task_id,
    submit_publish_task,
)
from clippyme.domain.job_artifacts import is_clip_verified_published
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
    _validate_drop_ranges,
)
from clippyme.api.security import (
    ALLOWED_ORIGINS,
    enforce_api_token,
    enforce_rate_limit,
    require_trusted_config_request,
    verify_supabase_jwt,
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
# Idempotency keys for POST /api/process: (user_id, key) -> job_id.
# Guards against duplicate jobs when the client retries a submission whose
# response was lost (tunnel hang → "failed to fetch" → user clicks again).
# Bounded in memory to the most recent entries so it can't grow without
# limit; DURABLE via an append-only JSONL journal in DATA_DIR so a backend
# restart doesn't lose the mapping and resurrect duplicates. The journal is
# compacted whenever it grows past _IDEMPOTENCY_JOURNAL_COMPACT_LINES lines.
_idempotency_keys: Dict[tuple, str] = {}
_IDEMPOTENCY_KEY_CAP = 2000
_IDEMPOTENCY_JOURNAL_FILENAME = "idempotency_journal.jsonl"
_IDEMPOTENCY_JOURNAL_COMPACT_LINES = 4000
_idempotency_journal_lock = threading.Lock()


def _idempotency_journal_path() -> str:
    return os.path.join(DATA_DIR, _IDEMPOTENCY_JOURNAL_FILENAME)


def _replay_idempotency_journal() -> None:
    """Load (user_id, key) -> job_id mappings from the durable journal.

    Runs at startup (before the dispatcher) so retried submissions still
    dedupe against jobs created in a previous server life. Corrupt lines are
    skipped; a missing journal is not an error.
    """
    path = _idempotency_journal_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return
    except OSError:
        logger.exception("Could not read idempotency journal — starting empty")
        return
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            user_id, key, job_id = rec["user_id"], rec["key"], rec["job_id"]
        except (ValueError, KeyError, TypeError):
            continue
        if user_id and key and job_id:
            _idempotency_keys[(str(user_id), str(key))] = str(job_id)
            while len(_idempotency_keys) > _IDEMPOTENCY_KEY_CAP:
                _idempotency_keys.pop(next(iter(_idempotency_keys)))


def _compact_idempotency_journal_locked() -> None:
    """Rewrite the journal with the current in-memory mappings (lock held)."""
    path = _idempotency_journal_path()
    tmp = path + ".tmp"
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for (user_id, key), job_id in _idempotency_keys.items():
                f.write(json.dumps({"user_id": user_id, "key": key,
                                    "job_id": job_id, "ts": time.time()}) + "\n")
        os.replace(tmp, path)
    except OSError:
        logger.exception("Idempotency journal compaction failed")
        try:
            os.remove(tmp)
        except OSError:
            pass


def _remember_idempotency_key(user_id: str, key: str, job_id: str) -> None:
    _idempotency_keys[(user_id, key)] = job_id
    while len(_idempotency_keys) > _IDEMPOTENCY_KEY_CAP:
        _idempotency_keys.pop(next(iter(_idempotency_keys)))
    # Durable append so the mapping survives a restart.
    with _idempotency_journal_lock:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(_idempotency_journal_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps({"user_id": user_id, "key": key,
                                    "job_id": job_id, "ts": time.time()}) + "\n")
        except OSError:
            logger.exception("Idempotency journal append failed — key kept in memory only")
            return
        # Compact when the journal grows well past the memory cap (every
        # append writes one line, so this check is cheap: file size >> lines).
        try:
            if os.path.getsize(_idempotency_journal_path()) > _IDEMPOTENCY_JOURNAL_COMPACT_LINES * 160:
                _compact_idempotency_journal_locked()
        except OSError:
            pass


def _lookup_idempotency_key(user_id: str, key: str) -> Optional[str]:
    job_id = _idempotency_keys.get((user_id, key))
    if job_id is None:
        # Memory miss (evicted by the cap or lost before journaling existed):
        # scan the durable journal for the latest record for this pair. This
        # is the rare path — steady-state lookups hit the in-memory dict.
        job_id = _scan_idempotency_journal(user_id, key)
        if job_id is not None:
            _idempotency_keys[(user_id, key)] = job_id
    if job_id and job_id in jobs:
        return job_id
    if job_id:
        # Job is gone (evicted / backend restarted): key is stale, forget it.
        _idempotency_keys.pop((user_id, key), None)
    return None


def _scan_idempotency_journal(user_id: str, key: str) -> Optional[str]:
    """Return the latest job_id journalled for (user_id, key), or None."""
    path = _idempotency_journal_path()
    latest = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("user_id") == user_id and rec.get("key") == key and rec.get("job_id"):
                    latest = str(rec["job_id"])
    except (FileNotFoundError, OSError):
        return None
    return latest
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
    # S4: fail closed — SaaS mode must never serve with auth misconfigured.
    _saas_mode = os.environ.get("AUTH_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on")
    if _saas_mode and not os.environ.get("SUPABASE_JWT_SECRET", "").strip():
        logger.error("REFUSING TO SERVE: AUTH_ENABLED=1 but SUPABASE_JWT_SECRET is not configured.")
        raise RuntimeError("Refusing to serve: AUTH_ENABLED=1 requires SUPABASE_JWT_SECRET to be set.")

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

    # Replay the durable idempotency journal so submissions retried after a
    # restart still dedupe against jobs from the previous server life.
    try:
        _replay_idempotency_journal()
    except Exception:
        logger.exception("Idempotency journal replay failed — starting with an empty key map")

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
async def _supabase_jwt_gate(request: Request, call_next):
    """Verify the Supabase session JWT on /api requests (public SaaS path).

    The public frontend sends ``Authorization: Bearer <supabase_jwt>`` (see
    dashboard/src/lib/supabaseClient.js). When the token verifies against the
    project JWT secret, the request is stamped with the user id and
    require_trusted_config_request + enforce_rate_limit honour it. An invalid
    or absent token is NOT an error here — the request simply continues down
    the normal local-trust path (fail safe, local-first unchanged).
    """
    if request.url.path.startswith("/api/") and request.method != "OPTIONS":
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            claims = verify_supabase_jwt(auth[7:].strip())
            if claims:
                request.state.supabase_user = claims["sub"]
                # Full claims (incl. email) so get_current_user can build a
                # per-user identity without re-verifying the token.
                request.state.supabase_claims = claims
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
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
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


def _resolve_gemini_key(request: Request, persisted: Optional[dict]) -> Optional[str]:
    """Resolve the effective Gemini API key for a request.

    Priority: explicit ``X-Gemini-Key`` header, then the persisted server key,
    then the ``GEMINI_API_KEY`` env var. A whitespace-only (or empty) header
    is treated as missing so it can never shadow the server key with a blank
    value — previously ``" "`` was truthy and broke the fallback chain.
    """
    header_key = (request.headers.get("X-Gemini-Key") or "").strip()
    persisted_key = ((persisted or {}).get("GEMINI_API_KEY") or "").strip()
    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    return header_key or persisted_key or env_key or None


# Cap for JSON request bodies parsed manually via ``await request.json()``
# (multipart uploads already stream through FastAPI's File handling).
# Prevents a huge JSON payload from being buffered fully into memory.
MAX_JSON_BODY_BYTES = 1_000_000  # 1 MB — submit payloads are a few KB


async def _read_capped_json(request: Request) -> dict:
    """Read a JSON body with a size cap; 413 if it exceeds the cap."""
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > MAX_JSON_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    raw = await request.body()
    if len(raw) > MAX_JSON_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    try:
        return json.loads(raw.decode("utf-8")) if raw else {}
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")


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
    api_key = _resolve_gemini_key(request, persisted)
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
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await _read_capped_json(request)  # 1 MB cap, 413 over
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
            })
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors())

    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    from clippyme.domain.quota_service import check_user_quota
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    # Idempotency: a retried submission (same Idempotency-Key header) returns
    # the already-created job instead of queueing a duplicate. The frontend
    # sends a fresh UUID per "Create clips" click.
    idempotency_key = request.headers.get("Idempotency-Key", "").strip()[:128]
    if idempotency_key:
        existing_job_id = _lookup_idempotency_key(user.id, idempotency_key)
        if existing_job_id:
            existing = jobs[existing_job_id]
            return {
                "job_id": existing_job_id,
                "status": existing.get("status", "queued"),
                "duplicate": True,
            }

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
    if idempotency_key:
        _remember_idempotency_key(user.id, idempotency_key, job_id)
        jobs[job_id]["idempotency_key"] = idempotency_key

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
    api_key = _resolve_gemini_key(request, persisted)
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
async def get_active_jobs(request: Request, user: AuthUser = Depends(get_current_user)):
    require_trusted_config_request(request)
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
async def get_status(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    require_trusted_config_request(request)
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
                # Only claim completion with evidence: a clip metadata file
                # means the pipeline finished cutting. A nonempty directory
                # alone (e.g. an interrupted download) must not report
                # "complete" — that would strand the frontend poller on a
                # terminal state for a job that never finished.
                has_metadata = any(
                    name.endswith("_metadata.json")
                    for name in os.listdir(job_dir)
                )
                if has_metadata and user.is_admin:
                    return {
                        "status": "complete",
                        "logs": ["Job loaded from storage"],
                        "result": {},
                    }
            except HTTPException:
                raise
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
    # A paused job must not hold its concurrency slot forever: release it so
    # queued jobs can run. Resume re-acquires. slot_held prevents the wrapper
    # from double-releasing when the job later finishes/is cancelled.
    if job.pop('slot_held', False):
        concurrency_semaphore.release()
        job['logs'].append("Concurrency slot released while paused.")
        logger.info("Job %s paused; concurrency slot released", job_id)
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

    # Re-acquire a concurrency slot before waking the tree: another job may
    # have taken the slot this job released when it was paused. This waits
    # for a free slot instead of oversubscribing the GPU.
    acquired_here = False
    if not job.get('slot_held'):
        await concurrency_semaphore.acquire()
        job['slot_held'] = True
        acquired_here = True
    try:
        n = await asyncio.to_thread(job_control.resume_tree, proc.pid)
    except Exception:
        # Don't leak the slot we just took if the resume itself failed.
        if acquired_here:
            job.pop('slot_held', False)
            concurrency_semaphore.release()
        raise
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
    api_key = _resolve_gemini_key(request, persisted)

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
    api_key = _resolve_gemini_key(request, persisted)

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
        if len(raw_body) > MAX_JSON_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
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
    """Conversational clip trim: a plain-English instruction → Gemini → the
    clip-relative spans to remove. The returned `drop_ranges` feed the SAME
    manual-trim machinery as the tap-to-cut UI (compose / publish honour them)."""
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
async def get_storage_stats(request: Request, user: AuthUser = Depends(get_current_user)):
    """Return categorized disk usage breakdown across output/ and uploads/."""
    require_trusted_config_request(request)
    require_admin(request, user)
    from clippyme.domain.job_artifacts import get_storage_breakdown
    return await asyncio.to_thread(get_storage_breakdown, OUTPUT_DIR, UPLOAD_DIR)

@app.post("/api/storage/cleanup")
async def trigger_storage_cleanup(request: Request, user: AuthUser = Depends(get_current_user)):
    """Run manual storage cleanup pass: purges partial downloads, ASR audio, uploads, and optionally source videos."""
    require_trusted_config_request(request)
    require_admin(request, user)
    body = {}
    if request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await _read_capped_json(request)
        except HTTPException as exc:
            if exc.status_code == 413:
                raise
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

    # Clean-base composition: for_compose excludes already-composed files
    # from the on-disk fallback chain (composing onto them would double-burn
    # captions) and refuses outright when only a composed file exists.
    resolved = await asyncio.to_thread(resolve_clip, job_id, clip_index, OUTPUT_DIR, for_compose=True)

    try:
        composed_filename = await compose_layers(
            base_clip=resolved.clip_path,
            job_dir=resolved.job_dir,
            clip_index=clip_index,
            metadata=resolved.metadata,
            clip_info=resolved.clip_info,
            toggles=req.toggles,
            hook_params=req.hook_params,
            subtitle_params=req.subtitle_params,
            logo_params=req.logo_params,
            grade_params=req.grade_params,
            banner_params=req.banner_params,
            drop_ranges=req.drop_ranges,
        )
        return {"composed_url": f"/videos/{job_id}/{composed_filename}"}
    except (HTTPException, ClippyMeError):
        raise
    except Exception as e:
        logger.error("Compose error for job %s clip %d: %s", job_id, clip_index, e)
        raise HTTPException(status_code=500, detail="Compose pipeline failed")


# ---------------------------------------------------------------------------
# Publish (Zernio) endpoints
# ---------------------------------------------------------------------------

@app.post("/api/publish/{job_id}/{clip_index}", status_code=202)
async def publish_clip_endpoint(job_id: str, clip_index: int, req: PublishRequest, request: Request, user: AuthUser = Depends(get_current_user)):
    """Queue a clip publish to Zernio (async).

    Returns 202 immediately with a ``task_id`` — the compose + upload run in
    a background task because the dashboard sits behind Cloudflare (100 s
    origin timeout -> HTTP 524 on long synchronous requests). Poll
    ``GET /api/publish/status/{task_id}`` for the outcome.

    If req.compose_first is True, the clip is composed (Smart Cut → Hook →
    Subtitles) using req.toggles before upload — same flow as /api/compose —
    unless a fresh composed file already exists on disk, in which case it is
    reused. Otherwise we look for an existing composed_clip_{i}.mp4 on disk
    and fall back to the base clip.
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
    if not (zernio_cfg or {}).get("api_key"):
        raise HTTPException(status_code=400, detail="Zernio API key not configured")
    # Fail fast on duplicates so we never queue a doomed task.
    if is_clip_verified_published(resolved.clip_info):
        raise HTTPException(status_code=409, detail="Clip already published")

    req_dict = req.model_dump()
    task_id = submit_publish_task(
        lambda: publish_clip_flow(
            job_id=job_id, clip_index=clip_index, resolved=resolved,
            req=req_dict, zernio_cfg=zernio_cfg,
        ),
        job_id=job_id,
        clip_index=clip_index,
    )
    return {"status": "accepted", "task_id": task_id, "job_id": job_id, "clip_index": clip_index}


@app.get("/api/publish/status/{task_id}")
async def publish_status_endpoint(task_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Return the state of a background publish task.

    ``state`` is one of: queued | running | done | error.
    On ``done``, ``result`` carries the publish result; on ``error``,
    ``error`` carries the message and ``error_status`` the HTTP status the
    synchronous endpoint would have returned.
    """
    require_trusted_config_request(request)
    if not is_valid_task_id(task_id):
        raise HTTPException(status_code=400, detail="Invalid task ID")
    task = get_publish_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Publish task not found or expired")
    _verify_job_ownership(task["job_id"], user)
    return {
        "task_id": task["task_id"],
        "job_id": task["job_id"],
        "clip_index": task["clip_index"],
        "state": task["state"],
        "result": task["result"],
        "error": task["error"],
        "error_status": task["error_status"],
        "updated_at": task["updated_at"],
    }


# ---------------------------------------------------------------------------
# Published Performance Analytics & Feedback Loop endpoints
# ---------------------------------------------------------------------------

@app.get("/api/analytics/summary")
async def get_analytics_summary_endpoint(request: Request):
    """Return aggregated stats, top-performing clips, and platform breakdowns."""
    require_trusted_config_request(request)
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
async def get_analytics_insights_endpoint(request: Request):
    """Return active learned performance rules and recommendations."""
    require_trusted_config_request(request)
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
        if len(raw) > MAX_JSON_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Request body too large")
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
    # Capped reader: 400 on malformed JSON, 413 when the body is too large.
    partial = await _read_capped_json(request)
    return {"monitor": live_monitor.update_config(monitor_id, partial)}


@app.post("/api/live-monitor/{monitor_id}/publishing")
async def live_monitor_set_publishing(monitor_id: str, request: Request):
    """Pause/resume auto-publishing with a strict boolean request body."""
    require_trusted_config_request(request)
    enforce_rate_limit(request, "livemonitor", capacity=10, refill_per_sec=10 / 60)
    # Capped reader: 400 on malformed JSON, 413 when the body is too large.
    body = await _read_capped_json(request)
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
            if len(body) > MAX_JSON_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
            overrides = await request.json()
    except HTTPException:
        raise
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
async def restore_job(job_id: str, request: Request, user: AuthUser = Depends(get_current_user)):
    """Restore a past job into the in-memory jobs dict so edit/hook/subtitle endpoints work."""
    require_trusted_config_request(request)
    _verify_job_ownership(job_id, user)
    if not is_valid_job_id(job_id):
        raise HTTPException(status_code=400, detail="Invalid job ID")
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    job_entry = restore_job_from_disk(job_id, OUTPUT_DIR, job_dir)
    jobs[job_id] = job_entry
    logger.info("Restored job %s into memory (%d clips)", job_id, len(job_entry["result"]["clips"]))
    return {"success": True, "status": "completed", "result": job_entry["result"]}
