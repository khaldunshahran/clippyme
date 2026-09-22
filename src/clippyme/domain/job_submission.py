"""Job-entry creation + enqueue, shared by /api/process, /api/batch, and /api/retry.

Owns queue-full rollback, initialization of runtime state, and resuming failed
jobs from durable checkpoints.
"""
import asyncio
import json
import logging
import os
import shutil

from clippyme.domain.errors import ClippyMeError, NotFoundError, ValidationError
from clippyme.domain.runtime_state import RuntimeState, runtime_result_fields

logger = logging.getLogger("clippyme")


class QueueFullError(ClippyMeError):
    """The job queue is at capacity (maps to 429)."""

    status_code = 429


def configured_max_attempts(env: dict[str, str] | None = None) -> int:
    """Return a bounded positive retry count despite malformed deployment env."""
    source = os.environ if env is None else env
    try:
        value = int(source.get("CLIPPYME_JOB_MAX_ATTEMPTS", "3") or 3)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(value, 10))


async def submit_job(
    *,
    jobs: dict,
    job_queue: asyncio.Queue,
    job_id: str,
    cmd: list,
    env: dict,
    job_output_dir: str,
    batch: bool = False,
    on_change=None,
    cleanup_paths=(),
    input_path: str | None = None,
    user_id: str = "default_user",
) -> None:
    """Register and enqueue a job, rolling every artefact back on queue-full."""
    max_attempts = configured_max_attempts()
    env["CLIPPYME_JOB_ID"] = job_id
    env["CLIPPYME_JOB_MAX_ATTEMPTS"] = str(max_attempts)
    env["CLIPPYME_USER_ID"] = user_id

    runtime = RuntimeState(job_output_dir, job_id=job_id)
    runtime.data["max_attempts"] = max_attempts
    runtime.data["attempt"] = 0
    runtime.data["detail"] = "waiting for a worker"
    runtime.data["user_id"] = user_id
    runtime.save()

    jobs[job_id] = {
        "status": "queued",
        "logs": [f"Job {job_id} queued (batch)." if batch else f"Job {job_id} queued."],
        "cmd": cmd,
        "env": env,
        "output_dir": job_output_dir,
        "input_path": input_path,
        "user_id": user_id,
        "result": {"clips": [], **runtime_result_fields(job_output_dir)},
        "attempt": 0,
        "max_attempts": max_attempts,
    }
    try:
        job_queue.put_nowait(job_id)
    except asyncio.QueueFull:
        jobs.pop(job_id, None)
        await asyncio.to_thread(shutil.rmtree, job_output_dir, True)
        for path in cleanup_paths or ():
            if path:
                try:
                    await asyncio.to_thread(os.remove, path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning(
                        "failed to remove rejected submission input %s",
                        path,
                        exc_info=True,
                    )
        raise QueueFullError("Server busy. Please try again later.")
    if on_change is not None:
        try:
            on_change()
        except Exception:
            logger.warning(
                "job-journal on_change hook failed for %s",
                job_id,
                exc_info=True,
            )


async def retry_job_action(
    *,
    jobs: dict,
    job_queue: asyncio.Queue,
    job_id: str,
    output_root: str,
    api_key: str | None = None,
    on_change=None,
) -> dict:
    """Re-enqueue a failed or interrupted job to resume from its durable checkpoints."""
    job_output_dir = os.path.join(output_root, job_id)
    if not os.path.isdir(job_output_dir):
        raise NotFoundError(f"Job directory not found: {job_id}")

    existing = jobs.get(job_id)
    if existing and existing.get("status") in ("processing", "queued"):
        return {
            "success": True,
            "job_id": job_id,
            "status": existing["status"],
            "detail": "Job is already active in queue",
        }

    cmd = list(existing.get("cmd")) if (existing and existing.get("cmd")) else None
    env = dict(existing.get("env")) if (existing and existing.get("env")) else None
    input_path = existing.get("input_path") if existing else None

    # If memory state is absent, reconstruct orchestrator invocation from disk checkpoints
    if not cmd:
        runtime_path = os.path.join(job_output_dir, ".clippyme_runtime.json")
        meta_path = os.path.join(job_output_dir, f"{job_id}_metadata.json")
        if not os.path.isfile(meta_path):
            meta_path = os.path.join(job_output_dir, "metadata.json")

        rt_data = {}
        if os.path.isfile(runtime_path):
            try:
                with open(runtime_path, "r", encoding="utf-8") as f:
                    rt_data = json.load(f)
            except Exception:
                pass

        art = rt_data.get("artifacts") or {}
        input_video = art.get("input_video")
        source_url = art.get("source_url") or rt_data.get("source_url")

        if not input_video or not os.path.isfile(input_video):
            # Probe directory for source video candidate
            allowed_exts = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")
            for fname in os.listdir(job_output_dir):
                if any(fname.lower().endswith(ext) for ext in allowed_exts):
                    if not fname.startswith(("composed_", "dubbed_", "highlight_reel_", "temp_")):
                        input_video = os.path.join(job_output_dir, fname)
                        break

        is_highlights = False
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    if "highlights" in m:
                        is_highlights = True
            except Exception:
                pass

        cmd = ["python", "-m", "clippyme.pipeline.orchestrator"]
        if input_video and os.path.isfile(input_video):
            cmd.extend(["-i", input_video])
            input_path = input_video
        elif source_url:
            cmd.extend(["-u", source_url])
        else:
            raise ValidationError(f"Cannot retry job {job_id}: no source video or URL found on disk")

        cmd.extend(["-o", job_output_dir])
        if is_highlights:
            cmd.append("--highlights")

    if not env:
        env = os.environ.copy()
    if api_key:
        env["GEMINI_API_KEY"] = api_key

    env["CLIPPYME_JOB_ID"] = job_id
    max_attempts = configured_max_attempts(env)
    env["CLIPPYME_JOB_MAX_ATTEMPTS"] = str(max_attempts)

    runtime = RuntimeState(job_output_dir, job_id=job_id)
    runtime.data["last_error"] = None
    runtime.data["detail"] = "resuming from checkpoint"
    runtime.save()

    logs = list(existing.get("logs", [])) if existing else []
    logs.append("♻️ Retry requested by user. Resuming pipeline from latest checkpoint...")

    jobs[job_id] = {
        "status": "queued",
        "logs": logs,
        "cmd": cmd,
        "env": env,
        "output_dir": job_output_dir,
        "input_path": input_path,
        "result": {"clips": [], **runtime_result_fields(job_output_dir)},
        "attempt": 0,
        "max_attempts": max_attempts,
    }

    try:
        job_queue.put_nowait(job_id)
    except asyncio.QueueFull:
        jobs[job_id]["status"] = "failed"
        raise QueueFullError("Server busy. Please try again later.")

    if on_change is not None:
        try:
            on_change()
        except Exception:
            logger.warning("job-journal on_change hook failed for retry %s", job_id, exc_info=True)

    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "detail": "Job re-enqueued to resume from checkpoints",
    }
