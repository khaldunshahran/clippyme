"""Background publish task registry.

The publish endpoint returns 202 immediately and runs the heavy
compose+upload work in a background asyncio task, because the dashboard sits
behind Cloudflare (100 s origin timeout -> HTTP 524 on long requests).

Tasks live in memory only; entries older than TASK_TTL_SEC are pruned lazily.
A container restart drops in-flight tasks — the clips' publish records on
disk remain the source of truth, and the 409 already-published guard makes
retries safe.
"""

import asyncio
import logging
import time
import uuid

logger = logging.getLogger("clippyme")

TASK_TTL_SEC = 3600
_MAX_TASKS = 500

_TASKS: dict[str, dict] = {}


def _prune() -> None:
    now = time.time()
    stale = [tid for tid, t in _TASKS.items() if now - t.get("created_at", now) > TASK_TTL_SEC]
    for tid in stale:
        _TASKS.pop(tid, None)
    if len(_TASKS) > _MAX_TASKS:
        # Drop the oldest finished tasks first.
        ordered = sorted(_TASKS.items(), key=lambda kv: kv[1].get("created_at", 0))
        for tid, t in ordered:
            if t.get("state") in ("done", "error"):
                _TASKS.pop(tid, None)
            if len(_TASKS) <= _MAX_TASKS:
                break


def is_valid_task_id(task_id: str) -> bool:
    return isinstance(task_id, str) and len(task_id) == 32 and all(
        c in "0123456789abcdef" for c in task_id
    )


def submit_publish_task(coro_factory, *, job_id: str, clip_index: int) -> str:
    """Schedule publish_clip_flow() in the background; return task_id.

    ``coro_factory`` is a zero-arg callable returning the coroutine to run.
    """
    _prune()
    task_id = uuid.uuid4().hex
    _TASKS[task_id] = {
        "task_id": task_id,
        "job_id": job_id,
        "clip_index": clip_index,
        "state": "queued",  # queued -> running -> done | error
        "result": None,
        "error": None,
        "error_status": None,
        "created_at": time.time(),
        "updated_at": time.time(),
    }

    async def _runner() -> None:
        entry = _TASKS.get(task_id)
        if entry is None:
            return
        entry["state"] = "running"
        entry["updated_at"] = time.time()
        try:
            result = await coro_factory()
            entry["state"] = "done"
            entry["result"] = result if isinstance(result, dict) else {"ok": True}
        except Exception as e:  # never let the task die silently
            status = getattr(e, "status_code", None)
            entry["state"] = "error"
            entry["error"] = str(e) or type(e).__name__
            entry["error_status"] = status
            logger.error(
                "publish task %s failed for %s/%d: %s", task_id, job_id, clip_index, e
            )
        entry["updated_at"] = time.time()

    asyncio.create_task(_runner())
    logger.info("publish task %s queued for %s/%d", task_id, job_id, clip_index)
    return task_id


def get_publish_task(task_id: str) -> dict | None:
    _prune()
    return _TASKS.get(task_id)
