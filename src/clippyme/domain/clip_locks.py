"""Per-clip and per-job mutexes shared by reframe, compose, and highlights paths.

Prevents concurrent modifications to deterministic clip and job metadata files.
"""
import asyncio
import contextlib
import os
import threading

# key -> [asyncio.Lock, refcount]
_LOCKS: dict = {}

# key -> [threading.RLock, refcount]
_THREAD_JOB_LOCKS: dict = {}
_THREAD_JOB_LOCKS_GUARD = threading.Lock()


@contextlib.asynccontextmanager
async def clip_lock(job_dir: str, clip_index: int):
    """Serialise reframe/compose work on one clip: ``async with clip_lock(...)``."""
    key = (os.path.abspath(job_dir), int(clip_index))
    entry = _LOCKS.get(key)
    if entry is None:
        entry = [asyncio.Lock(), 0]
        _LOCKS[key] = entry
    entry[1] += 1
    try:
        async with entry[0]:
            yield
    finally:
        entry[1] -= 1
        if entry[1] <= 0:
            _LOCKS.pop(key, None)


@contextlib.contextmanager
def job_metadata_lock(job_dir: str):
    """Process-wide per-job re-entrant sync mutex: ``with job_metadata_lock(job_dir): ...``.

    Serialises metadata read-modify-write and highlight operations for a job directory safely across threads.
    Uses RLock so nested calls within the same thread (e.g. batch orchestrator -> render reel) do not self-deadlock.
    """
    key = os.path.abspath(job_dir)
    with _THREAD_JOB_LOCKS_GUARD:
        entry = _THREAD_JOB_LOCKS.get(key)
        if entry is None:
            entry = [threading.RLock(), 0]
            _THREAD_JOB_LOCKS[key] = entry
        entry[1] += 1
        lock = entry[0]
    try:
        with lock:
            yield lock
    finally:
        with _THREAD_JOB_LOCKS_GUARD:
            entry[1] -= 1
            if entry[1] <= 0 and len(_THREAD_JOB_LOCKS) > 256:
                if _THREAD_JOB_LOCKS.get(key) is entry:
                    _THREAD_JOB_LOCKS.pop(key, None)
