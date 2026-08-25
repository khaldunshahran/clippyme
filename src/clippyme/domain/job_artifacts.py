"""Filesystem helpers for locating and relocating job artifacts written by main.py."""
import glob
import json
import logging
import os
import shutil
from typing import Tuple

logger = logging.getLogger("clippyme")


def find_job_metadata_path(job_id: str, output_dir: str) -> str:
    """Return the path to a job's ``*_metadata.json`` file.

    Raises ``FileNotFoundError`` if no metadata file exists.
    """
    job_dir = os.path.join(output_dir, job_id)
    matches = glob.glob(os.path.join(job_dir, "*_metadata.json"))
    if not matches:
        raise FileNotFoundError(f"Metadata not found for job {job_id}")
    # Newest-by-mtime, consistent with job_results._pick_latest_metadata. A bare
    # glob[0] is filesystem-order dependent, so when a job dir holds >1 metadata
    # file (e.g. a reprocess) smartcut/reframe could operate on a different file
    # than the user sees.
    return max(matches, key=os.path.getmtime)


def load_job_metadata(job_id: str, output_dir: str) -> Tuple[str, dict]:
    """Load a job's metadata JSON.

    Returns ``(metadata_path, data)``. Raises ``FileNotFoundError`` if the
    metadata file does not exist.
    """
    metadata_path = find_job_metadata_path(job_id, output_dir)
    with open(metadata_path, "r", encoding="utf-8") as f:
        return metadata_path, json.load(f)


import threading
import time

_METADATA_LOCK = threading.Lock()


def save_job_metadata(metadata_path: str, data: dict) -> None:
    """Persist a job's metadata JSON back to disk atomically.

    Writes to a unique ``<metadata_path>.<pid>_<thread>_<ts>.tmp`` sibling and
    ``os.replace()``s it into place so a crash / SIGKILL mid-write cannot leave
    the caller with a half-written or truncated JSON file. Thread-safe with
    Windows retry on file locking.
    """
    tmp_path = f"{metadata_path}.{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.tmp"
    with _METADATA_LOCK:
        try:
            fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            # Retry loop for Windows file-locking race conditions.
            # Batch "Publish all" can fire 15+ concurrent threads that all
            # hit record_clip_publish on the SAME metadata.json; the old
            # 5-retry / 50 ms base wasn't enough under heavy contention.
            for attempt in range(8):
                try:
                    os.replace(tmp_path, metadata_path)
                    break
                except (PermissionError, OSError) as exc:
                    if attempt == 7:
                        raise
                    time.sleep(0.1 * (2 ** attempt))
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise


def record_clip_publish(job_id: str, clip_index: int, output_dir: str, record: dict) -> None:
    """Append a publish record onto a clip's metadata entry (atomic).

    Best-effort by design: callers should treat a failure here as non-fatal
    (the publish itself already succeeded) and just log it.
    """
    metadata_path, data = load_job_metadata(job_id, output_dir)
    shorts = data.get("shorts", [])
    if 0 <= clip_index < len(shorts):
        shorts[clip_index].setdefault("published", []).append(record)
        save_job_metadata(metadata_path, data)


def mark_clip_deleted(job_id: str, clip_index: int, output_dir: str) -> None:
    """Mark a clip as deleted_after_publish in its metadata file.
    
    This avoids dropping the array element entirely (which would break indices).
    """
    metadata_path, data = load_job_metadata(job_id, output_dir)
    shorts = data.get("shorts", [])
    if 0 <= clip_index < len(shorts):
        shorts[clip_index]["deleted_after_publish"] = True
        save_job_metadata(metadata_path, data)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.debug("Failed to remove %s: %s", path, exc)


def is_clip_verified_published(clip: dict) -> bool:
    """Returns True ONLY if a publish record exists with a confirmed post_id from Zernio/social platforms."""
    if not isinstance(clip, dict):
        return False
    published_list = clip.get("published") or []
    if isinstance(published_list, list):
        for rec in published_list:
            if isinstance(rec, dict) and bool(rec.get("post_id")):
                return True
    if isinstance(published_list, dict) and bool(published_list.get("post_id")):
        return True
    return False


def delete_clip_artifacts(job_id: str, clip: dict, clip_path: str, upload_path: str, output_dir: str) -> None:
    """Best-effort removal of a published clip's on-disk video artifacts + a metadata mark.

    Never raises — the publish already succeeded.
    Keeps metadata.json, .clippyme_runtime.json, and cover thumbnails permanently.
    """
    try:
        job_dir = os.path.join(output_dir, job_id)
        clip_filename = os.path.basename(clip_path) if clip_path else ""
        idx = clip.get("original_index")
        targets = []
        if clip_path:
            targets.append(clip_path)
        if upload_path and upload_path != clip_path:
            targets.append(upload_path)
        if clip_filename:
            targets.append(os.path.join(job_dir, f"source_{clip_filename}"))
        if idx is not None:
            from clippyme.domain.clip_resolve import composed_clip_basename
            targets.append(os.path.join(job_dir, composed_clip_basename(clip, idx)))

        for path in targets:
            _safe_remove(path)

        if idx is not None:
            mark_clip_deleted(job_id, idx, output_dir)
    except Exception:
        logger.warning("Artifact cleanup failed for %s", job_id, exc_info=True)


def relocate_root_job_artifacts(job_id: str, job_output_dir: str, output_dir: str) -> bool:
    """Backward-compat rescue.

    If ``main.py`` accidentally wrote metadata/clips into ``output_dir`` root
    (e.g. ``output/<jobid>_...``), move them into ``output/<job_id>/`` so the
    API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        pattern = os.path.join(output_dir, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not meta_candidates:
            return False

        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        clip_pattern = os.path.join(output_dir, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        temp_clip_pattern = os.path.join(output_dir, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception as exc:
        logger.warning("relocate_root_job_artifacts failed for %s: %s", job_id, exc)
        return False


def purge_partial_downloads(output_dir: str) -> dict:
    """Scan output directory and subdirectories for partial/temporary download files.

    Removes *.part, *.ytdl, *.tmp, *.temp files and returns freed_bytes and removed_count.
    Never raises exceptions.
    """
    freed_bytes = 0
    removed_count = 0
    if not os.path.exists(output_dir):
        return {"freed_bytes": 0, "removed_count": 0}

    partial_extensions = {".part", ".ytdl", ".tmp", ".temp"}
    try:
        for root, _, files in os.walk(output_dir):
            for file in files:
                lower = file.lower()
                is_partial = any(lower.endswith(ext) for ext in partial_extensions) or (".f" in lower and lower.endswith(".mp4.part"))
                if is_partial:
                    file_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_bytes += size
                        removed_count += 1
                        logger.info("Purged partial download file: %s (%d bytes)", file_path, size)
                    except OSError as exc:
                        logger.debug("Failed to purge partial file %s: %s", file_path, exc)
    except Exception as exc:
        logger.warning("purge_partial_downloads encountered an error: %s", exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count}


def cleanup_failed_job_artifacts(job_id: str, output_dir: str) -> dict:
    """Purge partial downloads and transient fragments for a failed job."""
    freed_bytes = 0
    removed_count = 0
    job_dir = os.path.join(output_dir, job_id)
    if not os.path.exists(job_dir):
        return {"freed_bytes": 0, "removed_count": 0}

    try:
        p_res = purge_partial_downloads(job_dir)
        freed_bytes += p_res["freed_bytes"]
        removed_count += p_res["removed_count"]
    except Exception as exc:
        logger.warning("cleanup_failed_job_artifacts failed for %s: %s", job_id, exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count}


def cleanup_published_job(job_id: str, output_dir: str, clip_index: int | None = None) -> dict:
    """Clean up video files for published clips and source video ONLY when verified as published.

    Never touches unpublished clips. Never removes metadata or thumbnail cover images.
    """
    freed_bytes = 0
    removed_count = 0
    job_dir = os.path.join(output_dir, job_id)
    if not os.path.exists(job_dir):
        return {"freed_bytes": 0, "removed_count": 0}

    try:
        try:
            metadata_path, data = load_job_metadata(job_id, output_dir)
        except Exception:
            return {"freed_bytes": 0, "removed_count": 0}

        shorts = data.get("shorts") or data.get("clips") or []
        if not isinstance(shorts, list) or not shorts:
            return {"freed_bytes": 0, "removed_count": 0}

        indices_to_check = [clip_index] if clip_index is not None else list(range(len(shorts)))

        for idx in indices_to_check:
            if not (0 <= idx < len(shorts)):
                continue
            clip = shorts[idx]
            if is_clip_verified_published(clip):
                clip_fn = clip.get("clip_filename")
                targets = []
                if clip_fn:
                    targets.append(os.path.join(job_dir, clip_fn))
                    targets.append(os.path.join(job_dir, f"source_{clip_fn}"))
                from clippyme.domain.clip_resolve import composed_clip_basename
                targets.append(os.path.join(job_dir, composed_clip_basename(clip, idx)))

                for target_path in targets:
                    if os.path.isfile(target_path):
                        try:
                            size = os.path.getsize(target_path)
                            os.remove(target_path)
                            freed_bytes += size
                            removed_count += 1
                            logger.info("Cleaned verified published clip video: %s (%d bytes)", target_path, size)
                        except OSError as exc:
                            logger.debug("Failed to remove published clip %s: %s", target_path, exc)
                mark_clip_deleted(job_id, idx, output_dir)

        ready_clips = [c for c in shorts if c.get("qa", {}).get("ok", True)]
        all_ready_published = bool(ready_clips) and all(is_clip_verified_published(c) for c in ready_clips)

        if all_ready_published:
            for file in os.listdir(job_dir):
                file_path = os.path.join(job_dir, file)
                lower = file.lower()
                is_video = lower.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov"))
                is_essential = lower.endswith(("_metadata.json", ".json", ".jpg", ".png", ".log"))
                if os.path.isfile(file_path) and is_video and not is_essential:
                    try:
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        freed_bytes += size
                        removed_count += 1
                        logger.info("Cleaned post-publish raw source video: %s (%d bytes)", file_path, size)
                    except OSError as exc:
                        logger.debug("Failed to remove raw source video %s: %s", file_path, exc)

    except Exception as exc:
        logger.warning("cleanup_published_job failed for %s: %s", job_id, exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count}


def run_storage_cleanup(output_dir: str) -> dict:
    """Master storage cleanup pass: purges partial downloads and verified published clip video files.

    Returns summary dictionary with freed_bytes, freed_mb, removed_files, and cleaned_jobs.
    Never raises. Never deletes unpublished clips.
    """
    total_freed = 0
    total_files = 0
    cleaned_jobs = 0

    if not os.path.exists(output_dir):
        return {"freed_bytes": 0, "freed_mb": 0.0, "removed_files": 0, "cleaned_jobs": 0}

    try:
        p_res = purge_partial_downloads(output_dir)
        total_freed += p_res["freed_bytes"]
        total_files += p_res["removed_count"]

        for entry in os.listdir(output_dir):
            job_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(job_dir) or os.path.islink(job_dir):
                continue
            job_id = entry

            res = cleanup_published_job(job_id, output_dir)
            if res["freed_bytes"] > 0:
                total_freed += res["freed_bytes"]
                total_files += res["removed_count"]
                cleaned_jobs += 1

    except Exception as exc:
        logger.warning("run_storage_cleanup encountered an error: %s", exc)

    freed_mb = round(total_freed / (1024 * 1024), 2)
    logger.info("Storage cleanup pass complete: freed %s MB across %d files and %d jobs", freed_mb, total_files, cleaned_jobs)
    return {
        "freed_bytes": total_freed,
        "freed_mb": freed_mb,
        "removed_files": total_files,
        "cleaned_jobs": cleaned_jobs,
    }

