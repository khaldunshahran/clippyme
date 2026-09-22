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


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is currently alive."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            # On Windows, PermissionError indicates the process exists but is un-signalable.
            return True
        except (OSError, ProcessLookupError):
            return False


def purge_partial_downloads(
    target_dirs: str | list[str] | tuple[str, ...],
    min_age_seconds: float = 0.0,
    active_paths: tuple[str, ...] | set[str] | list[str] = (),
    protected_job_ids: tuple[str, ...] | set[str] | list[str] = (),
) -> dict:
    """Scan directory or directories for partial, temporary, and orphaned ASR audio dumps.

    Removes *.part, *.ytdl, *.tmp, *.temp, and orphaned .asr_*.flac/.wav audio files.
    If min_age_seconds > 0, protects files modified more recently. Never raises exceptions.
    Guarantees that files belonging to active jobs, active paths, or running PIDs are never deleted.
    """
    freed_bytes = 0
    removed_count = 0
    dirs = [target_dirs] if isinstance(target_dirs, str) else list(target_dirs)
    partial_extensions = {".part", ".ytdl", ".tmp", ".temp"}
    now = time.time()
    active_paths_set = {os.path.abspath(p) for p in active_paths if p}
    protected_jobs = {str(jid).strip().lower() for jid in protected_job_ids if jid}

    for d in dirs:
        if not d or not os.path.exists(d):
            continue
        if protected_jobs and os.path.basename(os.path.abspath(d)).lower() in protected_jobs:
            continue
        try:
            for root, dirs_in_root, files in os.walk(d):
                # Never descend into active job folders
                if protected_jobs:
                    dirs_in_root[:] = [sub for sub in dirs_in_root if sub.lower() not in protected_jobs]
                if any(part.lower() in protected_jobs for part in os.path.relpath(root, d).replace("\\", "/").split("/")):
                    continue

                for file in files:
                    lower = file.lower()
                    is_asr_audio = (
                        lower.startswith(".asr_")
                        or (lower.startswith("temp_") and lower.endswith(".flac"))
                        or (lower.endswith(".flac") and "asr" in lower)
                    )
                    is_partial = (
                        any(lower.endswith(ext) for ext in partial_extensions)
                        or (".f" in lower and lower.endswith(".mp4.part"))
                        or is_asr_audio
                    )
                    if is_partial:
                        file_path = os.path.join(root, file)
                        if os.path.abspath(file_path) in active_paths_set:
                            continue

                        # If this is an .asr_ timestamp_pid file, check if owning process is still running
                        if lower.startswith(".asr_"):
                            name_part = file.rsplit(".", 1)[0]
                            parts = name_part.split("_")
                            if len(parts) >= 3 and parts[-1].isdigit():
                                file_pid = int(parts[-1])
                                if _is_pid_alive(file_pid):
                                    logger.debug("Skipping .asr file owned by running PID %d: %s", file_pid, file_path)
                                    continue

                        try:
                            if min_age_seconds > 0 and (now - os.path.getmtime(file_path) < min_age_seconds):
                                continue
                            size = os.path.getsize(file_path)
                            os.remove(file_path)
                            freed_bytes += size
                            removed_count += 1
                            logger.info("Purged transient/partial file: %s (%d bytes)", file_path, size)
                        except OSError as exc:
                            logger.debug("Failed to purge partial file %s: %s", file_path, exc)
        except Exception as exc:
            logger.warning("purge_partial_downloads encountered an error in %s: %s", d, exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count}


def purge_orphaned_uploads(
    upload_dir: str,
    active_paths: tuple[str, ...] | set[str] | list[str] = (),
    min_age_seconds: float = 900.0,
) -> dict:
    """Scan uploads directory and remove files not currently active.

    Files modified within min_age_seconds (default 15 minutes) are protected to
    prevent deleting newly uploaded files that haven't yet been enqueued.
    Files referenced in active_paths are strictly preserved.
    """
    freed_bytes = 0
    removed_count = 0
    if not upload_dir or not os.path.exists(upload_dir):
        return {"freed_bytes": 0, "removed_count": 0}

    now = time.time()
    normalized_active = {os.path.abspath(p) for p in active_paths if p}
    try:
        for fname in os.listdir(upload_dir):
            fpath = os.path.join(upload_dir, fname)
            if not os.path.isfile(fpath) or os.path.islink(fpath):
                continue
            if os.path.abspath(fpath) in normalized_active:
                continue
            try:
                mtime = os.path.getmtime(fpath)
                if now - mtime < min_age_seconds:
                    continue
                size = os.path.getsize(fpath)
                os.remove(fpath)
                freed_bytes += size
                removed_count += 1
                logger.info("Purged orphaned upload: %s (%d bytes)", fpath, size)
            except OSError as exc:
                logger.debug("Failed to remove upload %s: %s", fpath, exc)
    except Exception as exc:
        logger.warning("purge_orphaned_uploads failed: %s", exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count}


def is_job_completed_for_source_purge(job_dir: str, job_id: str) -> bool:
    """True if a job has finished its pipeline and its raw source video can be purged safely."""
    try:
        # Check highlight reels
        hl_matches = glob.glob(os.path.join(job_dir, "highlight_reel_*.mp4"))
        if hl_matches:
            return True

        # Check regular shorts metadata
        matches = glob.glob(os.path.join(job_dir, "*_metadata.json"))
        if not matches:
            return False
        meta_path = max(matches, key=os.path.getmtime)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shorts = data.get("shorts") or data.get("clips") or []
        if not shorts:
            return False
        # Confirms shorts exist in metadata
        return True
    except Exception:
        return False


def purge_completed_job_source_videos(
    output_dir: str,
    max_age_seconds: float | None = None,
    protected_job_ids: tuple[str, ...] | set[str] | list[str] = (),
) -> dict:
    """Purge full-length raw source videos from completed jobs, keeping clips and slices intact.

    If max_age_seconds is set (e.g. 3 * 86400 for 3 days), only jobs older than that age
    will have their source video purged. If None, all completed jobs are eligible.
    Never removes rendered clips, reframing slices (source_*_clip_*.mp4), or metadata.
    """
    freed_bytes = 0
    removed_count = 0
    cleaned_jobs = 0
    if not os.path.exists(output_dir):
        return {"freed_bytes": 0, "removed_count": 0, "cleaned_jobs": 0}

    from clippyme.pipeline.run_ops import find_source_video_candidate, is_clip_artifact

    now = time.time()
    protected_jobs = {str(jid).strip().lower() for jid in protected_job_ids if jid}
    try:
        for entry in os.listdir(output_dir):
            job_dir = os.path.join(output_dir, entry)
            if not os.path.isdir(job_dir) or os.path.islink(job_dir):
                continue
            if entry.lower() in protected_jobs:
                continue

            if max_age_seconds is not None:
                try:
                    if now - os.path.getmtime(job_dir) < max_age_seconds:
                        continue
                except OSError:
                    continue

            if not is_job_completed_for_source_purge(job_dir, entry):
                continue

            src_candidate = find_source_video_candidate(job_dir)
            if not src_candidate or not os.path.isfile(src_candidate):
                continue

            # Strict protection against clip artifacts
            if is_clip_artifact(os.path.basename(src_candidate)):
                continue

            try:
                sz = os.path.getsize(src_candidate)
                os.remove(src_candidate)
                freed_bytes += sz
                removed_count += 1
                cleaned_jobs += 1
                logger.info("Purged raw source video for completed job %s: %s (%d bytes)", entry, src_candidate, sz)
            except OSError as exc:
                logger.debug("Failed to remove source video %s: %s", src_candidate, exc)
    except Exception as exc:
        logger.warning("purge_completed_job_source_videos encountered an error: %s", exc)

    return {"freed_bytes": freed_bytes, "removed_count": removed_count, "cleaned_jobs": cleaned_jobs}


def get_storage_breakdown(output_dir: str, upload_dir: str | None = None) -> dict:
    """Analyze and categorize disk usage across output and upload directories."""
    stats = {
        "total_bytes": 0,
        "total_mb": 0.0,
        "source_videos_bytes": 0,
        "source_videos_mb": 0.0,
        "source_slices_bytes": 0,
        "source_slices_mb": 0.0,
        "rendered_clips_bytes": 0,
        "rendered_clips_mb": 0.0,
        "uploads_bytes": 0,
        "uploads_mb": 0.0,
        "transient_temp_bytes": 0,
        "transient_temp_mb": 0.0,
        "metadata_bytes": 0,
        "metadata_mb": 0.0,
        "total_jobs": 0,
    }

    if os.path.exists(output_dir):
        try:
            for entry in os.listdir(output_dir):
                job_path = os.path.join(output_dir, entry)
                if not os.path.isdir(job_path) or os.path.islink(job_path):
                    continue
                stats["total_jobs"] += 1
                for root, _, files in os.walk(job_path):
                    for file in files:
                        fp = os.path.join(root, file)
                        try:
                            sz = os.path.getsize(fp)
                        except OSError:
                            continue
                        stats["total_bytes"] += sz
                        lower = file.lower()
                        if lower.startswith(".asr_") or (lower.endswith(".flac") and "asr" in lower) or lower.endswith((".part", ".ytdl", ".tmp", ".temp")):
                            stats["transient_temp_bytes"] += sz
                        elif lower.startswith("source_") and lower.endswith(".mp4"):
                            stats["source_slices_bytes"] += sz
                        elif lower.startswith("composed_") or lower.startswith("highlight_reel_") or lower.startswith("clip_") or ("_clip_" in lower and lower.endswith(".mp4")):
                            stats["rendered_clips_bytes"] += sz
                        elif lower.endswith((".jpg", ".png", ".json", ".log", ".ass")):
                            stats["metadata_bytes"] += sz
                        elif lower.endswith((".mp4", ".mkv", ".webm", ".avi", ".mov")):
                            stats["source_videos_bytes"] += sz
                        else:
                            stats["metadata_bytes"] += sz
        except Exception as exc:
            logger.warning("get_storage_breakdown error scanning output_dir: %s", exc)

    if upload_dir and os.path.exists(upload_dir):
        try:
            for fname in os.listdir(upload_dir):
                fp = os.path.join(upload_dir, fname)
                if not os.path.isfile(fp) or os.path.islink(fp):
                    continue
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                stats["total_bytes"] += sz
                lower = fname.lower()
                if lower.startswith(".asr_") or lower.endswith(".flac") or lower.endswith((".part", ".tmp")):
                    stats["transient_temp_bytes"] += sz
                else:
                    stats["uploads_bytes"] += sz
        except Exception as exc:
            logger.warning("get_storage_breakdown error scanning upload_dir: %s", exc)

    stats["total_mb"] = round(stats["total_bytes"] / (1024 * 1024), 2)
    stats["source_videos_mb"] = round(stats["source_videos_bytes"] / (1024 * 1024), 2)
    stats["source_slices_mb"] = round(stats["source_slices_bytes"] / (1024 * 1024), 2)
    stats["rendered_clips_mb"] = round(stats["rendered_clips_bytes"] / (1024 * 1024), 2)
    stats["uploads_mb"] = round(stats["uploads_bytes"] / (1024 * 1024), 2)
    stats["transient_temp_mb"] = round(stats["transient_temp_bytes"] / (1024 * 1024), 2)
    stats["metadata_mb"] = round(stats["metadata_bytes"] / (1024 * 1024), 2)

    return stats


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
                if not clip.get("deleted_after_publish"):
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


def run_storage_cleanup(
    output_dir: str,
    upload_dir: str | None = None,
    purge_raw_sources: bool = False,
    source_max_age_seconds: float | None = None,
    active_paths: tuple[str, ...] | set[str] | list[str] = (),
    partial_min_age_seconds: float = 0.0,
    protected_job_ids: tuple[str, ...] | set[str] | list[str] = (),
) -> dict:
    """Master storage cleanup pass across all data tiers.

    - Purges partial downloads and orphaned ASR audio dumps across output and upload dirs.
    - Purges orphaned uploads (unless referenced by active_paths).
    - Cleans verified published clips and their source slices.
    - Optionally purges completed job raw source videos (or those older than source_max_age_seconds).
    - Never raises. Never deletes unpublished clips or active job files.
    """
    total_freed = 0
    total_files = 0
    cleaned_jobs = 0

    target_dirs = [output_dir]
    if upload_dir and os.path.exists(upload_dir):
        target_dirs.append(upload_dir)

    protected_jobs = {str(jid).strip().lower() for jid in protected_job_ids if jid}

    # 1. Purge partial downloads and transient ASR audio dumps
    p_res = {"freed_bytes": 0, "removed_count": 0}
    try:
        p_res = purge_partial_downloads(
            target_dirs,
            min_age_seconds=partial_min_age_seconds,
            active_paths=active_paths,
            protected_job_ids=protected_job_ids,
        )
        total_freed += p_res["freed_bytes"]
        total_files += p_res["removed_count"]
    except Exception as exc:
        logger.warning("run_storage_cleanup partial purge error: %s", exc)

    # 2. Purge orphaned uploads (if upload_dir provided)
    u_res = {"freed_bytes": 0, "removed_count": 0}
    if upload_dir and os.path.exists(upload_dir):
        try:
            u_res = purge_orphaned_uploads(upload_dir, active_paths=active_paths)
            total_freed += u_res["freed_bytes"]
            total_files += u_res["removed_count"]
        except Exception as exc:
            logger.warning("run_storage_cleanup orphaned uploads purge error: %s", exc)

    # 3. Clean published clips in output_dir
    pub_freed = 0
    if os.path.exists(output_dir):
        try:
            for entry in os.listdir(output_dir):
                job_dir = os.path.join(output_dir, entry)
                if not os.path.isdir(job_dir) or os.path.islink(job_dir):
                    continue
                job_id = entry
                if job_id.lower() in protected_jobs:
                    continue

                res = cleanup_published_job(job_id, output_dir)
                if res["freed_bytes"] > 0:
                    pub_freed += res["freed_bytes"]
                    total_freed += res["freed_bytes"]
                    total_files += res["removed_count"]
                    cleaned_jobs += 1
        except Exception as exc:
            logger.warning("run_storage_cleanup published job sweep error: %s", exc)

    # 4. Source videos of completed jobs (if requested or if source_max_age_seconds is set)
    src_res = {"freed_bytes": 0, "removed_count": 0, "cleaned_jobs": 0}
    if (purge_raw_sources or source_max_age_seconds is not None) and os.path.exists(output_dir):
        try:
            src_res = purge_completed_job_source_videos(
                output_dir,
                max_age_seconds=source_max_age_seconds,
                protected_job_ids=protected_job_ids,
            )
            total_freed += src_res["freed_bytes"]
            total_files += src_res["removed_count"]
            cleaned_jobs += src_res.get("cleaned_jobs", 0)
        except Exception as exc:
            logger.warning("run_storage_cleanup source video purge error: %s", exc)

    freed_mb = round(total_freed / (1024 * 1024), 2)
    logger.info(
        "Storage cleanup pass complete: freed %s MB across %d files and %d jobs",
        freed_mb,
        total_files,
        cleaned_jobs,
    )
    return {
        "freed_bytes": total_freed,
        "freed_mb": freed_mb,
        "removed_files": total_files,
        "cleaned_jobs": cleaned_jobs,
        "categories": {
            "transient_temp_freed_bytes": p_res["freed_bytes"],
            "orphaned_uploads_freed_bytes": u_res["freed_bytes"],
            "published_clips_freed_bytes": pub_freed,
            "source_videos_freed_bytes": src_res["freed_bytes"],
        },
    }

