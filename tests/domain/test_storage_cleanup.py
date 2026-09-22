"""Unit tests for storage cleanup domain functions."""
import os
import json
import shutil
import tempfile
import pytest
from clippyme.domain.job_artifacts import (
    purge_partial_downloads,
    cleanup_failed_job_artifacts,
    cleanup_published_job,
    run_storage_cleanup,
    is_clip_verified_published,
)

@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

def test_is_clip_verified_published():
    # Unpublished clip
    assert not is_clip_verified_published({})
    assert not is_clip_verified_published({"published": []})
    assert not is_clip_verified_published({"published": [{"status": "failed"}]})

    # Verified published clip
    assert is_clip_verified_published({
        "published": [{"post_id": "zernio_post_999", "platforms": ["tiktok"]}]
    })

def test_purge_partial_downloads(temp_output_dir):
    part_file = os.path.join(temp_output_dir, "test_video.mp4.part")
    ytdl_file = os.path.join(temp_output_dir, "test_video.ytdl")
    valid_file = os.path.join(temp_output_dir, "valid_video.mp4")

    with open(part_file, "wb") as f:
        f.write(b"x" * 1024)
    with open(ytdl_file, "wb") as f:
        f.write(b"y" * 512)
    with open(valid_file, "wb") as f:
        f.write(b"z" * 2048)

    res = purge_partial_downloads(temp_output_dir)
    assert res["removed_count"] == 2
    assert res["freed_bytes"] == 1536
    assert not os.path.exists(part_file)
    assert not os.path.exists(ytdl_file)
    assert os.path.exists(valid_file)

def test_unpublished_clips_are_never_deleted_by_storage_cleanup(temp_output_dir):
    job_id = "job_freshly_completed"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    clip1 = os.path.join(job_dir, "video_clip_1.mp4")
    clip2 = os.path.join(job_dir, "video_clip_2.mp4")
    cover1 = os.path.join(job_dir, "video_clip_1_cover.jpg")
    meta_file = os.path.join(job_dir, f"{job_id}_metadata.json")

    with open(clip1, "wb") as f:
        f.write(b"video1" * 1000)
    with open(clip2, "wb") as f:
        f.write(b"video2" * 1000)
    with open(cover1, "wb") as f:
        f.write(b"cover1" * 100)

    meta_data = {
        "job_id": job_id,
        "shorts": [
            {"clip_filename": "video_clip_1.mp4", "qa": {"ok": True}},
            {"clip_filename": "video_clip_2.mp4", "qa": {"ok": True}},
        ],
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    # Run master background sweep
    res = run_storage_cleanup(temp_output_dir)
    assert res["removed_files"] == 0
    assert res["freed_bytes"] == 0

    # Ensure all clips and covers are completely preserved
    assert os.path.exists(clip1)
    assert os.path.exists(clip2)
    assert os.path.exists(cover1)
    assert os.path.exists(meta_file)

def test_only_verified_published_clip_is_deleted(temp_output_dir):
    job_id = "job_partially_published"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    clip1 = os.path.join(job_dir, "video_clip_1.mp4")
    clip2 = os.path.join(job_dir, "video_clip_2.mp4")
    cover1 = os.path.join(job_dir, "video_clip_1_cover.jpg")
    cover2 = os.path.join(job_dir, "video_clip_2_cover.jpg")
    meta_file = os.path.join(job_dir, f"{job_id}_metadata.json")

    with open(clip1, "wb") as f:
        f.write(b"video1" * 1000)
    with open(clip2, "wb") as f:
        f.write(b"video2" * 1000)
    with open(cover1, "wb") as f:
        f.write(b"cover1" * 100)
    with open(cover2, "wb") as f:
        f.write(b"cover2" * 100)

    meta_data = {
        "job_id": job_id,
        "shorts": [
            {
                "clip_filename": "video_clip_1.mp4",
                "qa": {"ok": True},
                "published": [{"post_id": "zernio_post_1", "platforms": ["tiktok"]}],
            },
            {
                "clip_filename": "video_clip_2.mp4",
                "qa": {"ok": True},
                "published": [],
            },
        ],
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    # Run cleanup on job
    res = cleanup_published_job(job_id, temp_output_dir)
    assert res["removed_count"] >= 1

    # Clip 1 should be cleaned, but Clip 2 MUST be preserved!
    assert not os.path.exists(clip1)
    assert os.path.exists(clip2)

    # Thumbnails and metadata MUST be preserved!
    assert os.path.exists(cover1)
    assert os.path.exists(cover2)
    assert os.path.exists(meta_file)

def test_all_clips_published_cleans_source_and_preserves_meta(temp_output_dir):
    job_id = "job_fully_published"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source_video = os.path.join(job_dir, "source_Epstein_388.mp4")
    clip1 = os.path.join(job_dir, "video_clip_1.mp4")
    cover1 = os.path.join(job_dir, "video_clip_1_cover.jpg")
    meta_file = os.path.join(job_dir, f"{job_id}_metadata.json")

    with open(source_video, "wb") as f:
        f.write(b"source" * 5000)
    with open(clip1, "wb") as f:
        f.write(b"clip" * 1000)
    with open(cover1, "wb") as f:
        f.write(b"cover" * 100)

    meta_data = {
        "job_id": job_id,
        "shorts": [
            {
                "clip_filename": "video_clip_1.mp4",
                "qa": {"ok": True},
                "published": [{"post_id": "zernio_post_1", "platforms": ["tiktok"]}],
            }
        ],
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    res = run_storage_cleanup(temp_output_dir)
    assert res["removed_files"] >= 2
    assert res["cleaned_jobs"] == 1

    # Videos cleaned, but metadata & thumbnail remain!
    assert not os.path.exists(source_video)
    assert not os.path.exists(clip1)
    assert os.path.exists(cover1)
    assert os.path.exists(meta_file)


def test_purge_partial_downloads_removes_flac_asr_dumps(temp_output_dir):
    flac_asr = os.path.join(temp_output_dir, ".asr_1787037867_21724.flac")
    temp_flac = os.path.join(temp_output_dir, "temp_audio.flac")
    normal_flac = os.path.join(temp_output_dir, "music.flac")

    with open(flac_asr, "wb") as f:
        f.write(b"x" * 2048)
    with open(temp_flac, "wb") as f:
        f.write(b"y" * 1024)
    with open(normal_flac, "wb") as f:
        f.write(b"z" * 512)

    res = purge_partial_downloads(temp_output_dir)
    assert res["removed_count"] >= 2
    assert not os.path.exists(flac_asr)
    assert not os.path.exists(temp_flac)
    assert os.path.exists(normal_flac)


def test_purge_partial_downloads_preserves_alive_pid_asr(temp_output_dir):
    my_pid = os.getpid()
    active_asr = os.path.join(temp_output_dir, f".asr_1787037867_{my_pid}.flac")
    dead_asr = os.path.join(temp_output_dir, ".asr_1787037867_9999999.flac")

    with open(active_asr, "wb") as f:
        f.write(b"alive" * 1024)
    with open(dead_asr, "wb") as f:
        f.write(b"dead" * 1024)

    res = purge_partial_downloads(temp_output_dir)
    assert os.path.exists(active_asr)
    assert not os.path.exists(dead_asr)


def test_purge_partial_downloads_protects_active_job_ids(temp_output_dir):
    active_job_id = "job_in_flight_123"
    active_dir = os.path.join(temp_output_dir, active_job_id)
    os.makedirs(active_dir, exist_ok=True)

    active_part = os.path.join(active_dir, "downloading.mp4.part")
    active_flac = os.path.join(active_dir, ".asr_123_456.flac")
    with open(active_part, "wb") as f:
        f.write(b"part" * 500)
    with open(active_flac, "wb") as f:
        f.write(b"flac" * 500)

    # Calling with protected_job_ids skips this directory completely
    res = purge_partial_downloads(temp_output_dir, protected_job_ids=[active_job_id])
    assert res["removed_count"] == 0
    assert os.path.exists(active_part)
    assert os.path.exists(active_flac)


def test_purge_orphaned_uploads():
    temp_upload_dir = tempfile.mkdtemp()
    try:
        active_file = os.path.join(temp_upload_dir, "active_upload.mp4")
        stale_file = os.path.join(temp_upload_dir, "stale_upload.mp4")

        with open(active_file, "wb") as f:
            f.write(b"active" * 500)
        with open(stale_file, "wb") as f:
            f.write(b"stale" * 500)

        # Set stale mtime to 1 hour ago
        stale_mtime = os.path.getmtime(stale_file) - 3600
        os.utime(stale_file, (stale_mtime, stale_mtime))

        from clippyme.domain.job_artifacts import purge_orphaned_uploads
        res = purge_orphaned_uploads(
            temp_upload_dir,
            active_paths=(active_file,),
            min_age_seconds=900,
        )
        assert res["removed_count"] == 1
        assert os.path.exists(active_file)
        assert not os.path.exists(stale_file)
    finally:
        shutil.rmtree(temp_upload_dir, ignore_errors=True)


def test_purge_completed_job_source_videos_preserves_slices_and_clips(temp_output_dir):
    from clippyme.domain.job_artifacts import purge_completed_job_source_videos

    job_id = "job_reclaim_test"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source_video = os.path.join(job_dir, "Big_Full_Interview.mp4")
    reframe_slice = os.path.join(job_dir, "source_Big_Full_Interview_clip_1.mp4")
    rendered_clip = os.path.join(job_dir, "Big_Full_Interview_clip_1.mp4")
    cover = os.path.join(job_dir, "Big_Full_Interview_clip_1_cover.jpg")
    meta_file = os.path.join(job_dir, f"{job_id}_metadata.json")

    with open(source_video, "wb") as f:
        f.write(b"SOURCE" * 10000)
    with open(reframe_slice, "wb") as f:
        f.write(b"SLICE" * 2000)
    with open(rendered_clip, "wb") as f:
        f.write(b"CLIP" * 2000)
    with open(cover, "wb") as f:
        f.write(b"COVER" * 100)

    meta_data = {
        "job_id": job_id,
        "shorts": [
            {
                "clip_filename": "Big_Full_Interview_clip_1.mp4",
                "qa": {"ok": True},
                "published": [],  # NOT published
            }
        ],
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_data, f)

    res = purge_completed_job_source_videos(temp_output_dir, max_age_seconds=None)
    assert res["removed_count"] == 1
    assert res["cleaned_jobs"] == 1

    # Raw full-length source video is purged:
    assert not os.path.exists(source_video)

    # Reframe slice, rendered clip, thumbnail, and metadata MUST remain intact:
    assert os.path.exists(reframe_slice)
    assert os.path.exists(rendered_clip)
    assert os.path.exists(cover)
    assert os.path.exists(meta_file)


def test_purge_completed_job_source_videos_respects_3day_retention(temp_output_dir):
    from clippyme.domain.job_artifacts import purge_completed_job_source_videos

    job_id = "job_3day_retention_test"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)

    source_video = os.path.join(job_dir, "Long_Video.mp4")
    rendered_clip = os.path.join(job_dir, "Long_Video_clip_1.mp4")
    meta_file = os.path.join(job_dir, f"{job_id}_metadata.json")

    with open(source_video, "wb") as f:
        f.write(b"S" * 20000)
    with open(rendered_clip, "wb") as f:
        f.write(b"C" * 2000)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({"job_id": job_id, "shorts": [{"clip_filename": "Long_Video_clip_1.mp4"}]}, f)

    # 1. Fresh job (< 3 days): should NOT be purged
    three_days_seconds = 3 * 86400
    res_fresh = purge_completed_job_source_videos(temp_output_dir, max_age_seconds=three_days_seconds)
    assert res_fresh["removed_count"] == 0
    assert os.path.exists(source_video)

    # 2. Stale job (set mtime to 4 days ago): should be purged
    four_days_ago = os.path.getmtime(job_dir) - (4 * 86400)
    os.utime(job_dir, (four_days_ago, four_days_ago))
    res_stale = purge_completed_job_source_videos(temp_output_dir, max_age_seconds=three_days_seconds)
    assert res_stale["removed_count"] == 1
    assert not os.path.exists(source_video)
    assert os.path.exists(rendered_clip)


def test_get_storage_breakdown(temp_output_dir):
    from clippyme.domain.job_artifacts import get_storage_breakdown

    job_dir = os.path.join(temp_output_dir, "job_stats")
    os.makedirs(job_dir, exist_ok=True)

    with open(os.path.join(job_dir, "source.mp4"), "wb") as f:
        f.write(b"A" * 1024 * 1024)  # 1 MB source video
    with open(os.path.join(job_dir, "source_clip_1.mp4"), "wb") as f:
        f.write(b"B" * 512 * 1024)   # 0.5 MB slice
    with open(os.path.join(job_dir, "clip_1.mp4"), "wb") as f:
        f.write(b"C" * 256 * 1024)   # 0.25 MB rendered clip
    with open(os.path.join(job_dir, ".asr_test.flac"), "wb") as f:
        f.write(b"D" * 128 * 1024)   # 0.125 MB temp audio

    stats = get_storage_breakdown(temp_output_dir)
    assert stats["total_jobs"] == 1
    assert stats["source_videos_mb"] >= 0.9
    assert stats["source_slices_mb"] >= 0.4
    assert stats["rendered_clips_mb"] >= 0.2
    assert stats["transient_temp_mb"] >= 0.1


def test_is_pid_alive():
    from clippyme.domain.job_artifacts import _is_pid_alive
    assert _is_pid_alive(os.getpid()) is True
    assert _is_pid_alive(0) is False
    assert _is_pid_alive(-10) is False
    assert _is_pid_alive(99999999) is False


def test_purge_partial_downloads_protects_active_job_dir(temp_output_dir):
    job_id = "job_active_123"
    job_dir = os.path.join(temp_output_dir, job_id)
    os.makedirs(job_dir, exist_ok=True)
    temp_asr = os.path.join(job_dir, f".asr_12345_{os.getpid()}.flac")
    with open(temp_asr, "wb") as f:
        f.write(b"AUDIO")

    # Passing job_id in protected_job_ids must ensure the file is NOT purged
    res = purge_partial_downloads(temp_output_dir, protected_job_ids=[job_id])
    assert res["removed_count"] == 0
    assert os.path.exists(temp_asr)


