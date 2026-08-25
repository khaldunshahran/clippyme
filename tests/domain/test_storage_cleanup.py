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
