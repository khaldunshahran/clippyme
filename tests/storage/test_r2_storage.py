"""Unit tests for Cloudflare R2 / S3 storage integration."""
import datetime
import json
import os
from unittest.mock import MagicMock, patch
import pytest

from clippyme.storage.r2_storage import (
    get_clip_url,
    get_r2_config,
    get_remote_clip_url,
    is_r2_configured,
    sign_v4_headers,
    sync_job_clips_to_r2,
    upload_bytes_to_r2,
    upload_file_to_r2,
)


def test_get_r2_config_defaults(monkeypatch):
    monkeypatch.delenv("R2_ENABLED", raising=False)
    monkeypatch.delenv("R2_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("R2_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("R2_BUCKET_NAME", raising=False)
    monkeypatch.delenv("R2_PUBLIC_DOMAIN", raising=False)

    cfg = get_r2_config()
    assert cfg["enabled"] is False
    assert is_r2_configured(cfg) is False


def test_get_r2_config_configured(monkeypatch):
    monkeypatch.setenv("R2_ENABLED", "1")
    monkeypatch.setenv("R2_ACCOUNT_ID", "acc123")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "key123")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "sec456")
    monkeypatch.setenv("R2_BUCKET_NAME", "mybucket")
    monkeypatch.setenv("R2_PUBLIC_DOMAIN", "https://cdn.example.com")

    cfg = get_r2_config()
    assert cfg["enabled"] is True
    assert cfg["endpoint_url"] == "https://acc123.r2.cloudflarestorage.com"
    assert is_r2_configured(cfg) is True


def test_sign_v4_headers_deterministic():
    fixed_time = datetime.datetime(2026, 9, 6, 12, 0, 0, tzinfo=datetime.timezone.utc)
    payload = b"test clip data"
    headers = {"Content-Type": "video/mp4"}
    
    signed = sign_v4_headers(
        method="PUT",
        url="https://acc123.r2.cloudflarestorage.com/mybucket/clip.mp4",
        headers=headers,
        payload_bytes=payload,
        access_key="test_access_key",
        secret_key="test_secret_key",
        region="auto",
        service="s3",
        timestamp=fixed_time,
    )

    assert signed["Host"] == "acc123.r2.cloudflarestorage.com"
    assert signed["x-amz-date"] == "20260906T120000Z"
    assert "x-amz-content-sha256" in signed
    assert signed["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=test_access_key/20260906/auto/s3/aws4_request")
    assert "Signature=" in signed["Authorization"]


def test_upload_bytes_unconfigured():
    with pytest.raises(ValueError, match="not configured"):
        upload_bytes_to_r2(b"dummy", "clip.mp4", cfg={"enabled": False})


def test_upload_bytes_successful():
    cfg = {
        "enabled": True,
        "access_key_id": "test_key",
        "secret_access_key": "test_sec",
        "bucket_name": "test_bucket",
        "endpoint_url": "https://acc123.r2.cloudflarestorage.com",
        "public_domain": "https://cdn.clippyme.com",
        "region": "auto",
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        url = upload_bytes_to_r2(b"video content", "clips/job1/clip_0.mp4", cfg=cfg)
        assert url == "https://cdn.clippyme.com/clips/job1/clip_0.mp4"
        assert mock_urlopen.called
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "PUT"
        assert req.full_url == "https://acc123.r2.cloudflarestorage.com/test_bucket/clips/job1/clip_0.mp4"


def test_get_clip_url_local_fallback(tmp_path):
    url = get_clip_url("job_123", "clip_0.mp4", str(tmp_path))
    assert url == "/videos/job_123/clip_0.mp4"


def test_get_clip_url_cached_r2(tmp_path):
    cache_file = tmp_path / "r2_urls.json"
    cache_file.write_text(json.dumps({"clip_0.mp4": "https://cdn.clippyme.com/clips/job_123/clip_0.mp4"}), encoding="utf-8")

    url = get_clip_url("job_123", "clip_0.mp4", str(tmp_path))
    assert url == "https://cdn.clippyme.com/clips/job_123/clip_0.mp4"
    assert get_remote_clip_url("job_123", "clip_0.mp4", str(tmp_path)) == "https://cdn.clippyme.com/clips/job_123/clip_0.mp4"


def test_sync_job_clips_to_r2(tmp_path):
    job_dir = tmp_path / "job_sync_test"
    job_dir.mkdir()

    clip_file = job_dir / "clip_0.mp4"
    clip_file.write_bytes(b"dummy mp4 data")
    
    thumb_file = job_dir / "thumb_0.jpg"
    thumb_file.write_bytes(b"dummy jpg data")

    source_file = job_dir / "source_raw.mp4"
    source_file.write_bytes(b"raw 16:9 source that should not sync")

    cfg = {
        "enabled": True,
        "access_key_id": "test_key",
        "secret_access_key": "test_sec",
        "bucket_name": "test_bucket",
        "endpoint_url": "https://acc123.r2.cloudflarestorage.com",
        "public_domain": "https://cdn.clippyme.com",
        "region": "auto",
        "auto_cleanup": False,
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        uploaded = sync_job_clips_to_r2("job_sync_test", str(job_dir), cfg=cfg)

    assert "clip_0.mp4" in uploaded
    assert "thumb_0.jpg" in uploaded
    assert "source_raw.mp4" not in uploaded

    cache_path = job_dir / "r2_urls.json"
    assert cache_path.exists()
    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_data["clip_0.mp4"] == "https://cdn.clippyme.com/clips/job_sync_test/clip_0.mp4"
    assert clip_file.exists()  # auto_cleanup was False


def test_sync_job_clips_auto_cleanup(tmp_path):
    job_dir = tmp_path / "job_cleanup_test"
    job_dir.mkdir()

    clip_file = job_dir / "clip_1.mp4"
    clip_file.write_bytes(b"dummy mp4 data")

    cfg = {
        "enabled": True,
        "access_key_id": "test_key",
        "secret_access_key": "test_sec",
        "bucket_name": "test_bucket",
        "endpoint_url": "https://acc123.r2.cloudflarestorage.com",
        "public_domain": "https://cdn.clippyme.com",
        "region": "auto",
        "auto_cleanup": True,
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        uploaded = sync_job_clips_to_r2("job_cleanup_test", str(job_dir), cfg=cfg)

    assert "clip_1.mp4" in uploaded
    assert not clip_file.exists()  # Successfully removed by auto_cleanup!
    cache_data = json.loads((job_dir / "r2_urls.json").read_text(encoding="utf-8"))
    assert cache_data["clip_1.mp4"] == "https://cdn.clippyme.com/clips/job_cleanup_test/clip_1.mp4"
