"""Tests for clippyme.pipeline.download helpers (host-runnable; no network)."""
import os
from unittest.mock import MagicMock

import pytest

from clippyme import netutil
from clippyme.pipeline import download as dl


def _fake_getaddrinfo(*ips):
    """Build a getaddrinfo stub returning the given IP strings."""
    def _stub(host, port, *a, **k):
        return [(None, None, None, "", (ip, 0)) for ip in ips]
    return _stub


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abc",
    "https://youtu.be/abcdefghijk?t=1",
    "https://www.twitch.tv/videos/123",
    "https://clips.twitch.tv/FancyClip",
    "https://kick.com/video/1234",
])
def test_validate_supported_source_url_accepts_official_https_hosts(url):
    assert dl.validate_supported_source_url(url) == url


@pytest.mark.parametrize("url", [
    "http://www.youtube.com/watch?v=abc",
    "https://example.com/video.mp4",
    "https://youtube.com.evil.example/watch?v=abc",
    "https://user@www.youtube.com/watch?v=abc",
    "https://www.youtube.com:444/watch?v=abc",
    "file:///etc/passwd",
    "not a url",
])
def test_validate_supported_source_url_rejects_untrusted_sources(url):
    with pytest.raises(ValueError):
        dl.validate_supported_source_url(url)


def test_reject_rebound_literal_internal_ip_raises():
    with pytest.raises(ValueError):
        dl._reject_rebound_internal("http://127.0.0.1/video")


def test_reject_rebound_literal_public_ip_passes():
    # 8.8.8.8 is public — the lower-level rebound guard remains generic.
    dl._reject_rebound_internal("http://8.8.8.8/video")


def test_reject_rebound_all_internal_resolution_raises(monkeypatch):
    monkeypatch.setattr(netutil.socket, "getaddrinfo", _fake_getaddrinfo("192.168.1.10", "127.0.0.1"))
    with pytest.raises(ValueError):
        dl._reject_rebound_internal("http://rebind.evil.test/x")


def test_reject_rebound_public_resolution_passes(monkeypatch):
    monkeypatch.setattr(netutil.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
    dl._reject_rebound_internal("http://example.com/x")  # no raise


def test_reject_rebound_mixed_public_and_internal_raises(monkeypatch):
    # ANY internal address → reject. A split-horizon / round-robin host that
    # returns one public + one loopback/private address must NOT pass.
    monkeypatch.setattr(netutil.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34", "10.0.0.1"))
    with pytest.raises(ValueError):
        dl._reject_rebound_internal("http://example.com/x")


def test_reject_rebound_no_host_returns_none():
    assert dl._reject_rebound_internal("not a url") is None


def test_reject_rebound_resolution_failure_is_swallowed(monkeypatch):
    def _boom(*a, **k):
        raise OSError("dns down")
    monkeypatch.setattr(netutil.socket, "getaddrinfo", _boom)
    assert dl._reject_rebound_internal("http://example.com/x") is None


def test_sanitize_filename_strips_invalid_chars():
    assert dl.sanitize_filename('a<b>c:d"e/f\\g|h?i*j') == "abcdefghij"


def test_sanitize_filename_replaces_spaces():
    assert dl.sanitize_filename("my cool video") == "my_cool_video"


def test_sanitize_filename_truncates_to_100():
    assert len(dl.sanitize_filename("x" * 250)) == 100


def test_resolve_cookies_explicit_wins(tmp_path):
    explicit = str(tmp_path / "given.txt")
    assert dl._resolve_cookies_path(explicit) == explicit


def test_resolve_cookies_repo_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    os.makedirs("data", exist_ok=True)
    open(os.path.join("data", "cookies.txt"), "w").close()
    resolved = dl._resolve_cookies_path(None)
    assert resolved.endswith(os.path.join("data", "cookies.txt"))
    assert os.path.isabs(resolved)


def test_resolve_cookies_from_env_materializes_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOUTUBE_COOKIES", "# Netscape cookie\nfoo\tbar")
    resolved = dl._resolve_cookies_path(None)
    assert resolved.endswith(os.path.join("data", "cookies_env.txt"))
    with open(resolved) as f:
        assert "Netscape" in f.read()


def test_resolve_cookies_none_when_nothing_available(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("YOUTUBE_COOKIES", raising=False)
    assert dl._resolve_cookies_path(None) is None


# ── player-client fallback chain ──────────────────────────────────────────

def test_player_client_chain_default(monkeypatch):
    monkeypatch.delenv("YTDLP_PLAYER_CLIENTS", raising=False)
    assert dl._player_client_chain() == ["android_vr", "android", "ios", "web_safari", "default"]


def test_player_client_chain_env_override(monkeypatch):
    monkeypatch.setenv("YTDLP_PLAYER_CLIENTS", " web_safari , tv , default ")
    assert dl._player_client_chain() == ["web_safari", "tv", "default"]


def test_player_client_chain_blank_env_falls_back(monkeypatch):
    monkeypatch.setenv("YTDLP_PLAYER_CLIENTS", "   ")
    assert dl._player_client_chain() == ["android_vr", "android", "ios", "web_safari", "default"]


def test_extractor_args_default_is_none():
    assert dl._extractor_args_for("default") is None
    assert dl._extractor_args_for("") is None


def test_extractor_args_single_client():
    assert dl._extractor_args_for("web_safari") == {
        "youtube": {"player_client": ["web_safari"]}
    }


def test_extractor_args_joined_clients():
    assert dl._extractor_args_for("tv+tv_embedded") == {
        "youtube": {"player_client": ["tv", "tv_embedded"]}
    }


# ── classify_download_error (retry vs fatal) ──────────────────────────────

@pytest.mark.parametrize("msg", [
    "ERROR: unable to download video data: HTTP Error 403: Forbidden",
    "Requested format is not available",
    "requested format not available. Use --list-formats",
    "No video formats found!; please report this issue",
    "empty formats returned by extractor",
    "403 Forbidden",
    "ERROR: Sign in to confirm you're not a bot. Use --cookies",
])
def test_classify_retry(msg):
    assert dl.classify_download_error(msg) == "retry"


@pytest.mark.parametrize("msg", [
    "ERROR: Private video. Sign in if you've been granted access",
    "This video is private",
    "Video unavailable. This video has been removed by the user",
    "Video unavailable",
    "The uploader has not made this video available in your country",
    "The uploader has blocked it in your country on copyright grounds",
    "This video is no longer available because the account was terminated",
    "some totally unrecognised failure mode",
    "",
])
def test_classify_fatal(msg):
    assert dl.classify_download_error(msg) == "fatal"


def test_classify_bot_wall_beats_any_incidental_403():
    msg = "Sign in to confirm you're not a bot (HTTP Error 403)"
    assert dl.classify_download_error(msg) == "retry"


def test_download_youtube_video_fallback_on_connect_error(monkeypatch, tmp_path):
    """When port 8001 microservice is unreachable, download_youtube_video falls back in-process."""
    import httpx
    from unittest.mock import MagicMock

    def _fail_client(*a, **k):
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused: [WinError 10061]")
        mock_ctx.__enter__.return_value = mock_client
        mock_ctx.__exit__.return_value = False
        return mock_ctx

    monkeypatch.setattr(httpx, "Client", _fail_client)

    fallback_called = {}
    def _mock_local_download(req):
        fallback_called["url"] = req.url
        fallback_called["output_dir"] = req.output_dir
        return {
            "downloaded_file": os.path.join(req.output_dir, "fallback_vid.mp4"),
            "sanitized_title": "fallback_vid",
        }

    monkeypatch.setattr("clippyme.services.downloader_api.download_video", _mock_local_download)

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    out_file, title = dl.download_youtube_video(url, output_dir=str(tmp_path))

    assert fallback_called.get("url") == url
    assert title == "fallback_vid"
    assert out_file.endswith("fallback_vid.mp4")


def test_download_youtube_video_propagates_microservice_error_detail(monkeypatch, tmp_path):
    import httpx
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.json.return_value = {"detail": "Fatal download error: Video unavailable"}
    mock_resp.text = '{"detail": "Fatal download error: Video unavailable"}'

    http_err = httpx.HTTPStatusError("Client error '400 Bad Request'", request=MagicMock(), response=mock_resp)

    def _fail_client(*args, **kwargs):
        mock_ctx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.side_effect = http_err
        mock_ctx.__enter__.return_value = mock_client
        mock_ctx.__exit__.return_value = False
        return mock_ctx

    monkeypatch.setattr(httpx, "Client", _fail_client)

    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    with pytest.raises(RuntimeError) as exc_info:
        dl.download_youtube_video(url, output_dir=str(tmp_path))

    assert "Fatal download error: Video unavailable" in str(exc_info.value)


def test_is_safe_output_dir_casing():
    from clippyme.services.downloader_api import _is_safe_output_dir
    assert _is_safe_output_dir("output/test") is True
    assert _is_safe_output_dir(os.path.abspath("output/test").upper()) is True
    assert _is_safe_output_dir(os.path.abspath("output/test").lower()) is True
    assert _is_safe_output_dir(r"C:\Windows\System32") is False


def test_downloader_api_rejects_ssrf_url():
    from fastapi import HTTPException
    from clippyme.services.downloader_api import download_video, DownloadRequest
    req = DownloadRequest(
        url="http://169.254.169.254/latest/meta-data/",
        output_dir="output/safe",
    )
    with pytest.raises(HTTPException) as exc:
        download_video(req)
    assert exc.value.status_code == 400


def test_downloader_api_rejects_external_host_without_token():
    from fastapi import HTTPException
    from clippyme.services.downloader_api import download_video, DownloadRequest
    class MockClient:
        host = "198.51.100.2"
    class MockReq:
        client = MockClient()
        headers = {}

    req = DownloadRequest(
        url="https://www.youtube.com/watch?v=valid",
        output_dir="output/safe",
    )
    with pytest.raises(HTTPException) as exc:
        download_video(req, request=MockReq())
    assert exc.value.status_code == 403


