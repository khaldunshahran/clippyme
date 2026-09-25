"""YouTube/Twitch/Kick download + filename/cookies helpers.

Extracted from ``pipeline.main`` as part of the decomposition. Depends only on
``yt_dlp`` + stdlib (no cv2/torch/mediapipe), so it imports and is testable on
the host.
"""
import ipaddress
import json
import os
import re
import sys
import time
from urllib.parse import urlparse

import yt_dlp

from clippyme.netutil import resolve_host_addresses


# Remote URL jobs are intentionally limited to the platforms ClippyMe actually
# supports.  The old validator accepted every public HTTP(S) host; because
# yt-dlp follows redirects and extractor-provided media URLs, that exposed a
# broad server-side fetch primitive even though the first hostname was checked
# for private IPs.  Exact official hosts + HTTPS keep user jobs on the expected
# trust boundary while still covering YouTube, Twitch clips/VODs and Kick VODs.
_SUPPORTED_SOURCE_HOSTS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
    "twitch.tv",
    "www.twitch.tv",
    "m.twitch.tv",
    "clips.twitch.tv",
    "kick.com",
    "www.kick.com",
})


def validate_supported_source_url(url: str) -> str:
    """Validate a user/monitor URL before yt-dlp is allowed to resolve it."""
    raw = (url or "").strip()
    try:
        parsed = urlparse(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid source URL") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme.lower() != "https"
        or host not in _SUPPORTED_SOURCE_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise ValueError(
            "source URL must be an official HTTPS YouTube, Twitch, or Kick URL"
        )
    return raw


def _reject_rebound_internal(url: str) -> None:
    """Re-resolve the URL host at download time and refuse internal ranges.

    This is a second line of defence against DNS rebinding after the API-layer
    validation. Resolution failures are left to yt-dlp, but any internal address
    in a mixed answer is rejected.
    """
    try:
        host = urlparse(url).hostname
        if not host:
            return
        try:
            ip_obj = ipaddress.ip_address(host)
            addrs = [ip_obj]
        except ValueError:
            addrs = resolve_host_addresses(host, timeout=5.0)
        if any(
            a.is_private
            or a.is_loopback
            or a.is_link_local
            or a.is_reserved
            or a.is_multicast
            or a.is_unspecified
            for a in addrs
        ):
            raise ValueError(f"refusing download: {host} resolves to an internal address")
    except ValueError:
        raise
    except Exception:
        # A transient resolver failure is not an SSRF bypass now that the host
        # itself is an exact supported-platform allow-list entry. yt-dlp will
        # surface the actual network error to the job.
        return


def sanitize_filename(filename: str) -> str:
    """Remove invalid and problematic unicode/Windows characters from filename."""
    if not filename:
        return "video"
    mapped = (
        filename.replace("｜", "")
        .replace("：", "")
        .replace("／", "")
        .replace("＼", "")
        .replace("？", "")
        .replace("＊", "")
        .replace("＜", "")
        .replace("＞", "")
        .replace("“", "")
        .replace("”", "")
        .replace("‘", "")
        .replace("’", "")
        .replace("'", "")
    )
    cleaned = re.sub(r'[<>:"/\\|?*#%\x00-\x1f]', "", mapped)
    cleaned = re.sub(r"\s+", "_", cleaned).lstrip("-.")
    return cleaned[:100] or "video"


def _resolve_cookies_path(explicit: str | None) -> str | None:
    """Resolve the cookies.txt path used by yt-dlp.

    Precedence:
      1. Explicit path passed on the CLI / by the caller.
      2. Repo-root ``data/cookies.txt`` (the path the dashboard writes to).
      3. ``YOUTUBE_COOKIES`` env var materialized into ``data/cookies_env.txt``.
      4. None (no cookies).
    """
    if explicit:
        return explicit
    repo_root_cookies = os.path.join("data", "cookies.txt")
    if os.path.exists(repo_root_cookies):
        return os.path.abspath(repo_root_cookies)
    env_cookies = os.environ.get("YOUTUBE_COOKIES")
    if env_cookies:
        env_path = os.path.join("data", "cookies_env.txt")
        os.makedirs(os.path.dirname(env_path) or ".", exist_ok=True)
        fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(env_cookies)
        return os.path.abspath(env_path)
    return None


# Video format ladder. avc1 (H.264) first so downstream ffmpeg/x264 passes
# never have to transcode VP9/AV1; the generic tail stops AV1/VP9-only serves
# from hard-failing when no avc1 rendition exists.
_FORMAT_LADDER = (
    'bestvideo[vcodec^=avc1][height>=720][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
    'bestvideo[height>=720][height<=1080]+bestaudio/'
    'bestvideo[height<=1080]+bestaudio/'
    'best'
)

# Player-client fallback chain (bot-resistant mobile/VR clients first).
_DEFAULT_PLAYER_CLIENTS = ("android_vr", "android", "ios", "web_safari", "default")


def _player_client_chain():
    """Return player-client attempts, optionally overridden by the environment."""
    raw = (os.environ.get("YTDLP_PLAYER_CLIENTS") or "").strip()
    if raw:
        chain = [a.strip() for a in raw.split(",") if a.strip()]
        if chain:
            return chain
    return list(_DEFAULT_PLAYER_CLIENTS)


def _extractor_args_for(attempt: str):
    """Map an attempt spec to yt-dlp extractor_args, or None for defaults."""
    if not attempt or attempt.lower() == "default":
        return None
    clients = [c.strip() for c in attempt.split("+") if c.strip()]
    return {"youtube": {"player_client": clients}}


def classify_download_error(msg: str) -> str:
    """Classify a yt-dlp error as ``retry`` or ``fatal``."""
    m = (msg or "").lower()
    m = m.replace("\u2019", "'").replace("\u2018", "'")
    fatal_signals = (
        "private video",
        "this video is private",
        "video has been removed",
        "removed by the user",
        "account associated with this video has been terminated",
        "video is no longer available",
        "video unavailable",
        "not available in your country",
        "not available in your location",
        "blocked it in your country",
        "geo-restrict",
        "geo restrict",
        "geoblock",
        "geo-block",
        "geo block",
    )
    if any(s in m for s in fatal_signals):
        return "fatal"
    retry_signals = (
        "http error 403",
        "403 forbidden",
        "403:",
        "requested format is not available",
        "requested format not available",
        "no formats found",
        "no video formats",
        "empty formats",
        "page needs to be reloaded",
        "the page needs to be reloaded",
        "cookies are no longer valid",
        "only images are available",
        "skipping client",
        "sign in to confirm you're not a bot",
        "sign in to confirm youre not a bot",
        "confirm your age",
    )
    if any(s in m for s in retry_signals):
        return "retry"
    return "fatal"


SOURCE_INFO_FILENAME = "source_info.json"


def _write_source_info(output_dir, info):
    """Persist source-channel metadata as a best-effort sidecar."""
    try:
        from clippyme.domain.banner import suggest_banner

        channel_url = info.get("channel_url") or info.get("uploader_url")
        webpage_url = info.get("webpage_url") or info.get("original_url")
        uploader_id = info.get("uploader_id") or info.get("channel_id")
        banner = suggest_banner(channel_url or webpage_url or "", channel_hint=uploader_id)
        data = {
            "uploader_id": uploader_id,
            "channel_url": channel_url,
            "webpage_url": webpage_url,
            "banner": banner,
        }
        tmp = os.path.join(output_dir, SOURCE_INFO_FILENAME + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, os.path.join(output_dir, SOURCE_INFO_FILENAME))

        # Capture and persist rich audience intelligence (comments, timestamps, description)
        from clippyme.pipeline.audience_intel import parse_audience_intel, save_audience_intel
        intel = parse_audience_intel(info)
        if intel:
            save_audience_intel(output_dir, intel)
    except Exception as exc:  # pragma: no cover - telemetry only, never fatal
        print(f"   ⚠️  source_info/audience_intel capture skipped: {exc}")


def download_youtube_video(url, output_dir=".", cookies_file_path=None):
    """Download a supported remote source by calling the downloader microservice.

    Returns ``(downloaded_file, sanitized_title, quality_warning)`` —
    ``quality_warning`` is a human-readable string when the download probed
    below 720p (Phase 1D(a)), else None. Callers persist it on the job
    metadata so the dashboard can badge degraded sources.
    """
    import httpx

    url = validate_supported_source_url(url)
    _reject_rebound_internal(url)
    print("📥 Calling Downloader Microservice...")
    step_start_time = time.time()

    # We pass absolute path for output_dir because the microservice shares the filesystem
    abs_output_dir = os.path.abspath(output_dir)
    abs_cookies_file_path = os.path.abspath(cookies_file_path) if cookies_file_path else None

    payload = {
        "url": url,
        "output_dir": abs_output_dir,
        "cookies_file_path": abs_cookies_file_path,
    }

    headers = {}
    internal_token = os.environ.get("CLIPPYME_INTERNAL_TOKEN") or os.environ.get("CLIPPYME_API_TOKEN")
    if internal_token:
        headers["X-API-Token"] = internal_token

    try:
        with httpx.Client(timeout=3600.0) as client:
            response = client.post("http://127.0.0.1:8001/download", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            downloaded_file = data["downloaded_file"]
            sanitized_title = data["sanitized_title"]
            quality_warning = data.get("quality_warning")

        step_end_time = time.time()
        print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
        if quality_warning:
            print(f"⚠️  QUALITY WARNING (persisted on job): {quality_warning}", flush=True)
        return downloaded_file, sanitized_title, quality_warning
    except httpx.ConnectError as conn_err:
        print(f"⚠️ Downloader microservice on port 8001 not reachable ({conn_err}). Falling back to in-process download...", file=sys.stderr)
        try:
            from clippyme.services.downloader_api import download_video as local_download, DownloadRequest
            req = DownloadRequest(
                url=url,
                output_dir=abs_output_dir,
                cookies_file_path=abs_cookies_file_path,
            )
            data = local_download(req)
            downloaded_file = data["downloaded_file"]
            sanitized_title = data["sanitized_title"]
            quality_warning = data.get("quality_warning")
            step_end_time = time.time()
            print(f"✅ Video downloaded via in-process fallback in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
            if quality_warning:
                print(f"⚠️  QUALITY WARNING (persisted on job): {quality_warning}", flush=True)
            return downloaded_file, sanitized_title, quality_warning
        except Exception as fallback_err:
            print(f"❌ In-process fallback failed: {fallback_err}", file=sys.stderr)
            raise RuntimeError(f"Downloader microservice failed: {conn_err} (in-process fallback failed: {fallback_err})")
    except httpx.HTTPError as e:
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            try:
                resp_json = e.response.json()
                detail = resp_json.get("detail", "")
            except Exception:
                detail = e.response.text or ""

        print(f"❌ Downloader Microservice failed: {e}")
        if detail:
            print(f"Response: {detail}")

        print("🚨 SOURCE DOWNLOAD ERROR 🚨", file=sys.stderr)
        error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: SOURCE DOWNLOAD FAILED
❌ ================================================================= ❌

The remote platform refused or could not complete the download.
Technical Details: {detail or e}
        """
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        err_str = f": {detail}" if detail else f": {e}"
        raise RuntimeError(f"Downloader microservice failed{err_str}")

