"""YouTube uploads-feed poller (long-form only) + channel-id resolution."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger("clippyme")

FEED_URL = "https://www.youtube.com/feeds/videos.xml?playlist_id={playlist}"
USER_AGENT = "Mozilla/5.0 (compatible; ClippyMe-LiveMonitor/1.0)"
MAX_FEED_BYTES = 2 * 1024 * 1024
CHANNEL_RESOLVE_TIMEOUT = 15

_UC_RE = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
_PLAYLIST_RE = re.compile(r"^(?:UULF|UU)[A-Za-z0-9_-]{22}$")
_VIDEO_ID_XML_RE = re.compile(r"<yt:videoId>([^<]+)</yt:videoId>")
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

CHANNEL_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_FEED_OPENER = urllib.request.build_opener(_NoRedirectHandler)


def uploads_playlist_id(channel_id: str) -> str:
    """UC… channel id -> its long-form uploads playlist id (UULF…)."""
    cid = (channel_id or "").strip()
    if not _UC_RE.fullmatch(cid):
        raise ValueError(f"expected a canonical UC channel id, got {channel_id!r}")
    return "UULF" + cid[2:]


def feed_url(playlist_id: str) -> str:
    playlist = (playlist_id or "").strip()
    if not _PLAYLIST_RE.fullmatch(playlist):
        raise ValueError(f"invalid uploads playlist id: {playlist_id!r}")
    return FEED_URL.format(playlist=playlist)


def channel_feed_url(channel_id: str) -> str:
    """Canonical channel_id feed URL for fallback when UULF playlist is unavailable."""
    cid = (channel_id or "").strip()
    if not _UC_RE.fullmatch(cid):
        raise ValueError(f"expected a canonical UC channel id, got {channel_id!r}")
    return CHANNEL_FEED_URL.format(channel_id=cid)


def parse_feed(xml) -> list:
    """Return ``[{id, url}]`` in feed order from a small Atom document."""
    text = xml.decode("utf-8", "replace") if isinstance(xml, (bytes, bytearray)) else (xml or "")
    out = []
    for match in _VIDEO_ID_XML_RE.finditer(text):
        video_id = match.group(1).strip()
        if _VIDEO_ID_RE.fullmatch(video_id):
            out.append({"id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}"})
    return out


def _validate_feed_url(url: str) -> str:
    parsed = urlparse((url or "").strip())
    if parsed.scheme != "https" or (parsed.hostname or "").lower() != "www.youtube.com":
        raise ValueError("feed URL must use https://www.youtube.com")
    if parsed.path != "/feeds/videos.xml":
        raise ValueError("unexpected YouTube feed path")
    qs = parse_qs(parsed.query, strict_parsing=True)
    playlist_values = qs.get("playlist_id", [])
    channel_values = qs.get("channel_id", [])
    if playlist_values and not channel_values:
        if len(playlist_values) != 1 or not _PLAYLIST_RE.fullmatch(playlist_values[0]):
            raise ValueError("invalid playlist_id in feed URL")
    elif channel_values and not playlist_values:
        if len(channel_values) != 1 or not _UC_RE.fullmatch(channel_values[0]):
            raise ValueError("invalid channel_id in feed URL")
    else:
        raise ValueError("feed URL must contain valid playlist_id or channel_id")
    return parsed.geturl()


def fetch_feed(url: str, timeout: float = 15.0) -> bytes:
    """Fetch a validated YouTube feed without redirects or unbounded reads."""
    safe_url = _validate_feed_url(url)
    req = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    try:
        with _FEED_OPENER.open(req, timeout=timeout) as response:  # nosec B310: exact HTTPS host/path validated above
            data = response.read(MAX_FEED_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"YouTube feed HTTP {exc.code}") from exc
    if len(data) > MAX_FEED_BYTES:
        raise ValueError("YouTube feed exceeded size cap")
    return data


def resolve_channel_id(channel_input: str) -> str:
    """Resolve an @handle / channel URL / UC id to a canonical UC id.

    The yt-dlp metadata lookup is explicitly bounded: without socket/retry
    limits a broken resolver or half-open connection could pin a monitor worker
    forever and prevent clean shutdown.
    """
    channel = (channel_input or "").strip()
    if _UC_RE.fullmatch(channel):
        return channel
    lowered = channel.lower()
    if lowered.startswith("http://"):
        raise ValueError("YouTube channel URLs must use HTTPS")
    if lowered.startswith(("youtube.com/", "www.youtube.com/")):
        url = f"https://{channel}"
    elif lowered.startswith("https://"):
        url = channel
    else:
        url = f"https://www.youtube.com/{channel.lstrip('/')}"
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid YouTube channel URL port") from exc
    path = parsed.path.rstrip("/")
    allowed_path = (
        path.startswith("/@")
        or path.startswith("/channel/UC")
        or path.startswith("/c/")
        or path.startswith("/user/")
    )
    if (
        parsed.scheme != "https"
        or host not in {"youtube.com", "www.youtube.com"}
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not allowed_path
    ):
        raise ValueError("YouTube channel must use an official channel/handle URL")
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
        "socket_timeout": CHANNEL_RESOLVE_TIMEOUT,
        "retries": 2,
        "extractor_retries": 2,
        "cachedir": False,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False, process=False) or {}
    channel_id = info.get("channel_id") or info.get("uploader_id") or info.get("id")
    if not channel_id or not _UC_RE.fullmatch(str(channel_id)):
        raise ValueError(f"could not resolve YouTube channel id from {channel_input!r}")
    return str(channel_id)


def canonical_youtube_live_url(channel_input: str) -> str:
    """Convert an @handle, UC channel id, or channel URL to its official /live URL."""
    raw = (channel_input or "").strip()
    if not raw:
        raise ValueError("channel is required")
    if raw.startswith("@"):
        return f"https://www.youtube.com/{raw}/live"
    if _UC_RE.fullmatch(raw):
        return f"https://www.youtube.com/channel/{raw}/live"
    lowered = raw.lower()
    if lowered.startswith(("http://", "https://")):
        url = raw
    elif lowered.startswith(("youtube.com/", "www.youtube.com/")):
        url = f"https://{raw}"
    else:
        url = f"https://www.youtube.com/{raw.lstrip('/')}"
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip("/")
    if clean_path.endswith("/live"):
        return f"https://www.youtube.com{clean_path}"
    return f"https://www.youtube.com{clean_path}/live"


def check_youtube_live(channel_input: str, timeout: float = 15.0) -> tuple[bool, str | None, datetime | None]:
    """Check whether a YouTube channel is currently live.

    Returns (is_live, stream_or_live_url, stream_started_at).
    """
    try:
        live_url = canonical_youtube_live_url(channel_input)
    except Exception as exc:
        logger.debug("canonical_youtube_live_url failed for %s: %s", channel_input, exc)
        return False, None, None

    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": True,
        "socket_timeout": timeout,
        "retries": 2,
        "extractor_retries": 2,
        "cachedir": False,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(live_url, download=False, process=False) or {}
    except Exception as exc:
        logger.debug("YouTube live probe error for %s: %s", live_url, exc)
        return False, None, None

    is_live = bool(info.get("is_live") or info.get("live_status") == "is_live")
    if not is_live:
        return False, None, None

    started_at = None
    ts = info.get("release_timestamp") or info.get("timestamp")
    if ts and isinstance(ts, (int, float)) and ts > 0:
        try:
            started_at = datetime.fromtimestamp(ts, tz=timezone.utc)
        except (ValueError, OSError):
            started_at = None

    playback_url = info.get("url") or live_url
    return True, playback_url, started_at


def get_youtube_live_stream_url(live_url: str, quality: str = "best") -> str | None:
    """Extract direct HLS/stream playback URL using yt-dlp."""
    from yt_dlp import YoutubeDL

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": quality or "best",
        "socket_timeout": 20,
        "retries": 2,
        "cachedir": False,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(live_url, download=False) or {}
            url = info.get("url")
            if url:
                return str(url)
    except Exception as exc:
        logger.warning("Failed to extract direct YouTube stream URL for %s: %s", live_url, exc)
    return None

