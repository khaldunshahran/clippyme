"""Persistent configuration loader/saver for ClippyMe."""
import contextlib
import json
import logging
import os
import tempfile
import threading

logger = logging.getLogger("clippyme")

DATA_DIR = "data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
VALID_CONFIG_KEYS = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_LITE_MODEL",
    "YOUTUBE_COOKIES",
    "HF_TOKEN",
    "DEEPGRAM_API_KEY",
    "ELEVENLABS_API_KEY",
    "TRANSCRIPTION_PROVIDER",
    "TWITCH_CLIENT_ID",
    "TWITCH_CLIENT_SECRET",
    "AUTO_CLEANUP_PUBLISHED",
    "AUTO_CLEANUP_FAILED",
)
ZERNIO_CONFIG_NAMESPACE = "zernio"
WATCHDOG_CONFIG_NAMESPACE = "watchdog"
_CONFIG_LOCK = threading.RLock()


def _read_raw_config() -> dict:
    with _CONFIG_LOCK:
        if not os.path.exists(CONFIG_FILE):
            return {}
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as file:
                data = json.load(file) or {}
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Error reading config.json: %s", exc)
            return {}


def _write_raw_config(data: dict) -> bool:
    """Atomically replace config.json with owner-only permissions.

    Writing to a sibling temporary file and then ``os.replace`` prevents a
    crash, disk-full condition, or concurrent reader from observing a
    half-truncated JSON document containing secrets.
    """
    with _CONFIG_LOCK:
        tmp_path = None
        try:
            os.makedirs(DATA_DIR, mode=0o700, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(DATA_DIR, 0o700)

            fd, tmp_path = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=DATA_DIR)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(data, file, indent=4)
                    file.flush()
                    os.fsync(file.fileno())
                with contextlib.suppress(OSError):
                    os.chmod(tmp_path, 0o600)
                os.replace(tmp_path, CONFIG_FILE)
                tmp_path = None
                with contextlib.suppress(OSError):
                    os.chmod(CONFIG_FILE, 0o600)
                # Persist the rename itself on POSIX filesystems when possible.
                try:
                    dir_fd = os.open(DATA_DIR, os.O_RDONLY)
                except OSError:
                    dir_fd = None
                if dir_fd is not None:
                    try:
                        os.fsync(dir_fd)
                    except OSError:
                        pass
                    finally:
                        os.close(dir_fd)
                return True
            except Exception:
                # fdopen owns/closes fd once entered; close only if creation
                # failed before that ownership transfer.
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
        except (OSError, TypeError, ValueError) as exc:
            logger.error("Error writing config.json: %s", exc)
            return False
        finally:
            if tmp_path:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)


def load_zernio_config() -> dict:
    raw = _read_raw_config()
    zernio = raw.get(ZERNIO_CONFIG_NAMESPACE) or {}
    if not isinstance(zernio, dict):
        zernio = {}
    accounts = zernio.get("accounts", {})
    return {
        "api_key": zernio.get("api_key", ""),
        "accounts": accounts if isinstance(accounts, dict) else {},
        "timezone": zernio.get("timezone", "Europe/Rome"),
    }


def save_zernio_config(api_key: str = None, accounts: dict = None, timezone: str = None) -> bool:
    """Merge-update Zernio settings as one locked read-modify-write."""
    with _CONFIG_LOCK:
        raw = _read_raw_config()
        current = raw.get(ZERNIO_CONFIG_NAMESPACE) or {}
        if not isinstance(current, dict):
            current = {}
        if api_key is not None:
            if api_key == "":
                current.pop("api_key", None)
            else:
                current["api_key"] = api_key
        if accounts is not None:
            merged = current.get("accounts") or {}
            if not isinstance(merged, dict):
                merged = {}
            for key, value in accounts.items():
                if value in (None, ""):
                    merged.pop(key, None)
                else:
                    merged[key] = value
            current["accounts"] = merged
        if timezone is not None:
            current["timezone"] = timezone
        raw[ZERNIO_CONFIG_NAMESPACE] = current
        return _write_raw_config(raw)


def zernio_config_status() -> dict:
    cfg = load_zernio_config()
    api_key = cfg.get("api_key", "")
    masked = f"{api_key[:6]}...{api_key[-4:]}" if api_key and len(api_key) > 10 else ""
    return {
        "configured": bool(api_key),
        "api_key_masked": masked,
        "accounts": cfg.get("accounts", {}),
        "timezone": cfg.get("timezone", "Europe/Rome"),
    }


def load_watchdog_config() -> dict:
    """Load watchdog and alert notification configuration."""
    raw = _read_raw_config()
    wd = raw.get(WATCHDOG_CONFIG_NAMESPACE) or {}
    if not isinstance(wd, dict):
        wd = {}
    return {
        "enabled": wd.get("enabled", True),
        "ai_diagnosis": wd.get("ai_diagnosis", True),
        "notify_on_failure": wd.get("notify_on_failure", True),
        "notify_on_success": wd.get("notify_on_success", False),
        "provider": wd.get("provider", "ntfy"),
        "ntfy_topic": wd.get("ntfy_topic", ""),
        "ntfy_server": wd.get("ntfy_server", "https://ntfy.sh"),
        "telegram_bot_token": wd.get("telegram_bot_token", ""),
        "telegram_chat_id": wd.get("telegram_chat_id", ""),
        "discord_webhook_url": wd.get("discord_webhook_url", ""),
        "webhook_url": wd.get("webhook_url", ""),
    }


def save_watchdog_config(updates: dict) -> bool:
    """Merge-update Watchdog alert settings."""
    with _CONFIG_LOCK:
        raw = _read_raw_config()
        current = raw.get(WATCHDOG_CONFIG_NAMESPACE) or {}
        if not isinstance(current, dict):
            current = {}
        for key in (
            "enabled", "ai_diagnosis", "notify_on_failure", "notify_on_success",
            "provider", "ntfy_topic", "ntfy_server", "telegram_bot_token",
            "telegram_chat_id", "discord_webhook_url", "webhook_url",
        ):
            if key in updates:
                val = updates[key]
                if val in (None, "") and key not in ("enabled", "ai_diagnosis", "notify_on_failure", "notify_on_success"):
                    current.pop(key, None)
                else:
                    current[key] = val
        raw[WATCHDOG_CONFIG_NAMESPACE] = current
        return _write_raw_config(raw)


def watchdog_config_status() -> dict:
    """Status dictionary with secrets safely masked for UI consumption."""
    cfg = load_watchdog_config()
    tg_token = cfg.get("telegram_bot_token", "")
    disc_url = cfg.get("discord_webhook_url", "")
    wh_url = cfg.get("webhook_url", "")

    def _mask_url(u: str) -> str:
        if not u or len(u) < 15:
            return ""
        return f"{u[:10]}...{u[-6:]}"

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "ai_diagnosis": bool(cfg.get("ai_diagnosis", True)),
        "notify_on_failure": bool(cfg.get("notify_on_failure", True)),
        "notify_on_success": bool(cfg.get("notify_on_success", False)),
        "provider": cfg.get("provider", "ntfy"),
        "ntfy_topic": cfg.get("ntfy_topic", ""),
        "ntfy_server": cfg.get("ntfy_server", "https://ntfy.sh"),
        "telegram_bot_token_masked": f"{tg_token[:6]}...{tg_token[-4:]}" if tg_token and len(tg_token) > 10 else "",
        "telegram_chat_id": cfg.get("telegram_chat_id", ""),
        "discord_webhook_masked": _mask_url(disc_url),
        "webhook_url_masked": _mask_url(wh_url),
        "has_telegram_token": bool(tg_token),
        "has_discord_webhook": bool(disc_url),
        "has_webhook_url": bool(wh_url),
    }


def _normalize_incoming_keys(data: dict) -> dict:
    if not data:
        return {}
    out = dict(data)
    if "HUGGINGFACE_TOKEN" in out and not out.get("HF_TOKEN"):
        out["HF_TOKEN"] = out.pop("HUGGINGFACE_TOKEN")
    else:
        out.pop("HUGGINGFACE_TOKEN", None)
    return out


def load_persistent_config() -> dict:
    config = {
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
        "GEMINI_MODEL": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        "GEMINI_LITE_MODEL": os.environ.get("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite"),
        "YOUTUBE_COOKIES": os.environ.get("YOUTUBE_COOKIES", ""),
        "HF_TOKEN": os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "",
        "DEEPGRAM_API_KEY": os.environ.get("DEEPGRAM_API_KEY", ""),
        "ELEVENLABS_API_KEY": os.environ.get("ELEVENLABS_API_KEY", ""),
        "TRANSCRIPTION_PROVIDER": os.environ.get("TRANSCRIPTION_PROVIDER", "whisper"),
        "TWITCH_CLIENT_ID": os.environ.get("TWITCH_CLIENT_ID", ""),
        "TWITCH_CLIENT_SECRET": os.environ.get("TWITCH_CLIENT_SECRET", ""),
    }
    raw = _read_raw_config()
    config.update({key: value for key, value in raw.items() if key in VALID_CONFIG_KEYS})
    return config


def save_persistent_config(new_config: dict) -> bool:
    """Persist core keys without racing the separate Zernio namespace."""
    with _CONFIG_LOCK:
        raw = _read_raw_config()
        normalized = _normalize_incoming_keys(new_config)
        sanitized = {
            key: normalized.get(key) for key in VALID_CONFIG_KEYS if key in normalized
        }
        for key, value in sanitized.items():
            if value in (None, ""):
                raw.pop(key, None)
            else:
                raw[key] = value
        if not _write_raw_config(raw):
            return False
        for key, value in sanitized.items():
            if value in (None, ""):
                os.environ.pop(key, None)
                if key == "HF_TOKEN":
                    os.environ.pop("HUGGINGFACE_TOKEN", None)
            else:
                os.environ[key] = str(value)
                if key == "HF_TOKEN":
                    os.environ["HUGGINGFACE_TOKEN"] = str(value)
        return True
