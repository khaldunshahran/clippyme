"""ElevenLabs Video Translation & Dubbing Integration for ClippyMe.

Translates and re-voices video clips into 30+ languages while preserving the original
voice characteristics and timing.
"""
import os
import time
import httpx
from typing import Optional, Dict, Any

ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "id": "Indonesian",
    "fil": "Filipino",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "ta": "Tamil",
}

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class DubbingError(Exception):
    """Raised for ElevenLabs Dubbing failures."""


def create_dubbing_project(
    video_path: str,
    target_language: str,
    api_key: str,
    source_language: Optional[str] = None,
    num_speakers: int = 0,
) -> Dict[str, Any]:
    """Create a new dubbing project with ElevenLabs."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    url = f"{ELEVENLABS_API_BASE}/dubbing"
    headers = {"xi-api-key": api_key}

    data = {
        "target_lang": target_language,
        "num_speakers": str(num_speakers),
        "watermark": "false",
    }
    if source_language:
        data["source_lang"] = source_language

    with open(video_path, "rb") as f:
        files = {"file": (os.path.basename(video_path), f, "video/mp4")}
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            if resp.status_code != 200:
                raise DubbingError(f"Failed to create dubbing project: {resp.status_code} - {resp.text}")
            return resp.json()


def get_dubbing_status(dubbing_id: str, api_key: str) -> Dict[str, Any]:
    """Get the status of a dubbing project."""
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}"
    headers = {"xi-api-key": api_key}

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            raise DubbingError(f"Failed to get dubbing status: {resp.status_code} - {resp.text}")
        return resp.json()


def download_dubbed_file(
    dubbing_id: str,
    target_language: str,
    output_path: str,
    api_key: str,
) -> str:
    """Download the dubbed audio or video file."""
    url = f"{ELEVENLABS_API_BASE}/dubbing/{dubbing_id}/audio/{target_language}"
    headers = {"xi-api-key": api_key}

    with httpx.Client(timeout=120.0) as client:
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            raise DubbingError(f"Failed to download dubbed file: {resp.status_code} - {resp.text}")

        with open(output_path, "wb") as f:
            f.write(resp.content)

    return output_path


def dub_video_sync(
    video_path: str,
    target_language: str,
    output_path: str,
    api_key: str,
    source_language: Optional[str] = None,
    poll_interval: int = 5,
    max_wait_sec: int = 600,
) -> str:
    """Synchronously create, poll, and download a dubbed video."""
    print(f"🌐 [Dubbing] Starting dubbing for {video_path} into '{target_language}'...")
    project = create_dubbing_project(
        video_path=video_path,
        target_language=target_language,
        api_key=api_key,
        source_language=source_language,
    )
    dubbing_id = project["dubbing_id"]

    start = time.time()
    while time.time() - start < max_wait_sec:
        status_info = get_dubbing_status(dubbing_id, api_key)
        state = status_info.get("status")
        if state == "dubbed":
            print("🌐 [Dubbing] Dubbing complete. Downloading dubbed video...")
            return download_dubbed_file(dubbing_id, target_language, output_path, api_key)
        elif state == "failed":
            err = status_info.get("error", "Unknown dubbing error")
            raise DubbingError(f"Dubbing failed: {err}")
        time.sleep(poll_interval)

    raise TimeoutError(f"Dubbing timed out after {max_wait_sec} seconds")
