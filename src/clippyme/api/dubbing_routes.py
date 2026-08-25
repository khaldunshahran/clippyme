"""HTTP API routes for AI Video Translation & Dubbing (ElevenLabs)."""
import asyncio
import logging
import os
from typing import Optional
from fastapi import APIRouter, Header
from pydantic import BaseModel

from clippyme.domain.clip_locks import clip_lock
from clippyme.domain.clip_resolve import resolve_clip
from clippyme.domain.errors import ValidationError, ClippyMeError
from clippyme.integrations.dubbing import SUPPORTED_LANGUAGES, dub_video_sync
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme.dubbing")
router = APIRouter()


class DubbingRequest(BaseModel):
    target_language: str
    source_language: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None


OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")


@router.get("/api/dubbing/languages")
async def list_dubbing_languages():
    """List supported target languages for ElevenLabs dubbing."""
    return {"languages": SUPPORTED_LANGUAGES}


@router.post("/api/dubbing/{job_id}/{clip_index}")
async def dub_clip(
    job_id: str,
    clip_index: int,
    body: DubbingRequest,
    x_elevenlabs_key: Optional[str] = Header(None, alias="x-elevenlabs-key"),
):
    """Dub a single clip into a target language. Outputs dubbed_<lang>_<clip_name>.mp4."""
    resolved = await asyncio.to_thread(resolve_clip, job_id, clip_index, output_root=OUTPUT_DIR)

    cfg = load_persistent_config()
    api_key = (
        body.elevenlabs_api_key
        or x_elevenlabs_key
        or cfg.get("ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_API_KEY")
    )
    if not api_key:
        raise ValidationError(
            "ElevenLabs API key is required for dubbing. Provide it in settings or request body."
        )

    target_lang = body.target_language.lower().strip()
    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValidationError(f"Unsupported target language '{target_lang}'.")

    clip_base = os.path.splitext(resolved.clip_filename)[0]
    out_filename = f"dubbed_{target_lang}_{clip_base}.mp4"
    out_path = os.path.join(resolved.job_dir, out_filename)

    async with clip_lock(resolved.job_dir, clip_index):
        try:
            await asyncio.to_thread(
                dub_video_sync,
                video_path=resolved.clip_path,
                target_language=target_lang,
                output_path=out_path,
                api_key=api_key,
                source_language=body.source_language,
            )
        except ClippyMeError:
            raise
        except ValueError as ve:
            raise ValidationError(str(ve))
        except Exception as e:
            logger.exception("Dubbing failed")
            raise ClippyMeError(f"Dubbing failed: {e}", status_code=500)

    return {
        "success": True,
        "job_id": job_id,
        "clip_index": clip_index,
        "target_language": target_lang,
        "dubbed_filename": out_filename,
        "dubbed_url": f"/videos/{job_id}/{out_filename}",
    }

