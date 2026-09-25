"""HTTP API routes for YouTube Studio: Viral Titles, Chapters, and AI Thumbnails."""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from clippyme.api.auth import AuthUser, get_current_user
from clippyme.api.security import enforce_rate_limit
from clippyme.domain.quota_service import check_user_quota

from clippyme.domain.errors import ValidationError, ClippyMeError
from clippyme.studio.youtube_studio import (
    generate_viral_titles,
    refine_viral_titles,
    generate_youtube_chapters,
    generate_youtube_thumbnail,
)
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme.studio")
router = APIRouter()


class TitlesRequest(BaseModel):
    transcript_text: Optional[str] = None
    language: str = "en"
    model_name: Optional[str] = None


class RefineTitlesRequest(BaseModel):
    context: str
    user_instruction: str
    history: Optional[List[Dict[str, str]]] = None


class ChaptersRequest(BaseModel):
    segments: List[Dict[str, Any]]


class ThumbnailRequest(BaseModel):
    title: str
    video_context: Optional[str] = ""
    extra_prompt: Optional[str] = ""
    aspect_ratio: Optional[str] = "16:9"
    model: Optional[str] = None


@router.post("/api/studio/titles")
async def get_viral_titles(
    body: TitlesRequest,
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
    user: AuthUser = Depends(get_current_user),
):
    """Generate 10 viral YouTube title suggestions from transcript text."""
    enforce_rate_limit(request, "studio", capacity=30, refill_per_sec=30/60)
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    if not body.transcript_text:
        raise ValidationError("Transcript text is required.")

    try:
        return await asyncio.to_thread(
            generate_viral_titles,
            transcript_text=body.transcript_text,
            api_key=api_key,
            language=body.language,
            model_name=body.model_name,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Title generation failed")
        raise ClippyMeError(f"Title generation failed: {e}", status_code=500)


@router.post("/api/studio/titles/refine")
async def refine_titles_endpoint(
    body: RefineTitlesRequest,
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
    user: AuthUser = Depends(get_current_user),
):
    """Refine titles via chat instruction."""
    enforce_rate_limit(request, "studio", capacity=30, refill_per_sec=30/60)
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    try:
        return await asyncio.to_thread(
            refine_viral_titles,
            context=body.context,
            user_instruction=body.user_instruction,
            api_key=api_key,
            history=body.history,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Title refinement failed")
        raise ClippyMeError(f"Title refinement failed: {e}", status_code=500)


@router.post("/api/studio/chapters")
async def generate_chapters_endpoint(
    body: ChaptersRequest,
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
    user: AuthUser = Depends(get_current_user),
):
    """Generate timestamped chapters and SEO description from transcript segments."""
    enforce_rate_limit(request, "studio", capacity=30, refill_per_sec=30/60)
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    try:
        return await asyncio.to_thread(
            generate_youtube_chapters,
            transcript_segments=body.segments,
            api_key=api_key,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Chapter generation failed")
        raise ClippyMeError(f"Chapter generation failed: {e}", status_code=500)


@router.post("/api/studio/thumbnail")
async def generate_thumbnail_endpoint(
    body: ThumbnailRequest,
    request: Request,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
    user: AuthUser = Depends(get_current_user),
):
    """Generate an AI YouTube thumbnail or social cover."""
    enforce_rate_limit(request, "studio", capacity=30, refill_per_sec=30/60)
    allowed, reason = check_user_quota(user)
    if not allowed:
        raise HTTPException(status_code=402, detail=reason)

    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    out_dir = os.path.join("output", "thumbnails")
    try:
        path = await asyncio.to_thread(
            generate_youtube_thumbnail,
            title=body.title,
            output_dir=out_dir,
            api_key=api_key,
            video_context=body.video_context or "",
            extra_prompt=body.extra_prompt or "",
            aspect_ratio=body.aspect_ratio or "16:9",
            model=body.model,
        )
        return {"success": True, "thumbnail_path": path}
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Thumbnail generation failed")
        raise ClippyMeError(f"Thumbnail generation failed: {e}", status_code=500)

