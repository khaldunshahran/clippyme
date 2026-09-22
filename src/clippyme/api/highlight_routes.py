"""HTTP API routes for AI Highlights & Supercut Studio."""
import asyncio
import logging
import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from clippyme.domain.errors import ValidationError, NotFoundError, ClippyMeError
from clippyme.domain.highlight_service import (
    plan_highlights_sync,
    render_highlight_reel_sync,
    generate_multi_tier_highlights_sync,
    reprocess_highlight_reel_sync,
    delete_highlight_reel_sync,
    load_or_create_job_metadata,
)
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme.highlights")
router = APIRouter()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")


class HighlightPlanRequest(BaseModel):
    target_duration: int = Field(60, ge=15, le=720)
    content_mode: Optional[str] = "podcast"
    output_style: Optional[str] = "recap"
    theme: Optional[str] = None
    merge_gap_seconds: float = Field(1.5, ge=0.0, le=10.0)
    model_name: Optional[str] = None


class HighlightGenerateAllRequest(BaseModel):
    aspect: str = Field("9:16", pattern=r"^(9:16|16:9|1:1|4:5)$")
    reframe_mode: str = "blur_pad"
    subtitles: Optional[Dict[str, Any]] = None
    grade_preset: str = "none"
    content_mode: Optional[str] = "podcast"
    output_style: Optional[str] = "recap"
    theme: Optional[str] = None
    merge_gap_seconds: float = Field(1.5, ge=0.0, le=10.0)
    model_name: Optional[str] = None


class HighlightApplyEditRequest(BaseModel):
    aspect: Optional[str] = None
    reframe_mode: Optional[str] = None
    subtitles: Optional[Dict[str, Any]] = None
    hook: Optional[Dict[str, Any]] = None
    grade_preset: Optional[str] = None
    cuts: Optional[List[Dict[str, Any]]] = None
    title: Optional[str] = None
    content_mode: Optional[str] = None
    output_style: Optional[str] = None
    theme: Optional[str] = None
    merge_gap_seconds: Optional[float] = None


class HighlightRenderRequest(BaseModel):
    target_duration: int = Field(60, ge=15, le=720)
    title: Optional[str] = None
    cuts: Optional[List[Dict[str, Any]]] = None
    aspect: str = Field("9:16", pattern=r"^(9:16|16:9|1:1|4:5)$")
    reframe_mode: str = "blur_pad"
    subtitles: Optional[Dict[str, Any]] = None
    hook: Optional[Dict[str, Any]] = None
    logo: Optional[Dict[str, Any]] = None
    grade_preset: str = "none"
    content_mode: Optional[str] = "podcast"
    output_style: Optional[str] = "recap"
    theme: Optional[str] = None
    merge_gap_seconds: float = Field(1.5, ge=0.0, le=10.0)
    model_name: Optional[str] = None


@router.post("/api/highlights/{job_id}/plan")
async def plan_highlights(
    job_id: str,
    body: HighlightPlanRequest,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
):
    """Stage 1: Generate an AI narrative timeline plan for a highlight reel."""
    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required to analyze highlights.")

    try:
        return await asyncio.to_thread(
            plan_highlights_sync,
            job_id=job_id,
            target_duration=body.target_duration,
            content_mode=body.content_mode,
            output_style=body.output_style,
            theme=body.theme,
            merge_gap_seconds=body.merge_gap_seconds,
            output_root=OUTPUT_DIR,
            api_key=api_key,
            model_name=body.model_name,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Highlight plan generation failed")
        raise ClippyMeError(f"Highlight planning failed: {e}", status_code=500)


@router.post("/api/highlights/{job_id}/render")
@router.post("/api/highlights/{job_id}")
async def render_highlight_reel(
    job_id: str,
    body: HighlightRenderRequest,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
):
    """Stage 2: Full multi-layer rendering of a highlight reel with aspect ratio sizing."""
    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    try:
        return await asyncio.to_thread(
            render_highlight_reel_sync,
            job_id=job_id,
            target_duration=body.target_duration,
            cuts=body.cuts,
            aspect=body.aspect,
            reframe_mode=body.reframe_mode,
            subtitles=body.subtitles,
            hook=body.hook,
            logo=body.logo,
            grade_preset=body.grade_preset,
            content_mode=body.content_mode,
            output_style=body.output_style,
            theme=body.theme,
            merge_gap_seconds=body.merge_gap_seconds,
            output_root=OUTPUT_DIR,
            api_key=api_key,
            model_name=body.model_name,
            title=body.title,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Highlight rendering failed")
        raise ClippyMeError(f"Highlight render failed: {e}", status_code=500)


@router.post("/api/highlights/{job_id}/generate-all")
async def generate_all_highlights(
    job_id: str,
    body: HighlightGenerateAllRequest,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
):
    """Analyze video and automatically generate multi-tier highlight reels (<60s, ~120s, 180s-720s)."""
    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

    try:
        return await asyncio.to_thread(
            generate_multi_tier_highlights_sync,
            job_id=job_id,
            aspect=body.aspect,
            reframe_mode=body.reframe_mode,
            subtitles=body.subtitles,
            grade_preset=body.grade_preset,
            content_mode=body.content_mode,
            output_style=body.output_style,
            theme=body.theme,
            merge_gap_seconds=body.merge_gap_seconds,
            output_root=OUTPUT_DIR,
            api_key=api_key,
            model_name=body.model_name,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Multi-tier highlight generation failed")
        raise ClippyMeError(f"Multi-tier highlight generation failed: {e}", status_code=500)


@router.post("/api/highlights/{job_id}/{highlight_id}/apply-edit")
async def apply_edit_highlight(
    job_id: str,
    highlight_id: str,
    body: HighlightApplyEditRequest,
):
    """Reprocess a specific highlight reel with updated edit parameters (reframe, subtitles, grade, hook, trim)."""
    try:
        return await asyncio.to_thread(
            reprocess_highlight_reel_sync,
            job_id=job_id,
            highlight_id=highlight_id,
            edit_params=body.model_dump(exclude_unset=True),
            output_root=OUTPUT_DIR,
        )
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("Highlight apply-edit failed")
        raise ClippyMeError(f"Highlight apply-edit failed: {e}", status_code=500)


@router.get("/api/highlights/{job_id}")
async def get_highlights(job_id: str):
    """List all generated highlight reels for a job."""
    try:
        _, data = await asyncio.to_thread(load_or_create_job_metadata, job_id, OUTPUT_DIR)
        return {"highlights": data.get("highlights", [])}
    except (NotFoundError, FileNotFoundError):
        raise NotFoundError(f"Job metadata not found for {job_id}")
    except Exception as e:
        logger.exception("Failed to load highlights")
        raise ClippyMeError(f"Failed to load highlights: {e}", status_code=500)


@router.delete("/api/highlights/{job_id}/{filename}")
async def delete_highlight(job_id: str, filename: str):
    """Delete a generated highlight reel."""
    try:
        return await asyncio.to_thread(
            delete_highlight_reel_sync,
            job_id=job_id,
            filename=filename,
            output_root=OUTPUT_DIR,
        )
    except Exception as e:
        logger.exception("Failed to delete highlight")
        raise ClippyMeError(f"Failed to delete highlight: {e}", status_code=500)
