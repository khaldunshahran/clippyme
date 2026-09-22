"""FastAPI router for Trend Radar & AI Content Sourcing.

Provides:
- GET  /api/trends: fetch active discovered US trending topics
- POST /api/trends/scan: trigger immediate on-demand trend research
- POST /api/trends/clip: 1-Click Clip endpoint to enqueue a trending video into ClippyMe
- GET  /api/trends/config: get trend radar settings (interval, categories)
- POST /api/trends/config: update trend radar settings
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from clippyme.api.auth import AuthUser, get_current_user
from clippyme.api.security import require_trusted_config_request
from clippyme.domain.errors import ValidationError
from clippyme.domain.job_results import build_main_cmd
from clippyme.domain.job_submission import submit_job
from clippyme.domain.quota_service import check_user_quota
from clippyme.domain.trend_discovery import (
    DEFAULT_CATEGORIES,
    load_trend_radar,
    mark_trend_clipped,
    run_trend_discovery,
    save_trend_radar,
)
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme")

TREND_CONFIG_FILE = os.path.join("data", "trend_config.json")
DEFAULT_TREND_CONFIG = {
    "interval_hours": 3,
    "auto_scan": True,
    "categories": DEFAULT_CATEGORIES,
}


def load_trend_config() -> Dict[str, Any]:
    """Load persisted trend configuration or return defaults."""
    if not os.path.exists(TREND_CONFIG_FILE):
        return dict(DEFAULT_TREND_CONFIG)
    try:
        with open(TREND_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {**DEFAULT_TREND_CONFIG, **(data if isinstance(data, dict) else {})}
    except Exception as exc:
        logger.warning("Failed to load trend config: %s", exc)
        return dict(DEFAULT_TREND_CONFIG)


def save_trend_config(config: Dict[str, Any]) -> bool:
    """Save trend configuration to disk."""
    try:
        os.makedirs("data", mode=0o700, exist_ok=True)
        with open(TREND_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as exc:
        logger.warning("Failed to save trend config: %s", exc)
        return False


class TrendClipRequest(BaseModel):
    video_url: str
    topic_id: Optional[str] = None
    channel_id: Optional[str] = None
    instructions: Optional[str] = ""
    preset_id: Optional[str] = "viral"
    reframe_mode: Optional[str] = "auto"
    min_duration: Optional[int] = 15
    max_duration: Optional[int] = 60
    min_clips: Optional[int] = 5
    max_clips: Optional[int] = 15


class TrendConfigUpdateRequest(BaseModel):
    interval_hours: Optional[int] = Field(None, ge=1, le=24)
    auto_scan: Optional[bool] = None
    categories: Optional[List[str]] = None


def make_trend_router(
    *,
    jobs: dict,
    job_queue: asyncio.Queue,
    output_dir: str,
    on_change=None,
) -> APIRouter:
    """Create and return the trend router with injected job submission state."""
    router = APIRouter(prefix="/api/trends", tags=["trends"])

    @router.get("")
    async def get_trends(
        category: Optional[str] = Query(None, description="Category filter (all, politics, breaking_world, entertainment)"),
        include_clipped: bool = Query(True, description="Whether to include topics that have already been clipped"),
    ):
        """Fetch latest discovered trending topics."""
        data = await asyncio.to_thread(load_trend_radar)
        topics = data.get("topics", [])

        if category and category.lower() != "all":
            cat_lower = category.lower()
            topics = [t for t in topics if t.get("category", "").lower() == cat_lower]

        if not include_clipped:
            topics = [t for t in topics if not t.get("clipped")]

        return {
            "last_scanned": data.get("last_scanned"),
            "topics": topics,
            "total": len(topics),
        }

    @router.post("/scan")
    async def trigger_scan(
        request: Request,
        x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    ):
        """Trigger an on-demand trend research scan."""
        require_trusted_config_request(request)

        persisted = await asyncio.to_thread(load_persistent_config)
        api_key = (
            x_gemini_key
            or (persisted.get("GEMINI_API_KEY") if persisted else None)
            or os.environ.get("GEMINI_API_KEY")
        )

        cfg = load_trend_config()
        categories = cfg.get("categories", DEFAULT_CATEGORIES)

        result = await asyncio.to_thread(
            run_trend_discovery,
            api_key=api_key,
            categories=categories,
        )

        return {
            "status": "ok",
            "last_scanned": result.get("last_scanned"),
            "topics_count": len(result.get("topics", [])),
            "topics": result.get("topics", []),
        }

    @router.post("/clip")
    async def clip_trending_video(
        request: Request,
        payload: TrendClipRequest,
        x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    ):
        """1-Click Clip action: enqueues a trending video into the ClippyMe pipeline."""
        require_trusted_config_request(request)

        url = payload.video_url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="Missing video_url")

        persisted = await asyncio.to_thread(load_persistent_config)
        api_key = (
            x_gemini_key
            or (persisted.get("GEMINI_API_KEY") if persisted else None)
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            raise HTTPException(status_code=400, detail="Missing Gemini API Key. Please configure in Settings.")

        # Authenticate user if auth is active
        user: AuthUser = get_current_user(request)
        allowed, reason = check_user_quota(user)
        if not allowed:
            raise HTTPException(status_code=402, detail=reason)

        job_id = str(uuid.uuid4())
        job_output_dir = os.path.join(output_dir, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        env = os.environ.copy()
        env["GEMINI_API_KEY"] = api_key

        from clippyme.domain.channel_service import get_channel, match_channel_for_category
        resolved_channel = None
        if payload.channel_id:
            resolved_channel = await asyncio.to_thread(get_channel, payload.channel_id)
        if not resolved_channel and payload.topic_id:
            radar_data = await asyncio.to_thread(load_trend_radar)
            for t in radar_data.get("topics", []):
                if t.get("id") == payload.topic_id:
                    resolved_channel = await asyncio.to_thread(match_channel_for_category, t.get("category"))
                    break

        reframe_mode = payload.reframe_mode or (resolved_channel.get("reframe_mode") if resolved_channel else "auto")

        try:
            cmd = build_main_cmd(
                url=url,
                input_path=None,
                output_dir=job_output_dir,
                instructions=payload.instructions or "",
                reframe_mode=reframe_mode,
                cookies_path=os.path.join("data", "cookies.txt"),
                min_duration=payload.min_duration,
                max_duration=payload.max_duration,
                min_clips=payload.min_clips,
                max_clips=payload.max_clips,
                clip_type="viral",
            )
        except ValueError as exc:
            await asyncio.to_thread(shutil_rmtree, job_output_dir)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if resolved_channel:
            env["CLIPPYME_CHANNEL_ID"] = resolved_channel["id"]
            env["CLIPPYME_CHANNEL_NAME"] = resolved_channel["name"]

        await submit_job(
            jobs=jobs,
            job_queue=job_queue,
            job_id=job_id,
            cmd=cmd,
            env=env,
            job_output_dir=job_output_dir,
            user_id=getattr(user, "email", "default_user"),
            on_change=on_change,
        )

        if payload.topic_id:
            await asyncio.to_thread(mark_trend_clipped, payload.topic_id, job_id)

        logger.info(
            "Trend 1-Click Clip submitted: job_id=%s url=%s channel=%s",
            job_id, url, resolved_channel.get("name") if resolved_channel else "none"
        )
        return {
            "status": "queued",
            "job_id": job_id,
            "video_url": url,
            "topic_id": payload.topic_id,
            "channel_id": resolved_channel.get("id") if resolved_channel else None,
            "channel_name": resolved_channel.get("name") if resolved_channel else None,
        }

    @router.get("/config")
    async def get_config():
        """Get trend configuration."""
        return load_trend_config()

    @router.post("/config")
    async def update_config(request: Request, payload: TrendConfigUpdateRequest):
        """Update trend configuration."""
        require_trusted_config_request(request)
        current = load_trend_config()
        if payload.interval_hours is not None:
            current["interval_hours"] = payload.interval_hours
        if payload.auto_scan is not None:
            current["auto_scan"] = payload.auto_scan
        if payload.categories is not None:
            current["categories"] = [c.lower() for c in payload.categories]
        save_trend_config(current)
        return {"status": "ok", "config": current}

    return router


def shutil_rmtree(path: str):
    import shutil

    shutil.rmtree(path, ignore_errors=True)
