"""FastAPI routes for Channel Profile Management and Topic-to-Channel Routing.

Allows users to configure and manage multiple social channels within ClippyMe,
assigning distinct branding presets, niches, and destination accounts.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from clippyme.api.auth import AuthUser, get_current_user
from clippyme.api.security import require_trusted_config_request
from clippyme.domain.channel_service import (
    create_channel,
    delete_channel,
    get_channel,
    load_channels,
    match_channel_for_category,
    update_channel,
)

router = APIRouter(prefix="/api/channels", tags=["channels"])


class ChannelCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = ""
    niches: Optional[List[str]] = None
    default_preset: Optional[str] = "viral"
    banner_platform: Optional[str] = "youtube"
    banner_handle: Optional[str] = ""
    banner_y_pct: Optional[float] = 0.85
    reframe_mode: Optional[str] = "auto"
    sub_preset: Optional[str] = "hormozi_bold"
    sub_font: Optional[str] = "Montserrat-Black"
    sub_color: Optional[str] = "#FFFFFF"
    publishing_targets: Optional[Dict[str, Any]] = None


class ChannelUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    niches: Optional[List[str]] = None
    default_preset: Optional[str] = None
    banner_platform: Optional[str] = None
    banner_handle: Optional[str] = None
    banner_y_pct: Optional[float] = None
    reframe_mode: Optional[str] = None
    sub_preset: Optional[str] = None
    sub_font: Optional[str] = None
    sub_color: Optional[str] = None
    publishing_targets: Optional[Dict[str, Any]] = None


@router.get("")
async def list_channels(user: AuthUser = Depends(get_current_user)):
    """List all configured channel profiles."""
    channels = await asyncio.to_thread(load_channels)
    return {"channels": channels, "total": len(channels)}


@router.get("/match")
async def match_channel(category: Optional[str] = Query(None), user: AuthUser = Depends(get_current_user)):
    """Find the best-fitting channel profile for a given category/niche."""
    ch = await asyncio.to_thread(match_channel_for_category, category)
    if not ch:
        raise HTTPException(status_code=404, detail="No matching channel profile found")
    return {"matched": ch}


@router.get("/{channel_id}")
async def get_channel_by_id(channel_id: str, user: AuthUser = Depends(get_current_user)):
    """Get details of a specific channel profile."""
    ch = await asyncio.to_thread(get_channel, channel_id)
    if not ch:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
    return ch


@router.post("")
async def create_new_channel(request: Request, payload: ChannelCreateRequest, user: AuthUser = Depends(get_current_user)):
    """Create a new channel profile."""
    require_trusted_config_request(request)
    data = payload.model_dump(exclude_unset=True)
    created = await asyncio.to_thread(create_channel, data)
    return {"status": "created", "channel": created}


@router.put("/{channel_id}")
async def update_existing_channel(request: Request, channel_id: str, payload: ChannelUpdateRequest, user: AuthUser = Depends(get_current_user)):
    """Update an existing channel profile."""
    require_trusted_config_request(request)
    patch = payload.model_dump(exclude_unset=True)
    updated = await asyncio.to_thread(update_channel, channel_id, patch)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
    return {"status": "updated", "channel": updated}


@router.delete("/{channel_id}")
async def delete_existing_channel(request: Request, channel_id: str, user: AuthUser = Depends(get_current_user)):
    """Delete a channel profile."""
    require_trusted_config_request(request)
    success = await asyncio.to_thread(delete_channel, channel_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
    return {"status": "deleted", "id": channel_id}
