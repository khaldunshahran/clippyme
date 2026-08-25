"""HTTP API routes for Generative AI Shorts / UGC Video Creation."""
import asyncio
import logging
import os
from typing import Optional, Dict, Any
from fastapi import APIRouter, Header
from pydantic import BaseModel

from clippyme.domain.errors import ValidationError, ClippyMeError
from clippyme.ugc.research import scrape_product_website, research_product_online
from clippyme.ugc.scripting import generate_ugc_scripts
from clippyme.storage.config_store import load_persistent_config

logger = logging.getLogger("clippyme.ugc")
router = APIRouter()


class ResearchRequest(BaseModel):
    url_or_description: str


class ScriptRequest(BaseModel):
    research_data: Dict[str, Any]
    target_language: str = "en"


@router.post("/api/ugc/research")
async def research_product(
    body: ResearchRequest,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
):
    """Scrape and research a product URL or description."""
    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    target = body.url_or_description.strip()
    if not target:
        raise ValidationError("URL or description is required.")

    try:
        if target.startswith("http://") or target.startswith("https://"):
            try:
                scraped = await asyncio.to_thread(scrape_product_website, target)
                target = f"{target}\nTitle: {scraped.get('title')}\nDescription: {scraped.get('description')}\nContent: {scraped.get('body_text', '')[:3000]}"
            except Exception as se:
                logger.warning("UGC scraping fallback warning: %s", se)

        return await asyncio.to_thread(research_product_online, target, api_key)
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("UGC Research failed")
        raise ClippyMeError(f"Research failed: {e}", status_code=500)


@router.post("/api/ugc/scripts")
async def generate_scripts_endpoint(
    body: ScriptRequest,
    x_gemini_key: Optional[str] = Header(None, alias="x-gemini-key"),
):
    """Generate viral short marketing scripts from research data."""
    cfg = load_persistent_config()
    api_key = x_gemini_key or cfg.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValidationError("Gemini API key is required.")

    try:
        scripts = await asyncio.to_thread(
            generate_ugc_scripts,
            body.research_data,
            api_key,
            body.target_language,
        )
        return {"scripts": scripts}
    except ClippyMeError:
        raise
    except ValueError as ve:
        raise ValidationError(str(ve))
    except Exception as e:
        logger.exception("UGC Script generation failed")
        raise ClippyMeError(f"Script generation failed: {e}", status_code=500)

