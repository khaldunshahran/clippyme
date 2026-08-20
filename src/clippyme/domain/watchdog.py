"""AI Watchdog & Error Diagnostics Engine for ClippyMe.

Monitors pipeline executions, FFmpeg rendering, and API publish flows. When an
error occurs, the watchdog:
1. Intercepts the stack trace and recent log records.
2. Prompts Gemini AI for a concise, plain-English root cause diagnosis.
3. Dispatches instant push notifications (via ntfy.sh, Telegram, Discord, or Webhook).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

from clippyme.integrations.notifier import (
    send_discord_webhook,
    send_generic_webhook,
    send_ntfy,
    send_telegram,
)
from clippyme.storage.config_store import load_watchdog_config

logger = logging.getLogger("clippyme")


def _get_gemini_api_key() -> str | None:
    """Retrieve Gemini API key from persistent config or environment."""
    from clippyme.storage.config_store import load_persistent_config
    cfg = load_persistent_config()
    return cfg.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")


async def diagnose_failure(
    error_text: str,
    log_tail: list[str] | str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ask Gemini AI to diagnose an error and provide a human-readable explanation."""
    logs_str = "\n".join(log_tail[-25:]) if isinstance(log_tail, list) else str(log_tail)[-2000:]
    ctx_str = json.dumps(context or {}, ensure_ascii=False)

    api_key = _get_gemini_api_key()
    if not api_key:
        return {
            "summary": error_text[:300] if error_text else "Pipeline error encountered.",
            "root_cause": "Detailed AI diagnosis unavailable (no Gemini API key configured).",
            "action": "Check the terminal logs or configure a Gemini API key in Settings.",
            "auto_recoverable": False,
        }

    prompt = f"""You are the ClippyMe AI Watchdog Doctor. Analyze the following video clipping/rendering error and return a JSON object with actionable insights.

ERROR MESSAGE:
{error_text}

CONTEXT:
{ctx_str}

RECENT LOGS:
{logs_str}

Respond with ONLY a valid JSON object matching this schema:
{{
  "summary": "1-2 sentence plain-English summary of what broke",
  "root_cause": "The specific technical reason (e.g. missing file, subtitle font issue, invalid aspect ratio, API token expired)",
  "action": "Actionable recommendation for the user or system",
  "auto_recoverable": true or false
}}
"""

    def _call_gemini() -> dict[str, Any]:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            raw_text = (response.text or "").strip()
            return json.loads(raw_text)
        except Exception as exc:
            logger.warning("Gemini watchdog diagnosis call failed: %s", exc)
            return {
                "summary": error_text[:300],
                "root_cause": f"Diagnosis failed: {exc}",
                "action": "Inspect logs manually in the ClippyMe dashboard.",
                "auto_recoverable": False,
            }

    return await asyncio.to_thread(_call_gemini)


async def dispatch_alert(
    *,
    title: str,
    message: str,
    diagnosis: dict[str, Any] | None = None,
    is_error: bool = True,
    url: str | None = None,
    cfg_override: dict[str, Any] | None = None,
) -> bool:
    """Send an alert to the user's configured notification channels."""
    cfg = cfg_override or load_watchdog_config()
    if not cfg.get("enabled", True) and not cfg_override:
        return False

    provider = (cfg.get("provider") or "ntfy").lower()
    sent = False

    # Format text with diagnosis if available
    body_lines = [message]
    if diagnosis:
        summary = diagnosis.get("summary")
        cause = diagnosis.get("root_cause")
        action = diagnosis.get("action")
        if summary:
            body_lines.append(f"\n🧠 AI Diagnosis: {summary}")
        if cause:
            body_lines.append(f"🔍 Root Cause: {cause}")
        if action:
            body_lines.append(f"💡 Recommended Action: {action}")

    full_text = "\n".join(body_lines)

    if provider == "ntfy":
        topic = cfg.get("ntfy_topic")
        if topic:
            tags = ["warning", "rotating_light"] if is_error else ["white_check_mark", "tada"]
            priority = "high" if is_error else "default"
            server = cfg.get("ntfy_server") or "https://ntfy.sh"
            sent = await send_ntfy(
                topic,
                title=f"ClippyMe: {title}",
                message=full_text,
                priority=priority,
                tags=tags,
                click_url=url,
                server=server,
            )

    elif provider == "telegram":
        token = cfg.get("telegram_bot_token")
        chat_id = cfg.get("telegram_chat_id")
        if token and chat_id:
            md_text = f"*{'🚨 ERROR' if is_error else '✅ SUCCESS'}: {title}*\n\n{full_text}"
            if url:
                md_text += f"\n\n[Open Dashboard]({url})"
            sent = await send_telegram(token, chat_id, text=md_text)

    elif provider == "discord":
        webhook_url = cfg.get("discord_webhook_url")
        if webhook_url:
            fields = []
            if diagnosis:
                if diagnosis.get("summary"):
                    fields.append({"name": "AI Summary", "value": diagnosis["summary"][:1024], "inline": False})
                if diagnosis.get("root_cause"):
                    fields.append({"name": "Root Cause", "value": diagnosis["root_cause"][:1024], "inline": False})
                if diagnosis.get("action"):
                    fields.append({"name": "Action", "value": diagnosis["action"][:1024], "inline": False})
            sent = await send_discord_webhook(
                webhook_url,
                title=f"{'🚨' if is_error else '✅'} ClippyMe: {title}",
                description=message,
                color=0xE02424 if is_error else 0x31C48D,
                fields=fields if fields else None,
            )

    elif provider == "webhook":
        webhook_url = cfg.get("webhook_url")
        if webhook_url:
            payload = {
                "event": "error" if is_error else "success",
                "title": title,
                "message": message,
                "diagnosis": diagnosis,
                "url": url,
            }
            sent = await send_generic_webhook(webhook_url, payload=payload)

    return sent


async def on_job_failed(job_id: str, job_data: dict[str, Any], error_msg: str | None = None) -> None:
    """Triggered when a backend video processing job fails."""
    cfg = load_watchdog_config()
    if not cfg.get("enabled", True) or not cfg.get("notify_on_failure", True):
        return

    logs = job_data.get("logs", [])
    title = f"Job Failed ({job_id[:8]})"
    err = error_msg or (logs[-1] if logs else "Job failed with an unknown error.")

    diagnosis = None
    if cfg.get("ai_diagnosis", True):
        diagnosis = await diagnose_failure(
            err,
            logs,
            context={"job_id": job_id, "url": job_data.get("url")},
        )

    await dispatch_alert(
        title=title,
        message=f"Job {job_id[:8]} encountered a failure: {err}",
        diagnosis=diagnosis,
        is_error=True,
    )


async def on_api_error(route: str, error_msg: str, status_code: int = 500, context: dict[str, Any] | None = None) -> None:
    """Triggered when an API endpoint (e.g. publish or compose) catches a 500 error."""
    cfg = load_watchdog_config()
    if not cfg.get("enabled", True) or not cfg.get("notify_on_failure", True):
        return

    title = f"Action Failed on {route}"
    diagnosis = None
    if cfg.get("ai_diagnosis", True):
        diagnosis = await diagnose_failure(
            error_msg,
            [f"HTTP {status_code} on {route}"],
            context=context,
        )

    await dispatch_alert(
        title=title,
        message=f"API error on {route}: {error_msg}",
        diagnosis=diagnosis,
        is_error=True,
    )


async def send_test_alert(cfg_override: dict[str, Any] | None = None) -> bool:
    """Send an immediate test alert to verify notification credentials."""
    return await dispatch_alert(
        title="Test Alert",
        message="ClippyMe AI Watchdog is active and successfully connected to your phone!",
        diagnosis={
            "summary": "Everything is running smoothly.",
            "root_cause": "Manual test alert requested from settings.",
            "action": "No action needed.",
            "auto_recoverable": True,
        },
        is_error=False,
        cfg_override=cfg_override,
    )
