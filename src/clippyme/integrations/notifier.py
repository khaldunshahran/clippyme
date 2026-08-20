"""Notification dispatchers for ClippyMe watchdog alerts (ntfy, Telegram, Discord, Webhooks).

All dispatchers are asynchronous, timeout-bounded, and exception-safe. Failures in
the notification channel are logged as warnings and never propagate or disrupt
the main video pipeline.
"""
from __future__ import annotations

import logging
from typing import Any
import httpx

logger = logging.getLogger("clippyme")
_TIMEOUT = 10.0


async def send_ntfy(
    topic: str,
    *,
    title: str,
    message: str,
    priority: str = "default",
    tags: list[str] | None = None,
    click_url: str | None = None,
    server: str = "https://ntfy.sh",
) -> bool:
    """Send a push notification via ntfy (e.g. ntfy.sh/<topic>)."""
    if not topic or not topic.strip():
        return False
    topic_clean = topic.strip().lstrip("/")
    base_server = (server or "https://ntfy.sh").rstrip("/")
    url = f"{base_server}/{topic_clean}"

    headers = {
        "Title": title[:200],
        "Priority": priority,
    }
    if tags:
        headers["Tags"] = ",".join(tags)
    if click_url:
        headers["Click"] = click_url

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, content=message.encode("utf-8"), headers=headers)
            if resp.status_code < 400:
                logger.info("ntfy notification sent successfully to %s", topic_clean)
                return True
            logger.warning("ntfy returned HTTP %s: %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Failed to send ntfy notification to %s: %s", topic_clean, exc)
    return False


async def send_telegram(
    bot_token: str,
    chat_id: str,
    *,
    text: str,
    parse_mode: str = "Markdown",
) -> bool:
    """Send a message via Telegram Bot API."""
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
    payload = {
        "chat_id": chat_id.strip(),
        "text": text[:4000],
        "parse_mode": parse_mode,
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info("Telegram notification sent successfully to chat %s", chat_id)
                return True
            logger.warning("Telegram API error (HTTP %s): %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Failed to send Telegram notification: %s", exc)
    return False


async def send_discord_webhook(
    webhook_url: str,
    *,
    title: str,
    description: str,
    color: int = 0xE02424,  # Red for alert
    fields: list[dict[str, Any]] | None = None,
) -> bool:
    """Send a rich embed message via Discord Webhook."""
    if not webhook_url or not webhook_url.startswith("http"):
        return False
    embed: dict[str, Any] = {
        "title": title[:250],
        "description": description[:2000],
        "color": color,
    }
    if fields:
        embed["fields"] = fields[:25]
    payload = {
        "username": "ClippyMe Watchdog",
        "embeds": [embed],
    }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(webhook_url.strip(), json=payload)
            if resp.status_code in (200, 204):
                logger.info("Discord webhook alert sent successfully")
                return True
            logger.warning("Discord webhook error (HTTP %s): %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Failed to send Discord webhook: %s", exc)
    return False


async def send_generic_webhook(
    url: str,
    *,
    payload: dict[str, Any],
) -> bool:
    """Send a JSON payload to a generic custom webhook."""
    if not url or not url.startswith("http"):
        return False
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(url.strip(), json=payload)
            if resp.status_code < 400:
                logger.info("Generic webhook notification sent to %s", url)
                return True
            logger.warning("Generic webhook error (HTTP %s): %s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.warning("Failed to send generic webhook to %s: %s", url, exc)
    return False
