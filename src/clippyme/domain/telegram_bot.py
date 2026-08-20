"""Two-way Telegram AI Assistant & Remote Control Listener for ClippyMe.

Runs a resilient long-polling loop in the background:
1. AI Chat: Answers questions about jobs, video editing, and error diagnostics via Gemini.
2. Remote Commands: /status, /history, /retry, /diagnose, /help.
3. Remote Clipping: Automatically downloads and processes any YouTube/Twitch URL pasted in chat.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Callable, Dict, Optional

import httpx

from clippyme.domain.watchdog import diagnose_failure, _get_gemini_api_key
from clippyme.integrations.notifier import send_telegram
from clippyme.storage.config_store import load_watchdog_config

logger = logging.getLogger("clippyme.telegram_bot")

YOUTUBE_TWITCH_REGEX = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+|youtu\.be/[\w-]+|twitch\.tv/[\w-]+|youtube\.com/shorts/[\w-]+)",
    re.IGNORECASE,
)


class TelegramBotListener:
    """Long-polling background listener for Telegram Bot interactions."""

    def __init__(
        self,
        *,
        jobs: dict[str, Any],
        job_queue: asyncio.Queue,
        output_dir: str,
        upload_dir: str,
        data_dir: str,
        run_job_fn: Optional[Callable] = None,
    ) -> None:
        self.jobs = jobs
        self.job_queue = job_queue
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        self.data_dir = data_dir
        self.run_job_fn = run_job_fn
        self._running = False
        self._last_offset = 0

    async def start(self) -> None:
        """Start the background polling loop."""
        self._running = True
        logger.info("Telegram bot listener loop initialized.")
        while self._running:
            try:
                cfg = load_watchdog_config()
                token = (cfg.get("telegram_bot_token") or "").strip()
                chat_id = str(cfg.get("telegram_chat_id") or "").strip()

                if not token or not chat_id:
                    await asyncio.sleep(10)
                    continue

                await self._poll_updates(token, chat_id)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Telegram polling iteration error: %s", exc)
                await asyncio.sleep(5)

    def stop(self) -> None:
        """Stop the background polling loop."""
        self._running = False

    async def _poll_updates(self, token: str, authorized_chat_id: str) -> None:
        """Fetch and process new updates from Telegram."""
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        params = {"timeout": 15, "allowed_updates": ["message"]}
        if self._last_offset > 0:
            params["offset"] = self._last_offset

        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                await asyncio.sleep(5)
                return
            data = resp.json()

        for update in data.get("result", []):
            up_id = update.get("update_id", 0)
            if up_id >= self._last_offset:
                self._last_offset = up_id + 1

            message = update.get("message")
            if not message:
                continue

            msg_chat = str(message.get("chat", {}).get("id", ""))
            text = (message.get("text") or "").strip()
            if not text:
                continue

            # Check authorization against all configured IDs
            allowed_ids = {
                cid.strip().lstrip("-")
                for cid in authorized_chat_id.replace(",", " ").split()
                if cid.strip()
            }
            # Always accept if match found or if msg_chat without prefix is in allowed_ids
            if msg_chat.lstrip("-") not in allowed_ids and msg_chat not in allowed_ids:
                logger.info("Ignoring Telegram message from unauthorized chat: %s (allowed: %s)", msg_chat, allowed_ids)
                continue

            # Process the message asynchronously
            asyncio.create_task(self._handle_user_message(token, msg_chat, text))

    async def _handle_user_message(self, token: str, chat_id: str, text: str) -> None:
        """Process incoming user command, link, or AI chat."""
        try:
            cmd = text.strip()
            lower = cmd.lower()

            if lower in ("/start", "/help"):
                await self._send_help(token, chat_id)
                return

            if lower == "/status":
                await self._send_status(token, chat_id)
                return

            if lower == "/history":
                await self._send_history(token, chat_id)
                return

            if lower == "/retry":
                await self._handle_retry(token, chat_id)
                return

            if lower == "/diagnose":
                await self._handle_diagnose(token, chat_id)
                return

            # Check if user sent a video URL to clip
            url_match = YOUTUBE_TWITCH_REGEX.search(text)
            if url_match:
                await self._handle_video_url(token, chat_id, url_match.group(0))
                return

            # Natural Language AI Chat
            await self._handle_ai_chat(token, chat_id, text)

        except Exception as exc:
            logger.exception("Error handling Telegram message: %s", exc)
            await send_telegram(
                token,
                chat_id,
                text=f"⚠️ *Error processing request*: {exc}",
            )

    async def _send_help(self, token: str, chat_id: str) -> None:
        help_msg = (
            "👋 *Hello! I'm your ClippyMe AI Assistant.*\n\n"
            "Here is what you can do directly from Telegram:\n\n"
            "🎬 *Clip a Video*\n"
            "Just paste any YouTube / Twitch link here, and I will download and create viral clips for you!\n\n"
            "📊 *Commands*\n"
            "• `/status` — View running jobs and queue status\n"
            "• `/history` — List recent videos and clips\n"
            "• `/retry` — Re-run the last failed job\n"
            "• `/diagnose` — Ask AI to diagnose the latest error\n\n"
            "💬 *AI Chat*\n"
            "Ask me anything about your clips, errors, caption settings, or video editing tips!"
        )
        await send_telegram(token, chat_id, text=help_msg)

    async def _send_status(self, token: str, chat_id: str) -> None:
        queued = [j for j in self.jobs.values() if j.get("status") == "queued"]
        running = [j for j in self.jobs.values() if j.get("status") == "running"]
        completed = [j for j in self.jobs.values() if j.get("status") == "completed"]
        failed = [j for j in self.jobs.values() if j.get("status") == "failed"]

        lines = [
            "📊 *ClippyMe System Status*",
            f"• 🏃 *Running*: {len(running)}",
            f"• ⏳ *Queued*: {len(queued)}",
            f"• ✅ *Completed*: {len(completed)}",
            f"• ❌ *Failed*: {len(failed)}",
        ]

        if running:
            lines.append("\n*Active Jobs:*")
            for job in running[:3]:
                jid = job.get("job_id", "")[:8]
                title = job.get("video_title") or "Processing video..."
                logs = job.get("logs", [])
                last_log = logs[-1] if logs else "Working..."
                lines.append(f"• `{jid}`: *{title}*\n  ↳ _{last_log[:60]}_")

        await send_telegram(token, chat_id, text="\n".join(lines))

    async def _send_history(self, token: str, chat_id: str) -> None:
        hist_file = os.path.join(self.data_dir, "history.json")
        items = []
        if os.path.isfile(hist_file):
            try:
                with open(hist_file, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception:
                pass

        if not items:
            await send_telegram(token, chat_id, text="ℹ️ *No history found yet.* Paste a video link to start clipping!")
            return

        lines = ["📜 *Recent ClippyMe Jobs:*"]
        for it in items[:5]:
            title = it.get("title") or "Untitled"
            clip_count = len(it.get("clips") or [])
            jid = it.get("jobId", "")[:8]
            lines.append(f"• *{title}*\n  `{jid}` · {clip_count} clips ready")

        await send_telegram(token, chat_id, text="\n".join(lines))

    async def _handle_diagnose(self, token: str, chat_id: str) -> None:
        failed_jobs = [j for j in self.jobs.values() if j.get("status") == "failed"]
        if not failed_jobs:
            await send_telegram(token, chat_id, text="✅ *All systems healthy!* No recent failed jobs.")
            return

        last_failed = failed_jobs[-1]
        err_text = last_failed.get("error") or "Unknown failure"
        logs = last_failed.get("logs") or []

        await send_telegram(token, chat_id, text="🔍 *Consulting Gemini Doctor...*")
        diag = await diagnose_failure(err_text, logs)

        msg = (
            "🩺 *AI Error Diagnosis:*\n\n"
            f"📌 *Summary*: {diag.get('summary')}\n\n"
            f"🔬 *Root Cause*: {diag.get('root_cause')}\n\n"
            f"💡 *Action*: {diag.get('action')}"
        )
        await send_telegram(token, chat_id, text=msg)

    async def _handle_retry(self, token: str, chat_id: str) -> None:
        failed_jobs = [j for j in self.jobs.values() if j.get("status") == "failed"]
        if not failed_jobs:
            await send_telegram(token, chat_id, text="ℹ️ No failed jobs found to retry.")
            return

        last_failed = failed_jobs[-1]
        jid = last_failed.get("job_id")
        cmd = last_failed.get("cmd")

        if not jid or not cmd:
            await send_telegram(token, chat_id, text="⚠️ Could not restore command for the failed job.")
            return

        last_failed["status"] = "queued"
        last_failed["error"] = None
        await self.job_queue.put(jid)
        await send_telegram(token, chat_id, text=f"🔄 *Job `{jid[:8]}` re-queued successfully!*")

    async def _handle_video_url(self, token: str, chat_id: str, url: str) -> None:
        import uuid
        from clippyme.domain.job_results import build_main_cmd
        from clippyme.domain.job_submission import submit_job

        job_id = str(uuid.uuid4())
        job_out = os.path.join(self.output_dir, job_id)
        os.makedirs(job_out, exist_ok=True)

        cmd = build_main_cmd(
            url=url,
            job_output_dir=job_out,
            reframe_mode="speaker_focus",
            min_duration=30,
            max_duration=60,
            target_clips=5,
            enable_smartcut=True,
        )

        env = dict(os.environ)
        await submit_job(
            jobs=self.jobs,
            job_queue=self.job_queue,
            job_id=job_id,
            cmd=cmd,
            env=env,
            job_output_dir=job_out,
        )

        await send_telegram(
            token,
            chat_id,
            text=(
                f"🎬 *Video Enqueued for Clipping!*\n\n"
                f"🔗 *URL*: `{url}`\n"
                f"🆔 *Job ID*: `{job_id[:8]}`\n\n"
                f"I'll notify you automatically as soon as your clips are ready!"
            ),
        )

    async def _handle_ai_chat(self, token: str, chat_id: str, prompt: str) -> None:
        """Answer user questions with Gemini conversational AI grounded in ClippyMe context."""
        api_key = _get_gemini_api_key()
        if not api_key:
            await send_telegram(
                token,
                chat_id,
                text="🤖 *AI Chat is currently offline* because no Gemini API key is configured. You can set it in ClippyMe Settings!",
            )
            return

        # Prepare context about the current system
        recent_jobs = list(self.jobs.values())[-3:]
        ctx_summary = []
        for j in recent_jobs:
            ctx_summary.append(f"- Job {j.get('job_id', '')[:8]}: {j.get('status')} | Title: {j.get('video_title', 'N/A')} | Error: {j.get('error', 'None')}")

        context_block = "\n".join(ctx_summary) if ctx_summary else "No active jobs."

        system_instruction = (
            "You are the intelligent, helpful, and friendly ClippyMe AI Assistant on Telegram. "
            "ClippyMe is an autonomous AI video clipping and shorts generator desktop application. "
            "The user talking to you is Khaldun. "
            "Answer clearly and concisely. Format using Telegram Markdown (e.g. *bold*, `code`, _italics_).\n\n"
            f"CURRENT APP CONTEXT:\n{context_block}"
        )

        def _ask_gemini() -> str:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{system_instruction}\n\nUser Question:\n{prompt}",
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                return f"Sorry, I had trouble thinking of an answer: {exc}"

        answer = await asyncio.to_thread(_ask_gemini)
        await send_telegram(token, chat_id, text=answer)
