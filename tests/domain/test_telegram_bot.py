"""Host tests for TelegramBotListener command routing and URL parsing."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from clippyme.domain.telegram_bot import TelegramBotListener, YOUTUBE_TWITCH_REGEX


class TestTelegramBotListener(unittest.IsolatedAsyncioTestCase):
    def test_regex_url_matching(self):
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://twitch.tv/shroud",
            "Check this out: https://youtube.com/shorts/abc123xyz and make clips",
        ]
        for u in urls:
            match = YOUTUBE_TWITCH_REGEX.search(u)
            self.assertIsNotNone(match, f"Failed matching {u}")

    async def test_handle_user_message_help(self):
        listener = TelegramBotListener(
            jobs={},
            job_queue=asyncio.Queue(),
            output_dir="/tmp",
            upload_dir="/tmp",
            data_dir="/tmp",
        )
        with patch("clippyme.domain.telegram_bot.send_telegram", new_callable=AsyncMock) as mock_send:
            await listener._handle_user_message("mock_token", "12345", "/help")
            mock_send.assert_called_once()
            self.assertIn("ClippyMe AI Assistant", mock_send.call_args.kwargs["text"])

    async def test_handle_user_message_status(self):
        listener = TelegramBotListener(
            jobs={"job1": {"status": "running", "video_title": "Test Vid", "logs": ["Step 1"]}},
            job_queue=asyncio.Queue(),
            output_dir="/tmp",
            upload_dir="/tmp",
            data_dir="/tmp",
        )
        with patch("clippyme.domain.telegram_bot.send_telegram", new_callable=AsyncMock) as mock_send:
            await listener._handle_user_message("mock_token", "12345", "/status")
            mock_send.assert_called_once()
            self.assertIn("ClippyMe System Status", mock_send.call_args.kwargs["text"])
            self.assertIn("*Running*: 1", mock_send.call_args.kwargs["text"])


if __name__ == "__main__":
    unittest.main()
