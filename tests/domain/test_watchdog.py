"""Host tests for the ClippyMe watchdog, AI diagnostics, and alert dispatchers."""
import asyncio
import os
import unittest
from unittest.mock import patch

from clippyme.domain.watchdog import (
    diagnose_failure,
    dispatch_alert,
    on_job_failed,
    on_api_error,
)
from clippyme.integrations.notifier import (
    send_discord_webhook,
    send_generic_webhook,
    send_ntfy,
    send_telegram,
)
from clippyme.storage.config_store import (
    load_watchdog_config,
    save_watchdog_config,
    watchdog_config_status,
)


class TestWatchdog(unittest.IsolatedAsyncioTestCase):
    async def test_diagnose_failure_graceful_fallback_without_key(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("clippyme.storage.config_store.load_persistent_config", return_value={}):
            res = await diagnose_failure("ffmpeg exited with code 1", ["frame 1", "frame 2"])
            self.assertIn("summary", res)
            self.assertIn("root_cause", res)
            self.assertIn("action", res)
            self.assertFalse(res.get("auto_recoverable"))

    async def test_notifier_invalid_inputs_fail_gracefully(self):
        self.assertFalse(await send_ntfy("", title="T", message="M"))
        self.assertFalse(await send_telegram("", "", text="M"))
        self.assertFalse(await send_discord_webhook("invalid-url", title="T", description="D"))
        self.assertFalse(await send_generic_webhook("invalid-url", payload={}))

    async def test_dispatch_alert_disabled_is_noop(self):
        res = await dispatch_alert(
            title="Test",
            message="Msg",
            cfg_override={"enabled": False},
        )
        self.assertFalse(res)


class TestWatchdogConfig(unittest.TestCase):
    def test_watchdog_config_persistence_and_masking(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as td:
            tmp_path = pathlib.Path(td)
            with patch("clippyme.storage.config_store.CONFIG_FILE", str(tmp_path / "config.json")), \
                 patch("clippyme.storage.config_store.DATA_DIR", str(tmp_path)):

                save_watchdog_config({
                    "enabled": True,
                    "ai_diagnosis": True,
                    "provider": "ntfy",
                    "ntfy_topic": "my-private-topic",
                    "telegram_bot_token": "123456789:ABCdefGHIjklMNOpqrsTUVwxyz",
                    "telegram_chat_id": "987654321",
                    "discord_webhook_url": "https://discord.com/api/webhooks/123/xyz",
                })

                cfg = load_watchdog_config()
                self.assertTrue(cfg["enabled"])
                self.assertEqual(cfg["ntfy_topic"], "my-private-topic")
                self.assertEqual(cfg["telegram_bot_token"], "123456789:ABCdefGHIjklMNOpqrsTUVwxyz")

                status = watchdog_config_status()
                self.assertTrue(status["enabled"])
                self.assertEqual(status["ntfy_topic"], "my-private-topic")
                self.assertTrue(status["has_telegram_token"])
                self.assertIn("...", status["telegram_bot_token_masked"])
                self.assertIn("...", status["discord_webhook_masked"])
                self.assertNotIn("123456789:ABCdefGHIjklMNOpqrsTUVwxyz", status["telegram_bot_token_masked"])


if __name__ == "__main__":
    unittest.main()
