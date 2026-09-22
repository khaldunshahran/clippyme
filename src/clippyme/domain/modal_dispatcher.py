"""Modal.com serverless GPU dispatcher.

Offloads intensive CV/ML video processing (Whisper, MediaPipe/YOLO, FFmpeg)
to on-demand Nvidia T4 GPUs on Modal.com ($30/month recurring free credit).
Falls back gracefully to local subprocess workers when MODAL_ENABLED=0.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("clippyme")


def is_modal_enabled() -> bool:
    """Return True if Modal GPU offloading is enabled via environment variables."""
    return os.getenv("MODAL_ENABLED", "0").lower() in ("1", "true", "yes")


def get_modal_config() -> Dict[str, Any]:
    """Return Modal deployment configuration."""
    return {
        "enabled": is_modal_enabled(),
        "app_name": os.getenv("MODAL_APP_NAME", "clippyme-gpu-worker"),
        "function_name": os.getenv("MODAL_FUNCTION_NAME", "run_clipping_job"),
        "gpu_type": os.getenv("MODAL_GPU_TYPE", "T4"),
        "timeout": int(os.getenv("MODAL_TIMEOUT", "900")),
    }


async def dispatch_job_to_modal(job_id: str, job_data: Dict[str, Any]) -> bool:
    """Attempt to dispatch a job to Modal.com serverless GPU.

    Returns True if successfully submitted to Modal, or False if falling
    back to local execution.
    """
    if not is_modal_enabled():
        return False

    try:
        import modal  # type: ignore
    except ImportError:
        logger.warning(
            "MODAL_ENABLED=1 but the 'modal' package is not installed. Falling back to local worker."
        )
        return False

    try:
        cfg = get_modal_config()
        app_name = cfg["app_name"]
        fn_name = cfg["function_name"]

        logger.info("Connecting to Modal function %s.%s for job %s", app_name, fn_name, job_id)
        fn = modal.Function.from_name(app_name, fn_name)

        # Prepare serializable payload
        payload = {
            "job_id": job_id,
            "cmd": job_data.get("cmd", []),
            "env": {
                k: v for k, v in job_data.get("env", {}).items()
                if not k.startswith("MODAL_")
            },
            "input_path": job_data.get("input_path"),
            "output_dir": job_data.get("output_dir"),
        }

        # Spawn asynchronously on Modal GPU cluster
        call = fn.spawn(payload)
        job_data["modal_call_id"] = getattr(call, "object_id", str(call))
        logger.info("Dispatched job %s to Modal (call ID: %s)", job_id, job_data["modal_call_id"])
        return True
    except Exception as exc:
        logger.error("Failed to dispatch job %s to Modal: %s. Falling back to local.", job_id, exc)
        return False
