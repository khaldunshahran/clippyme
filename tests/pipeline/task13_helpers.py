"""Task 13 pipeline-test helpers: production-valid Task 12 plans.

Builds minimal but production-valid turn-aware framing plans through the
real ``plan_from_dict``/``plan_to_dict`` parser (``framing_plan`` is
stdlib-only, so this is host-safe), so the domain bridge under test
exercises its real validation path. Each plan carries a caller-chosen
``marker`` in its digests so per-attempt plans are distinguishable.
"""

import json
import os
from pathlib import Path

from clippyme.pipeline.framing_plan import plan_from_dict, plan_to_dict


def sidecar_for(video_path: str) -> str:
    """Canonical sidecar path (mirrors reframe.turn_framing_sidecar_for)."""
    return os.path.splitext(video_path)[0] + ".turn_framing_plan.json"


def raw_task12_plan(marker: str, frame_count: int = 60, fps: float = 30.0) -> dict:
    """A production-valid Task 12 plan dict: one TRACK scene + one SPLIT."""
    half = frame_count // 2
    return {
        "schema_version": 3,
        "producer": "clippyme-turn-framing-plan",
        "source_fingerprint": "fp-" + "b" * 16,
        "frame_w": 1080,
        "frame_h": 1920,
        "fps": fps,
        "frame_count": frame_count,
        "scene_intervals": [[0, half - 1], [half, frame_count - 1]],
        "config_digest": f"cfg-{marker}",
        "tracker_digest": f"trk-{marker}",
        "output_aspect": 9 / 16,
        "analysis_complete": True,
        "created_at": "2026-10-09T00:00:00+0000",
        "scenes": [
            {
                "scene_id": 0,
                "strategy": "TRACK",
                "start_frame": 0,
                "end_frame": half - 1,
                "applies": True,
                "tracks": [
                    {"id": "t-0", "first_frame": 0,
                     "last_frame": half - 1, "n_obs": half - 5},
                ],
                "segments": [
                    {
                        "scene_id": 0,
                        "start_frame": 0,
                        "end_frame": half - 1,
                        "subject_ids": ["t-0"],
                        "kind": "single",
                        "confidence": 0.9,
                        "evidence": {},
                        "reason": "turn-accepted",
                        "crop": [200, 400, 500, 900],
                    },
                ],
                "keyframes": [
                    {"frame": 0, "x1": 200, "y1": 400, "x2": 500, "y2": 900,
                     "kind": "lock"},
                    {"frame": half - 1, "x1": 210, "y1": 410, "x2": 510,
                     "y2": 910, "kind": "ease"},
                ],
                "fallback": None,
                "fallback_crop": None,
            },
            {
                "scene_id": 1,
                "strategy": "SPLIT",
                "start_frame": half,
                "end_frame": frame_count - 1,
                "applies": False,
                "tracks": [],
                "segments": [],
                "keyframes": [],
                "fallback": None,
                "fallback_crop": None,
            },
        ],
    }


def write_task12_sidecar(video_path: str, marker: str) -> tuple[str, dict]:
    """Write a production-valid plan next to ``video_path`` (attempt-style).

    Returns (sidecar_path, canonical_plan_dict).
    """
    plan = plan_to_dict(plan_from_dict(raw_task12_plan(marker)))
    sidecar = sidecar_for(video_path)
    Path(sidecar).write_text(
        json.dumps(plan, sort_keys=True), encoding="utf-8")
    return sidecar, plan
