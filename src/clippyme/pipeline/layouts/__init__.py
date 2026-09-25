"""Multi-Layout Reframing and Classification package for ClippyMe.

Provides layout options beyond standard single-speaker tracking:
- SPLIT: 2-speaker podcast vertical stack (top/bottom)
- SCREENCAST: Full-width slides/screen on top, presenter below
- INSET: Corner webcam detector (gaming / Twitch VODs)
- PUNCH_IN: Audio beat/emphasis zoom pulses
- CLASSIFIER: Gemini 12-frame low-cost layout classifier
- SPEAKER_SWITCH (Phase 3C): diarization-driven multi-speaker layouts —
  word→turn merging, talk-time shares, layout selection, the
  DiarizationGuide (label→face correlation that steers SpeakerTracker),
  and the per-turn switch schedule.
"""
from clippyme.pipeline.layouts.split_layout import detect_split_scenes, split_filtergraph, split_geometry
from clippyme.pipeline.layouts.screencast_layout import content_bands, speaker_crop, screencast_filtergraph
from clippyme.pipeline.layouts.camera_inset import find_inset_box, is_cornered, inset_filtergraph
from clippyme.pipeline.layouts.punch_in import zoom_curve, find_emphasis_beats
from clippyme.pipeline.layouts.layout_classifier import classify_layout, pick_and_apply_layout
from clippyme.pipeline.layouts.speaker_switch import (
    DiarizationGuide,
    build_switch_schedule,
    select_multi_speaker_layout,
    speaker_turns_from_words,
    talk_time_shares,
)

__all__ = [
    "detect_split_scenes",
    "split_filtergraph",
    "split_geometry",
    "content_bands",
    "speaker_crop",
    "screencast_filtergraph",
    "find_inset_box",
    "is_cornered",
    "inset_filtergraph",
    "zoom_curve",
    "find_emphasis_beats",
    "classify_layout",
    "pick_and_apply_layout",
    "DiarizationGuide",
    "build_switch_schedule",
    "select_multi_speaker_layout",
    "speaker_turns_from_words",
    "talk_time_shares",
]
