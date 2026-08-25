"""Multi-Layout Reframing and Classification package for ClippyMe.

Provides layout options beyond standard single-speaker tracking:
- SPLIT: 2-speaker podcast vertical stack (top/bottom)
- SCREENCAST: Full-width slides/screen on top, presenter below
- INSET: Corner webcam detector (gaming / Twitch VODs)
- PUNCH_IN: Audio beat/emphasis zoom pulses
- CLASSIFIER: Gemini 12-frame low-cost layout classifier
"""
from clippyme.pipeline.layouts.split_layout import detect_split_scenes, split_filtergraph, split_geometry
from clippyme.pipeline.layouts.screencast_layout import content_bands, speaker_crop, screencast_filtergraph
from clippyme.pipeline.layouts.camera_inset import find_inset_box, is_cornered, inset_filtergraph
from clippyme.pipeline.layouts.punch_in import zoom_curve, find_emphasis_beats
from clippyme.pipeline.layouts.layout_classifier import classify_layout, pick_and_apply_layout

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
]
