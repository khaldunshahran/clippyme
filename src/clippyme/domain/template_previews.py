"""Caption template previews — render a real sample of every preset.

The dashboard's template gallery needs REAL previews (not mockups) so users
can pick a caption style with confidence. This module burns each preset in
``SUBTITLE_PRESETS`` onto a short sample slice and emits a preview MP4 +
poster frame per template.

Usage:
    from clippyme.domain.template_previews import render_template_previews
    render_template_previews(
        sample_video="/path/to/sample_9x16.mp4",
        transcript={"segments": [...]},   # word-level timestamps
        output_dir="/path/to/previews",
        sample_start=0.0,
        sample_duration=8.0,
    )
    # → output_dir/{preset}.mp4 + output_dir/{preset}.jpg

The sample video should be 9:16 with clear speech. A talking-head slice
works best (faces + speech = representative).
"""
import logging
import os
import subprocess

from clippyme.domain.subtitles import (
    SUBTITLE_PRESETS,
    burn_subtitles,
    generate_ass_karaoke,
)

logger = logging.getLogger(__name__)

# Display names + short descriptions for the dashboard gallery.
TEMPLATE_META = {
    "classic_white": {
        "name": "Classic",
        "blurb": "Bold white with yellow karaoke. The proven viral workhorse.",
    },
    "hormozi_bold": {
        "name": "Hormozi",
        "blurb": "Chunky Bangers type with green pop. High-energy talking head.",
    },
    "gold_standard": {
        "name": "Gold",
        "blurb": "Premium gold highlight on extra-bold type. For finance/luxury.",
    },
    "mrbeast_box": {
        "name": "Beast Box",
        "blurb": "Boxed captions with yellow punch. Maximum contrast.",
    },
    "fire_impact": {
        "name": "Impact",
        "blurb": "Anton with red highlight. Loud, urgent, motivational.",
    },
    "clay_plum": {
        "name": "Plum",
        "blurb": "Warm plum highlight. Brand-aligned, friendly, modern.",
    },
    "rust_stroke": {
        "name": "Rust",
        "blurb": "Bold rust highlight. Brand-aligned, confident, earthy.",
    },
    "minimal_clean": {
        "name": "Minimal",
        "blurb": "Clean and subtle. For educational / professional content.",
    },
    "neon_glow": {
        "name": "Neon",
        "blurb": "Cyan glow. For gaming/tech late-night vibes.",
    },
}


def list_templates():
    """Ordered template catalog for the API/dashboard."""
    return [
        {
            "id": preset,
            "name": TEMPLATE_META.get(preset, {}).get("name", preset),
            "blurb": TEMPLATE_META.get(preset, {}).get("blurb", ""),
        }
        for preset in SUBTITLE_PRESETS
    ]


def render_template_previews(sample_video, transcript, output_dir,
                             sample_start=0.0, sample_duration=8.0,
                             animate="pop"):
    """Render a preview MP4 + poster JPG for every caption preset.

    Returns {preset_id: {"mp4": path, "poster": path}}.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # Cut the sample slice once; every preset burns onto the same pixels.
    sample_slice = os.path.join(output_dir, "_sample_slice.mp4")
    if not os.path.exists(sample_slice):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(sample_start), "-i", sample_video,
            "-t", str(sample_duration),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-c:a", "aac", sample_slice,
        ]
        subprocess.run(cmd, check=True, timeout=300)

    for preset in SUBTITLE_PRESETS:
        ass_path = os.path.join(output_dir, f"_preview_{preset}.ass")
        mp4_path = os.path.join(output_dir, f"{preset}.mp4")
        jpg_path = os.path.join(output_dir, f"{preset}.jpg")
        try:
            ok = generate_ass_karaoke(
                transcript,
                sample_start,
                sample_start + sample_duration,
                ass_path,
                preset=preset,
                mode="word_group",
                words_per_group=3,
                animate=animate,
            )
            if not ok:
                logger.warning("template preview: no words for %s", preset)
                continue
            burn_subtitles(
                sample_slice, ass_path, mp4_path,
                2, 16, "Verdana", "#FFFFFF", "#000000", 2, "#000000", 0.0, 0,
            )
            # Poster: middle frame, small.
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", str(sample_duration / 2), "-i", mp4_path,
                    "-frames:v", "1", "-vf", "scale=360:640", jpg_path,
                ],
                check=True, timeout=120,
            )
            results[preset] = {"mp4": mp4_path, "poster": jpg_path}
            logger.info("template preview rendered: %s", preset)
        except Exception as exc:  # one bad preset must not kill the gallery
            logger.warning("template preview failed for %s: %s", preset, exc)
    return results
