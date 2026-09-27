"""Fast caption correction + re-render.

Problem: fixing a misheard caption currently requires re-running the whole
pipeline (download → transcribe → analyze → reframe → compose). This module
patches the transcript words in the job metadata and re-runs ONLY the
subtitle burn via compose_layers — seconds, not minutes.

Usage:
    from clippyme.domain.caption_fix import correct_and_recaption

    corrections = [
        {"start": 12.4, "end": 13.1, "text": "there"},  # fix "their" → "there"
    ]
    out = await correct_and_recaption(
        job_id="qa_talking", clip_index=0,
        output_root=r"C:\\nugget\\output",
        corrections=corrections,
        subtitle_params={"preset": "classic_white", "animate": "pop"},
    )
    # → composed clip filename with fixed captions

Corrections replace the words overlapping [start, end] (source-relative
seconds). The new text is split into words and distributed evenly across
the time range, preserving word-level karaoke timing.
"""
import asyncio
import copy
import json
import logging
import os

logger = logging.getLogger(__name__)


def _load_metadata(job_dir):
    for name in sorted(os.listdir(job_dir)):
        if name.endswith("_metadata.json"):
            with open(os.path.join(job_dir, name), encoding="utf-8") as f:
                return json.load(f), os.path.join(job_dir, name)
    raise FileNotFoundError(f"no metadata JSON in {job_dir}")


def apply_corrections_to_transcript(transcript, corrections, clip_start=0.0):
    """Patch word texts in a transcript dict. Returns (patched, n_changed).

    corrections: [{"start": float, "end": float, "text": str}] in the same
    time base as the transcript words (source-relative by default; pass
    clip_start to interpret corrections as clip-relative).
    """
    patched = copy.deepcopy(transcript)
    # Flatten words with back-pointers.
    flat = []
    for seg in patched.get("segments") or []:
        for w in seg.get("words") or []:
            if isinstance(w, dict) and w.get("start") is not None:
                flat.append(w)
    flat.sort(key=lambda w: float(w["start"]))

    n_changed = 0
    for corr in corrections or []:
        cs = float(corr["start"]) + clip_start
        ce = float(corr["end"]) + clip_start
        new_words = str(corr.get("text", "")).split()
        if not new_words or ce <= cs:
            continue
        # Words overlapping the correction window.
        hit = [w for w in flat
               if float(w.get("end", -1)) > cs and float(w.get("start", 1e9)) < ce]
        if not hit:
            continue
        # Distribute new words evenly across the window.
        span = ce - cs
        per = span / len(new_words)
        # If counts match, keep original per-word timing (best karaoke sync).
        if len(hit) == len(new_words):
            for w, nw in zip(hit, new_words):
                if str(w.get("word", "")).strip() != nw:
                    w["word"] = nw
                    n_changed += 1
        else:
            for w in hit:
                w["word"] = ""  # cleared; replaced below
            # Rewrite the hit range with evenly split new words.
            first_idx = flat.index(hit[0])
            for _ in hit:
                del flat[first_idx]
            for i, nw in enumerate(new_words):
                ws = cs + per * i
                flat.insert(first_idx + i, {
                    "word": nw, "start": ws, "end": ws + per,
                })
                n_changed += 1
            # Rebuild segments from flat list (single segment is fine for captions).
            patched["segments"] = [{"words": flat}]
    return patched, n_changed


async def correct_and_recaption(job_id, clip_index, output_root,
                                corrections, subtitle_params=None,
                                toggles=None, hook_params=None):
    """Apply caption corrections and re-render the clip's captions only.

    1. Loads job metadata, patches transcript words.
    2. Saves metadata (original backed up to *.bak).
    3. Re-runs compose_layers from the clean base clip with subtitles on.
       (Smart Cut / hook toggles pass through; transcription and reframe
       are NOT re-run.)

    Returns the composed filename.
    """
    from clippyme.domain.clip_resolve import resolve_clip
    from clippyme.domain.compose import compose_layers

    job_dir = os.path.join(output_root, job_id)
    metadata, meta_path = _load_metadata(job_dir)

    clip = (metadata.get("shorts") or [])[clip_index]
    clip_start = float(clip.get("start", 0))

    patched, n_changed = apply_corrections_to_transcript(
        metadata.get("transcript", {}), corrections, clip_start=0.0,
    )
    logger.info("caption_fix: %d word(s) corrected for job %s clip %d",
                n_changed, job_id, clip_index)
    if n_changed == 0:
        raise ValueError("no transcript words matched the corrections")

    # Backup + save.
    with open(meta_path + ".bak", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    metadata["transcript"] = patched
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    resolved = await asyncio.to_thread(
        resolve_clip, job_id, clip_index, output_root, for_compose=True,
    )
    # resolved.metadata was loaded before our patch; inject the patched one.
    resolved.metadata["transcript"] = patched

    out = await compose_layers(
        base_clip=resolved.clip_path,
        job_dir=resolved.job_dir,
        clip_index=clip_index,
        metadata=resolved.metadata,
        clip_info=resolved.clip_info,
        toggles=toggles or {"subtitles": True, "smartcut": False, "hook": False},
        hook_params=hook_params or {},
        subtitle_params=subtitle_params or {"preset": "classic_white", "animate": "pop"},
    )
    return out
