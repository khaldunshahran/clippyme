"""Phase B2: AI copilot prompt-to-edit (NL instruction -> ClipProject patch).

The floating copilot in the chat UI (and the editor) POSTs a plain-English
instruction to ``/api/edit-ai/{job_id}/{clip_index}`` with ``mode="patch"``.
Gemini translates the instruction into a *patch dict* over the clip project;
the patch is applied to a copy of the project and must pass
``validate_project()`` + ``validate_layer_order()`` before it is returned.
An invalid patch raises ``ValueError`` -- the API maps that to a clean 400
and the stored project is never touched (no silent corruption).

Prompt building + response parsing are pure (host-unit-testable), mirroring
``clip_edit_ai.py``. The Gemini call is a thin wrapper using the same
``google-genai`` client.
"""
from __future__ import annotations

import json
import logging
import math
import re

logger = logging.getLogger(__name__)

MAX_INSTRUCTION_CHARS = 1000
MAX_WORD_EDIT_CHARS = 80

# Patch keys the copilot may emit. Anything else -> ValueError (400).
PATCH_KEYS = frozenset({
    "hook_text",
    "caption_style",
    "caption_position",
    "grade_preset",
    "word_edits",
    "word_colors",
    "crop_nudge",
})


def build_patch_prompt(project_summary: dict, segments: list[dict], instruction: str) -> str:
    """Build the Gemini prompt. Pure.

    ``project_summary`` carries the current project facts the model needs
    (hook text, caption style/position, grade preset, source dims, word count).
    ``segments`` are clip-relative transcript lines for word-correction context.
    """
    summary = project_summary or {}
    lines = []
    for s in (segments or [])[:200]:
        try:
            st = float(s.get("start", 0))
            en = float(s.get("end", 0))
        except (TypeError, ValueError):
            continue
        txt = (s.get("text", "") or "").replace("\n", " ").strip()
        idx = s.get("index", "?")
        lines.append(f"[{idx}] [{st:.2f}-{en:.2f}] {txt}")
    transcript_block = "\n".join(lines) if lines else "(no transcript available)"
    instruction = (instruction or "").strip()[:MAX_INSTRUCTION_CHARS]

    current = (
        f"- hook text: {summary.get('hook_text')!r}\n"
        f"- caption style: {summary.get('caption_style')} "
        f"(allowed: {', '.join(sorted(summary.get('caption_styles') or []))})\n"
        f"- caption position: {summary.get('caption_position')} "
        f"(allowed: {', '.join(sorted(summary.get('caption_positions') or []))})\n"
        f"- grade preset: {summary.get('grade_preset')} "
        f"(allowed: {', '.join(sorted(summary.get('grade_presets') or []))})\n"
        f"- source: {summary.get('source_width')}x{summary.get('source_height')} px, "
        f"{summary.get('source_duration_s')} s; transcript words: {summary.get('word_count')}\n"
    )
    return (
        "You are a precise video-editing copilot. The user wants this change "
        "to the current clip edit:\n"
        f"\"{instruction}\"\n\n"
        "Current edit:\n"
        f"{current}\n"
        "TRANSCRIPT (word index in [brackets] for word corrections):\n"
        f"{transcript_block}\n\n"
        "Return ONLY a JSON object in this exact shape and nothing else:\n"
        '{"patch": {<only the keys below>}, "explanation": "<one short sentence>"}\n\n'
        "Allowed patch keys (include ONLY keys the instruction clearly asks for):\n"
        '- "hook_text": string -- replace the on-screen hook text.\n'
        '- "caption_style": one of the allowed caption styles above.\n'
        '- "caption_position": one of the allowed caption positions above.\n'
        '- "grade_preset": one of the allowed grade presets above.\n'
        '- "word_edits": object mapping transcript word-index (as string) to '
        "corrected word text, e.g. {\"42\": \"their\"}. Only fix real "
        "mis-transcriptions; never rewrite phrasing.\n"
        '- "word_colors": object mapping word ID to 0/1/2 (highlight words '
        "in a template color: 0 = default, 1 = Color 1, 2 = Color 2).\n"
        '- "crop_nudge": {"dx": <pixels>, "dy": <pixels>} -- shift the crop '
        "window right/down (negative = left/up), in source pixels.\n\n"
        "Rules:\n"
        "- If the instruction asks for nothing you can express with these "
        'keys, return {"patch": {}, "explanation": "<why nothing changed>"}.\n'
        "- Never invent keys outside the allowed set.\n"
        "- Never invent word indices outside the transcript.\n"
    )


def parse_patch_response(text: str) -> dict:
    """Parse the model's JSON into {"patch": {...}, "explanation": str}.

    Pure. Raises ValueError on malformed output or unknown patch keys --
    the caller maps that to a 400 (never a silent no-op).
    """
    if not text or not text.strip():
        raise ValueError("empty model response")
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = re.sub(r"^json\\s*", "", raw, flags=re.IGNORECASE).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("model response contained no JSON object")
    try:
        obj = json.loads(raw[start:end + 1])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"model response was not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("model response must be a JSON object")
    patch = obj.get("patch", {})
    if not isinstance(patch, dict):
        raise ValueError('"patch" must be an object')
    unknown = set(patch) - PATCH_KEYS
    if unknown:
        raise ValueError(f"unknown patch keys: {sorted(unknown)}")
    explanation = str(obj.get("explanation", ""))[:300]
    return {"patch": patch, "explanation": explanation}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def apply_copilot_patch(project, patch: dict):
    """Apply a copilot patch to a ClipProject (mutates + returns it).

    Supported keys: hook_text, caption_style, caption_position, grade_preset,
    word_edits, word_colors, crop_nudge. Raises ValueError on anything
    invalid -- never silently corrupts.
    """
    from clippyme.domain.clip_project import (
        CAPTION_POSITIONS,
        CAPTION_STYLES,
        GRADE_PRESETS,
        Overlay,
        validate_word_colors_mapping,
    )

    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")
    unknown = set(patch) - PATCH_KEYS
    if unknown:
        raise ValueError(f"unknown patch keys: {sorted(unknown)}")

    if "hook_text" in patch:
        text = patch["hook_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("hook_text must be a non-empty string")
        text = text.strip()[:200]
        hook = next((o for o in project.overlays if o.type == "hook"), None)
        if hook is not None:
            hook.text = text
        else:
            dur = project.source.duration or 10.0
            project.overlays.append(Overlay(
                type="hook", text=text, style="viral",
                start=0.0, end=round(min(3.0, dur), 3), position="top"))

    if "caption_style" in patch:
        style = patch["caption_style"]
        if style not in CAPTION_STYLES:
            raise ValueError(f"unknown caption style: {style!r}")
        project.captions.style = style

    if "caption_position" in patch:
        pos = patch["caption_position"]
        if pos not in CAPTION_POSITIONS:
            raise ValueError(f"unknown caption position: {pos!r}")
        project.captions.position = pos

    if "grade_preset" in patch:
        preset = patch["grade_preset"]
        if preset not in GRADE_PRESETS:
            raise ValueError(f"unknown grade preset: {preset!r}")
        project.grade.preset = preset

    if "word_edits" in patch:
        edits = patch["word_edits"]
        if not isinstance(edits, dict) or not edits:
            raise ValueError("word_edits must be a non-empty object")
        words = project.captions.words or []
        by_id = {w.id: i for i, w in enumerate(words) if w.id}
        for key, val in edits.items():
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"word_edits[{key}] must be a non-empty string")
            idx = None
            # Stable word IDs first; legacy word-index strings still accepted.
            if isinstance(key, str) and key in by_id:
                idx = by_id[key]
            elif isinstance(key, str) and key.isdigit() and int(key) < len(words):
                idx = int(key)
            if idx is None:
                raise ValueError(
                    f"word_edits key must be a word id or a word index "
                    f"string, got {key!r}")
            wid = words[idx].id or str(idx)
            project.captions.edits[wid] = val.strip()[:MAX_WORD_EDIT_CHARS]

    if "word_colors" in patch:
        wc = patch["word_colors"]
        if not isinstance(wc, dict) or not wc:
            raise ValueError("word_colors must be a non-empty object")
        words = project.captions.words or []
        known_ids = {w.id for w in words if w.id}
        # Same strict messages as the Captions model validator.
        validated = validate_word_colors_mapping(wc, known_ids)
        if project.captions.word_colors is None:
            project.captions.word_colors = {}
        for key, val in validated.items():
            project.captions.word_colors[key] = val

    if "crop_nudge" in patch:
        nudge = patch["crop_nudge"]
        if not isinstance(nudge, dict):
            raise ValueError("crop_nudge must be an object with dx/dy")
        try:
            dx = float(nudge.get("dx", 0.0))
            dy = float(nudge.get("dy", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"crop_nudge dx/dy must be numbers: {exc}") from exc
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError("crop_nudge dx/dy must be finite numbers")
        sw = float(project.source.width or 0)
        sh = float(project.source.height or 0)
        if abs(dx) > sw or abs(dy) > sh:
            raise ValueError("crop_nudge exceeds source dimensions")
        for seg in project.segments or []:
            crop = seg.crop
            crop.x = _clamp(crop.x + dx, 0.0, max(0.0, sw - crop.w))
            crop.y = _clamp(crop.y + dy, 0.0, max(0.0, sh - crop.h))

    return project


def apply_patch_to_project_data(project_data: dict, patch: dict) -> dict:
    """Validate -> apply patch -> re-validate -> layer-order check.

    Pure-ish (no I/O). Returns the patched project as a plain dict.
    Raises ValueError / pydantic.ValidationError / ComposeOrderError on any
    problem -- the API maps all of these to a clean 400.
    """
    from pydantic import ValidationError as PydanticValidationError

    from clippyme.domain.clip_project import validate_project
    from clippyme.domain.compose import validate_layer_order

    project = validate_project(project_data)
    apply_copilot_patch(project, patch)
    validated = validate_project(project.model_dump())
    # Projects always render in canonical layer order (the schema has no
    # custom order field); the check guards against future schema drift.
    validate_layer_order(None)
    return validated.model_dump()


def project_summary_for_prompt(project) -> dict:
    """Compact project facts for the patch prompt. Pure."""
    from clippyme.domain.clip_project import (
        CAPTION_POSITIONS,
        CAPTION_STYLES,
        GRADE_PRESETS,
    )

    hook = next((o for o in (project.overlays or []) if o.type == "hook"), None)
    return {
        "hook_text": hook.text if hook else None,
        "caption_style": project.captions.style,
        "caption_styles": sorted(CAPTION_STYLES),
        "caption_position": project.captions.position,
        "caption_positions": sorted(CAPTION_POSITIONS),
        "grade_preset": project.grade.preset,
        "grade_presets": sorted(GRADE_PRESETS),
        "source_width": project.source.width,
        "source_height": project.source.height,
        "source_duration_s": project.source.duration,
        "word_count": len(project.captions.words or []),
    }


def suggest_project_patch(
    *,
    api_key: str,
    model: str,
    project_summary: dict,
    segments: list[dict],
    instruction: str,
) -> dict:
    """Ask Gemini for a project patch. Returns {"patch": {...}, "explanation"}.

    Network/SDK errors collapse into an empty patch with the error in
    ``explanation`` (like the trim path, a failed suggestion never 500s).
    Malformed model output raises ValueError -> the API returns 400.
    """
    if not api_key:
        return {"patch": {}, "explanation": "Gemini API key not configured."}
    prompt = build_patch_prompt(project_summary, segments, instruction)
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", "") or ""
        result = parse_patch_response(text)
        logger.info(
            "copilot_patch: model=%s instruction_len=%d -> %d patch keys",
            model, len(instruction or ""), len(result["patch"]),
        )
        return result
    except ValueError:
        raise
    except Exception as e:  # pragma: no cover -- network path
        from clippyme.pipeline.gemini_service import _redact_key

        logger.warning("copilot_patch suggest_project_patch failed: %s", e)
        return {"patch": {}, "explanation": f"AI edit failed: {_redact_key(str(e))}"}
