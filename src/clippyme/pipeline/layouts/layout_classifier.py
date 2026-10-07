"""Intelligent Layout Classifier using Gemini with 12-frame low-cost visual sampling.

Sends 12 downscaled (1024px) JPEG frames (~3k tokens) to classify the optimal
vertical framing strategy:
- "none" / "standard": single speaker or standard scene
- "split": two people in conversation / podcast
- "screencast": screen recording, spreadsheet, presentation
- "inset": gaming or stream VOD with corner webcam
"""
import json
import os

ENABLED = os.environ.get("AUTO_LAYOUT", "0").strip().lower() in ("1", "true", "yes")
SAMPLE_FRAMES = int(os.environ.get("LAYOUT_SAMPLE_FRAMES", "12"))
SAMPLE_WIDTH = int(os.environ.get("LAYOUT_SAMPLE_WIDTH", "1024"))

VALID_LAYOUTS = {"none", "standard", "split", "screencast", "inset"}

LAYOUT_PROMPT = """Analyze these 12 sampled frames from a video.
Determine the most appropriate vertical (9:16) video framing layout among these options:
1. "split": A two-person podcast or dialogue where two main speakers sit apart in a wide shot.
2. "screencast": A screen recording, software demo, presentation, code editor, or spreadsheet where full-width screen content is essential.
3. "inset": A gaming stream or VOD where a small webcam facecam is pinned in a corner over full-screen gameplay/content.
4. "none": Standard talking head, vlog, monologue, or standard video (handled by standard smart active-speaker reframing).

Return strict JSON:
{
  "layout": "none" | "split" | "screencast" | "inset",
  "confidence": 0.0 to 1.0,
  "why": "concise explanation"
}
"""


def sample_frames(video_path, n=SAMPLE_FRAMES, width=SAMPLE_WIDTH):
    """Extract JPEG bytes for n frames distributed evenly across the video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        if total <= 0:
            return out
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * total / float(n)))
            ok, frame = cap.read()
            if not ok or frame.mean() < 16:
                continue
            h, w = frame.shape[:2]
            scaled = cv2.resize(frame, (width, max(2, int(h * width / float(w)))), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                out.append(buf.tobytes())
    finally:
        cap.release()
    return out


def classify_layout(video_path, api_key=None, model_name=None):
    """Classify video layout using Gemini. Never raises: returns 'none' on any error."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "none"

    from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model

    model_name = get_auxiliary_gemini_model(model_name)
    try:
        from google import genai
        from google.genai import types as genai_types

        frames = sample_frames(video_path)
        if not frames:
            return "none"

        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in frames]
        response = client.models.generate_content(
            model=model_name,
            contents=parts + [LAYOUT_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        if not response or not response.text:
            return "none"

        data = json.loads(response.text) or {}
        layout = str(data.get("layout", "none")).strip().lower()
        if layout == "standard":
            layout = "none"
        return layout if layout in VALID_LAYOUTS else "none"
    except Exception as e:
        print(f"[LayoutClassifier] Optional layout classification failed: {e}")
        return "none"


def pick_and_apply_layout(video_path, explicit_layout=None, api_key=None):
    """Determine the effective layout strategy."""
    if explicit_layout and explicit_layout not in ("auto", "none"):
        return explicit_layout

    if ENABLED or explicit_layout == "auto":
        detected = classify_layout(video_path, api_key=api_key)
        return detected if detected != "none" else "auto"

    return "auto"


# --- Task 02: scene-local semantic classification ---------------------------
# Separate semantic ELIGIBILITY (is there a screen?) from presenter GEOMETRY
# (where is the speaker?). Auto-promotion requires both.

# Policy defaults (provisional, not calibrated). Conservative: failures
# retain the previous strategy.
MIN_SEMANTIC_CONFIDENCE = 0.85
MIN_SCENE_SAMPLES = 3
MIN_PRESENTER_DETECTIONS = 3
MIN_PRESENTER_COVERAGE = 0.60
MAX_PRESENTER_SPREAD_FRACTION = 0.10

# Budget bounds (per scene)
MAX_SCENE_SAMPLES = 8
CLASSIFY_TIMEOUT_S = 30


def parse_layout_response(text):
    """Validate a provider JSON response into structured evidence.

    Returns dict with layout, confidence, screen_evidence, reason;
    or None when invalid. Never raises.
    """
    import json
    import math

    try:
        data = json.loads(text or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    layout = str(data.get("layout", "")).strip().lower()
    if layout == "standard":
        layout = "none"
    if layout not in VALID_LAYOUTS:
        return None
    conf = data.get("confidence")
    # Reject NaN/Infinity/bool/non-numeric; must be finite in [0, 1].
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return None
    if not math.isfinite(conf) or not (0.0 <= conf <= 1.0):
        return None
    reason = data.get("why") or data.get("reason") or ""
    if not isinstance(reason, str):
        return None
    # Positive screen-content evidence: explicit flag or a screencast/
    # inset label with high confidence. A bare label is not sufficient.
    screen_evidence = bool(data.get("screen_evidence"))
    if layout in ("screencast", "inset") and conf >= MIN_SEMANTIC_CONFIDENCE:
        screen_evidence = True
    return {
        "layout": layout,
        "confidence": float(conf),
        "screen_evidence": screen_evidence,
        "reason": reason[:500],
    }


def sample_scene_frames(video_path, start_frame, end_frame, n=MAX_SCENE_SAMPLES,
                        width=SAMPLE_WIDTH):
    """JPEG bytes for n frames within [start_frame, end_frame).

    Deduplicates indices; all samples lie inside the half-open scene
    bounds. Returns [] on unreadable video. Never raises.
    """
    import cv2

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        try:
            s, e = int(start_frame), int(end_frame)
            if e <= s:
                return []
            # Evenly spaced, deduplicated, clipped to [s, e).
            import numpy as np
            idxs = sorted(set(int(round(x)) for x in np.linspace(s, e - 1, min(n, e - s))))
            idxs = [i for i in idxs if s <= i < e]
            out = []
            for f_idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if not ok or frame is None or frame.mean() < 16:
                    continue
                h, w = frame.shape[:2]
                scaled = cv2.resize(frame, (width, max(2, int(h * width / float(w)))),
                                    interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    out.append(buf.tobytes())
            return out
        finally:
            cap.release()
    except Exception:
        return []


def classify_scene(video_path, scene_id, start_frame, end_frame,
                   provider=None, api_key=None, model_name=None):
    """Classify one scene's layout. Never raises; returns evidence dict or None.

    ``provider``: injectable callable(frames, prompt) -> response text.
    When None, uses the Gemini provider (requires AUTO_LAYOUT + api_key).
    Returns None on: disabled, no frames, provider error, invalid response,
    low confidence. Callers retain the previous strategy on None.
    """
    if not is_auto_layout_enabled():
        return None
    frames = sample_scene_frames(video_path, start_frame, end_frame)
    if len(frames) < MIN_SCENE_SAMPLES:
        return None
    try:
        if provider is not None:
            text = provider(frames, LAYOUT_PROMPT)
        else:
            text = _gemini_classify(frames, api_key=api_key, model_name=model_name)
    except Exception:
        return None
    if not text:
        return None
    parsed = parse_layout_response(text)
    if parsed is None:
        return None
    if parsed["confidence"] < MIN_SEMANTIC_CONFIDENCE:
        return None
    parsed["scene_id"] = scene_id
    parsed["samples"] = len(frames)
    return parsed


def _gemini_classify(frames, api_key=None, model_name=None):
    """Gemini provider. Returns response text or None. Never raises."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from clippyme.pipeline.gemini_service import get_auxiliary_gemini_model
        from google import genai
        from google.genai import types as genai_types

        model_name = get_auxiliary_gemini_model(model_name)
        client = genai.Client(api_key=api_key)
        parts = [genai_types.Part.from_bytes(data=b, mime_type="image/jpeg")
                 for b in frames]
        response = client.models.generate_content(
            model=model_name,
            contents=parts + [LAYOUT_PROMPT],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        if not response or not response.text:
            return None
        return response.text
    except Exception as e:
        print(f"[LayoutClassifier] scene classification failed: {e}")
        return None


def is_auto_layout_enabled():
    """AUTO_LAYOUT as opt-in flag. Unset/empty/false -> False. Never raises."""
    try:
        return os.environ.get("AUTO_LAYOUT", "0").strip().lower() in ("1", "true", "yes")
    except Exception:
        return False


# Keep the module-level ENABLED for backward compat (import-time snapshot).
# New code should call is_auto_layout_enabled() for invocation-time config.
