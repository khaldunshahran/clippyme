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
