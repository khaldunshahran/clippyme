"""YouTube Studio domain logic: Viral Titles, Conversational Refinement, AI Thumbnails, Chapters."""
import json
import os
from typing import Optional, List, Dict, Any
from PIL import Image

TEXT_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash"


def generate_viral_titles(
    transcript_text: str,
    api_key: str,
    language: str = "en",
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze transcript text with Gemini and suggest 10 CTR-optimized YouTube titles."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = model_name or TEXT_MODEL

    prompt = f"""You are an elite YouTube CTR & title strategist.
Analyze this transcript and suggest 10 high-performing YouTube titles designed to maximize CTR without being misleading clickbait.

TRANSCRIPT:
{transcript_text[:15000]}

RULES:
- Titles under 70 characters
- Use power words, curiosity gaps, and emotional hooks
- Mix styles: How-to, listicle, story-driven, controversial, question-based
- Language: {language}

OUTPUT JSON:
{{
  "titles": ["title1", "title2", ...],
  "summary": "2-3 sentence summary",
  "recommended": [
    {{"index": 0, "reason": "Why this title has highest CTR potential"}},
    {{"index": 1, "reason": "Alternative high-performing angle"}}
  ]
}}
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    text = response.text.strip()
    return json.loads(text)


def refine_viral_titles(
    context: str,
    user_instruction: str,
    api_key: str,
    history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Refine titles based on conversational user directions."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = model_name or TEXT_MODEL

    history_text = ""
    if history:
        for msg in history:
            history_text += f"\n{msg.get('role', 'user').upper()}: {msg.get('content', '')}"

    prompt = f"""You are a YouTube title strategist. Based on the video context and user feedback, generate 8 refined titles.

CONTEXT:
{context}

HISTORY:
{history_text}

USER INSTRUCTION:
{user_instruction}

OUTPUT JSON:
{{
  "titles": ["title1", "title2", ...]
}}
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    return json.loads(response.text.strip())


def generate_youtube_chapters(
    transcript_segments: List[Dict[str, Any]],
    api_key: str,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate timestamped chapters (e.g. 00:00 Intro, 02:15 The Secret) from transcript segments."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    model = model_name or TEXT_MODEL

    segments_snippet = []
    for s in transcript_segments[:200]:
        start = s.get("start", 0.0)
        text = s.get("text", "")
        mins = int(start // 60)
        secs = int(start % 60)
        segments_snippet.append(f"[{mins:02d}:{secs:02d}] {text}")

    prompt = f"""Based on these timestamped transcript segments, create a clean, compelling YouTube description with timestamped chapters.

SEGMENTS:
{chr(10).join(segments_snippet)}

RULES:
- Always start chapter 1 at 00:00 (e.g., 00:00 Intro)
- Create 4-10 concise, enticing chapter names
- Provide an engaging video description (2-4 paragraphs) and 5-10 relevant hashtags.

OUTPUT JSON:
{{
  "description": "Full YouTube description text...",
  "chapters": [
    {{"timestamp": "00:00", "title": "Introduction"}},
    ...
  ],
  "hashtags": ["#tag1", "#tag2", ...]
}}
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    return json.loads(response.text.strip())


IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL") or "imagen-3.0-generate-002"


def generate_youtube_thumbnail(
    title: str,
    output_dir: str,
    api_key: str,
    face_image_path: Optional[str] = None,
    bg_image_path: Optional[str] = None,
    extra_prompt: str = "",
    video_context: str = "",
    aspect_ratio: str = "16:9",
    model: Optional[str] = None,
) -> str:
    """Generate a high-CTR thumbnail or social cover image using Nano Banana / Gemini / Imagen.

    Supports aspect ratios: "16:9" (YouTube / Landscape), "9:16" (Reels / Shorts / TikTok), "1:1" (Square / Feed).
    """
    import base64
    import time
    from google import genai
    from google.genai import types

    valid_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4"}
    norm_ratio = aspect_ratio.strip() if aspect_ratio else "16:9"
    if norm_ratio not in valid_ratios:
        norm_ratio = "16:9"

    client = genai.Client(api_key=api_key)
    os.makedirs(output_dir, exist_ok=True)
    target_model = model or IMAGE_MODEL

    ratio_guide = {
        "16:9": "Horizontal (16:9 aspect ratio) composition for YouTube thumbnail. Large subject, punchy background.",
        "9:16": "Vertical (9:16 aspect ratio) cover image for TikTok, Instagram Reels, and YouTube Shorts. Centered focal subject.",
        "1:1": "Square (1:1 aspect ratio) cover image for Instagram Feed or social profile post. Balanced symmetrical composition.",
    }.get(norm_ratio, f"{norm_ratio} aspect ratio composition.")

    text_prompt = f"""Generate an eye-catching, high-CTR social cover thumbnail image ({norm_ratio} aspect ratio).
VIDEO TITLE: "{title}"
CONTEXT: {video_context}
USER INSTRUCTIONS: {extra_prompt}

COMPOSITION GUIDELINES:
- {ratio_guide}
- Punchy, high contrast, vibrant colors, clean lighting.
- Clear dynamic composition with strong visual hierarchy.
- Large focal subject with expressive emotion.
"""

    unique_ts = f"{int(time.time())}_{os.getpid()}"
    out_file = os.path.join(output_dir, f"thumbnail_{unique_ts}.png")

    # If an imagen model is requested and no image inputs were provided, use generate_images
    if target_model.startswith("imagen") and not (face_image_path or bg_image_path):
        try:
            result = client.models.generate_images(
                model=target_model,
                prompt=text_prompt,
                config=dict(
                    number_of_images=1,
                    aspect_ratio=norm_ratio,
                    output_mime_type="image/png",
                ),
            )
            if hasattr(result, "generated_images") and result.generated_images:
                img_obj = result.generated_images[0]
                img_bytes = getattr(getattr(img_obj, "image", None), "image_bytes", None)
                if img_bytes:
                    with open(out_file, "wb") as f:
                        f.write(img_bytes)
                    return out_file
        except Exception:
            # Fallback to multimodal content generation if generate_images encounters model restriction
            target_model = "gemini-2.5-flash"

    # Multimodal image generation path (Gemini native / Nano Banana with face/bg blending)
    prompt_parts = []
    if face_image_path and os.path.exists(face_image_path):
        prompt_parts.append(Image.open(face_image_path))
    if bg_image_path and os.path.exists(bg_image_path):
        prompt_parts.append(Image.open(bg_image_path))
    prompt_parts.append(text_prompt)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash" if target_model.startswith("imagen") else target_model,
            contents=prompt_parts,
        )
        if hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    with open(out_file, "wb") as f:
                        f.write(base64.b64decode(part.inline_data.data))
                    return out_file
    except Exception as exc:
        raise RuntimeError(f"AI Thumbnail generation failed: {exc}") from exc

    return out_file

