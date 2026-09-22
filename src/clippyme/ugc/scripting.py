"""Viral script generation module for AI Shorts."""
import json
import os
from typing import Dict, Any, List, Optional


def generate_ugc_scripts(
    research_data: Dict[str, Any],
    gemini_key: str,
    target_language: str = "en",
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate viral short video scripts with segments, visual prompts, and voiceover text."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_key)
    model = model_name or os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"

    prompt = f"""You are a master viral video creator specializing in TikTok, IG Reels, and YouTube Shorts marketing videos.
Based on the following product research, create 3 distinct, high-converting 30-45 second video scripts.

RESEARCH DATA:
{json.dumps(research_data, indent=2)}

TARGET LANGUAGE: {target_language}

STRUCTURE FOR EACH SCRIPT:
- Hook (0-3s): High-contrast visual + shocking claim or painful question
- Agitation / Problem (3-12s): Show why current alternatives fail
- Solution / Demo (12-25s): Showcase the product solving the problem effortlessly
- Call to Action (25-35s): Clear, urgent next step

OUTPUT JSON FORMAT:
{{
  "scripts": [
    {{
      "title": "Angle Name (e.g., The $10k Mistake)",
      "hook_text_overlay": "STOP DOING THIS 🛑",
      "voice_tone": "Energetic / Confident / Relatable",
      "full_script_text": "Complete spoken voiceover text...",
      "estimated_duration_sec": 35,
      "segments": [
        {{
          "segment_type": "talking_head" | "b_roll",
          "voiceover": "First 3 seconds spoken text...",
          "visual_description": "Close up talking head expressing disbelief"
        }},
        ...
      ]
    }},
    ...
  ]
}}
"""

    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    data = json.loads(response.text.strip())
    return data.get("scripts", [])
