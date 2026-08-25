"""Actor and Avatar portrait generator for AI Shorts."""
import os
import httpx
from typing import Optional, Dict, Any

FAL_QUEUE_BASE = "https://queue.fal.run"


def generate_actor_flux(
    prompt: str,
    fal_key: str,
    output_path: str,
    aspect_ratio: str = "9:16",
) -> str:
    """Generate an AI actor portrait using Flux Pro via fal.ai."""
    url = f"{FAL_QUEUE_BASE}/fal-ai/flux-pro/v1.1"
    headers = {
        "Authorization": f"Key {fal_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": f"Professional UGC smartphone portrait video still, eye level, natural lighting, looking at camera: {prompt}",
        "image_size": "portrait_16_9" if aspect_ratio == "9:16" else "square_hd",
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Handle async queue polling if returned
        image_url = None
        if "images" in data and data["images"]:
            image_url = data["images"][0]["url"]
        elif "request_id" in data:
            req_id = data["request_id"]
            status_url = f"{FAL_QUEUE_BASE}/fal-ai/flux-pro/requests/{req_id}"
            import time
            for _ in range(30):
                time.sleep(2)
                s_resp = client.get(status_url, headers=headers)
                if s_resp.status_code == 200:
                    s_data = s_resp.json()
                    if "images" in s_data and s_data["images"]:
                        image_url = s_data["images"][0]["url"]
                        break

        if not image_url:
            raise RuntimeError("Failed to retrieve generated image from fal.ai")

        img_resp = client.get(image_url)
        img_resp.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_resp.content)

    return output_path
