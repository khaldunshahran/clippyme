"""Modal.com Serverless GPU Worker for ClippyMe.

Deploys the full clipping pipeline to Nvidia T4 GPUs on Modal.
Runs on-demand with per-second billing ($30/month recurring free credit
= up to 3,000 minutes of GPU compute free every month).

Deploy to your Modal account:
    pip install modal
    modal setup
    modal deploy deploy/modal_pipeline.py
"""
import os
import subprocess
import modal

app = modal.App("clippyme-gpu-worker")

# Build container with full CV/ML and FFmpeg stack
clippyme_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "ffmpeg",
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "fonts-montserrat",
    )
    .pip_install(
        "numpy==2.4.6",
        "torch",
        "opencv-python-headless",
        "ultralytics",
        "faster-whisper",
        "scenedetect[opencv]",
        "yt-dlp>=2026.8.19",
        "google-genai>=1.70.0",
        "python-dotenv",
        "pydantic",
        "fastapi",
        "requests",
        "json-repair",
        "tqdm",
    )
)

# Persistent volume for checkpoints and cached whisper models
cache_volume = modal.Volume.from_name("clippyme-cache-vol", create_if_missing=True)


@app.function(
    image=clippyme_image,
    gpu="T4",
    timeout=900,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_name("clippyme-secrets", required_keys=["GEMINI_API_KEY"])],
)
def run_clipping_job(job_payload: dict):
    """Execute the clipping pipeline inside a GPU container on Modal."""
    job_id = job_payload.get("job_id")
    cmd = job_payload.get("cmd", [])
    env = dict(os.environ)
    env.update(job_payload.get("env", {}))

    print(f"Starting GPU processing for job {job_id}")
    print(f"Command: {' '.join(cmd)}")

    # Execute pipeline process
    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
    )

    print("STDOUT:\n", proc.stdout[-2000:] if len(proc.stdout) > 2000 else proc.stdout)
    if proc.returncode != 0:
        print("STDERR:\n", proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr)
        raise RuntimeError(f"Clipping failed on GPU worker with exit code {proc.returncode}")

    return {
        "job_id": job_id,
        "status": "completed",
        "exit_code": proc.returncode,
    }
