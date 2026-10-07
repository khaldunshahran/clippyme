"""Generate reproducible synthetic demo fixtures.

Creates accurate-duration mp4s, frozen manifest with fingerprints,
and mixed reviews. Run from repo root.
"""
import json
import subprocess
import hashlib
from pathlib import Path

FX = Path("tests/fixtures/editorial_benchmark")


def make_video(path: Path, duration: float, color: str = "blue"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"color={color}:s=160x90:r=10:d={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    # Clean old.
    for p in FX.glob("*.mp4"):
        p.unlink()
    # Sources: 60s each (as declared).
    make_video(FX / "assets" / "syn1.mp4", 60.0, "blue")
    make_video(FX / "assets" / "syn2.mp4", 60.0, "green")
    # Clips: accurate durations matching intervals.
    # c1: [10,25) = 15s, c2: [30,45) = 15s, c3: [0,10) = 10s
    make_video(FX / "assets" / "c1.mp4", 15.0, "red")
    make_video(FX / "assets" / "c2.mp4", 15.0, "red")
    make_video(FX / "assets" / "c3.mp4", 10.0, "red")

    manifest = {
        "version": 1,
        "asset_root": "assets",
        "sources": [
            {"id": "syn1", "file": "syn1.mp4", "duration": 60.0,
             "genre": "interview/podcast", "synthetic": True,
             "fingerprint": sha256(FX / "assets" / "syn1.mp4")},
            {"id": "syn2", "file": "syn2.mp4", "duration": 60.0,
             "genre": "reaction/comedy", "synthetic": True,
             "fingerprint": sha256(FX / "assets" / "syn2.mp4")},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc123",
             "settings": {"preset": "base"},
             "source_ids": ["syn1", "syn2"],
             "source_fingerprints": {
                 "syn1": sha256(FX / "assets" / "syn1.mp4"),
                 "syn2": sha256(FX / "assets" / "syn2.mp4")},
             "clips": [
                 {"clip_id": "c1", "source_id": "syn1",
                  "start": 10.0, "end": 25.0, "moment_id": "m1",
                  "file": "c1.mp4",
                  "fingerprint": sha256(FX / "assets" / "c1.mp4")},
                 {"clip_id": "c2", "source_id": "syn1",
                  "start": 30.0, "end": 45.0, "moment_id": "m1",
                  "file": "c2.mp4",
                  "fingerprint": sha256(FX / "assets" / "c2.mp4")},
             ]},
            {"run_id": "candidate", "commit": "def456",
             "settings": {"preset": "improved"},
             "source_ids": ["syn1"],
             "source_fingerprints": {
                 "syn1": sha256(FX / "assets" / "syn1.mp4")},
             "clips": [
                 {"clip_id": "c3", "source_id": "syn1",
                  "start": 0.0, "end": 10.0, "moment_id": "m2",
                  "file": "c3.mp4",
                  "fingerprint": sha256(FX / "assets" / "c3.mp4")},
             ]},
        ],
    }
    (FX / "manifest.json").write_text(json.dumps(manifest, indent=2))

    reviews = {"reviews": [
        {"clip_id": "c1", "accepted": True,
         "opening_understandable": True,
         "payoff_delivered": True, "ending_natural": False,
         "framing_errors": ["face_crop"],
         "hook_obstruction": False, "caption_obstruction": None},
        {"clip_id": "c2", "accepted": False,
         "opening_understandable": False},
        {"clip_id": "c3", "accepted": True, "moment_id": "m2"},
    ]}
    # Remove moment_id from review (it's in manifest).
    for r in reviews["reviews"]:
        r.pop("moment_id", None)
    (FX / "reviews.json").write_text(json.dumps(reviews, indent=2))
    print("Demo fixtures generated.")


if __name__ == "__main__":
    main()
