"""Generate reproducible synthetic demo fixtures.

Creates short moving audiovisual mp4s with accurate durations,
frozen manifest with fingerprints, and mixed reviews.
Run from repo root: python tests/fixtures/editorial_benchmark/generate_demo.py [output_dir]
"""
import json
import subprocess
import hashlib
import sys
from pathlib import Path

def get_fx_dir():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return Path("tests/fixtures/editorial_benchmark")


def make_video(path: Path, duration: float, with_audio: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"testsrc2=s=320x180:r=24:d={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=880:duration={duration}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main():
    FX = get_fx_dir()
    assets = FX / "assets"
    # Exact owned paths (never broad deletes).
    owned = [
        assets / "syn1.mp4", assets / "syn2.mp4",
        assets / "c1.mp4", assets / "c2.mp4", assets / "c3.mp4",
        assets / "invalid.mp4",
    ]
    for p in owned:
        if p.exists():
            p.unlink()

    # Valid: moving audiovisual, accurately declared.
    make_video(assets / "syn1.mp4", 3.0, with_audio=True)
    make_video(assets / "syn2.mp4", 3.0, with_audio=True)
    make_video(assets / "c1.mp4", 2.0, with_audio=True)
    make_video(assets / "c2.mp4", 2.0, with_audio=True)
    make_video(assets / "c3.mp4", 1.0, with_audio=True)
    # Invalid: static silent (for QA failure demonstration).
    make_video(assets / "invalid.mp4", 1.0, with_audio=False)

    manifest = {
        "version": 1,
        "asset_root": "assets",
        "sources": [
            {"id": "syn1", "file": "syn1.mp4", "duration": 3.0,
             "genre": "interview/podcast", "synthetic": True,
             "fingerprint": sha256(assets / "syn1.mp4")},
            {"id": "syn2", "file": "syn2.mp4", "duration": 3.0,
             "genre": "reaction/comedy", "synthetic": True,
             "fingerprint": sha256(assets / "syn2.mp4")},
            # Invalid QA demonstration: silent static clip.
            # Not in any run's source_ids, so excluded from human
            # editorial metrics; media_checks will flag it.
            {"id": "invalid_demo", "file": "invalid.mp4", "duration": 1.0,
             "genre": "interview/podcast", "synthetic": True,
             "fingerprint": sha256(assets / "invalid.mp4")},
        ],
        "runs": [
            {"run_id": "baseline", "commit": "abc123",
             "settings": {"preset": "base"},
             "source_ids": ["syn1", "syn2"],
             "source_fingerprints": {
                 "syn1": sha256(assets / "syn1.mp4"),
                 "syn2": sha256(assets / "syn2.mp4")},
             "clips": [
                 {"clip_id": "c1", "source_id": "syn1",
                  "start": 0.0, "end": 2.0, "moment_id": "m1",
                  "file": "c1.mp4",
                  "fingerprint": sha256(assets / "c1.mp4")},
                 {"clip_id": "c2", "source_id": "syn1",
                  "start": 0.0, "end": 2.0, "moment_id": "m1",
                  "file": "c2.mp4",
                  "fingerprint": sha256(assets / "c2.mp4")},
             ]},
            {"run_id": "candidate", "commit": "def456",
             "settings": {"preset": "improved"},
             "source_ids": ["syn1"],
             "source_fingerprints": {
                 "syn1": sha256(assets / "syn1.mp4")},
             "clips": [
                 {"clip_id": "c3", "source_id": "syn1",
                  "start": 0.0, "end": 1.0, "moment_id": "m2",
                  "file": "c3.mp4",
                  "fingerprint": sha256(assets / "c3.mp4")},
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
        {"clip_id": "c3", "accepted": True},
    ]}
    (FX / "reviews.json").write_text(json.dumps(reviews, indent=2))
    print(f"Demo fixtures generated in {FX}.")


if __name__ == "__main__":
    main()
