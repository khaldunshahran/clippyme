"""Host tests for pipeline.run_ops — pure entrypoint helpers.

These pin logic that previously lived inline in main.py's __main__ block,
where no unit test could reach it.
"""
import json
import os
import shutil
import subprocess

import pytest

from clippyme.pipeline.run_ops import (
    build_cut_command,
    build_vfr_normalization_command,
    clip_output_basename,
    extract_youtube_video_id,
    find_source_video_candidate,
    is_clip_artifact,
    is_same_video_source,
    resolve_output_dir,
)


# --- resolve_output_dir -------------------------------------------------------

def test_none_returns_default():
    assert resolve_output_dir(None, default="/d") == "/d"
    assert resolve_output_dir("", default="/d") == "/d"


def test_video_suffix_means_file_use_its_dirname(tmp_path):
    target = str(tmp_path / "out" / "final.mp4")
    assert resolve_output_dir(target, default="/d") == str(tmp_path / "out")


def test_video_suffix_without_dirname_falls_back_to_default():
    assert resolve_output_dir("final.mkv", default="/d") == "/d"


def test_new_directory_is_created_and_returned(tmp_path):
    # Regression: the old logic os.path.dirname'd a NEW directory path,
    # landing the output one level above the intended dir.
    target = str(tmp_path / "does-not-exist-yet")
    assert resolve_output_dir(target, default="/d") == target
    assert os.path.isdir(target)


def test_existing_directory_passthrough(tmp_path):
    assert resolve_output_dir(str(tmp_path), default="/d") == str(tmp_path)


# --- build_cut_command ---------------------------------------------------------

def test_cut_command_shape_and_precision():
    cmd = build_cut_command("/in/video.mp4", 12.3456, 47.9, "/out/source_clip.mp4")
    assert cmd[0] == "ffmpeg" and cmd[1] == "-y"
    # Fast input seek: -ss must come BEFORE -i.
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "12.346"          # 3-decimal rounding
    assert cmd[cmd.index("-t") + 1] == f"{47.9 - 12.3456:.3f}"
    assert cmd[-1] == "/out/source_clip.mp4"
    # Stream-copy contract: no re-encode flags ride along.
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-vsync" not in cmd
    assert "libx264" not in cmd


def test_cut_command_is_stream_copy_not_reencode():
    """build_cut_command must NOT re-encode the source slice.

    The slice is a fast, lossless stream copy; the old test asserting shared
    x264 settings encoded the pre-stream-copy contract.
    """
    cmd = build_cut_command("/in.mp4", 0, 10, "/out.mp4")
    assert cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "libx264" not in cmd
    assert "-crf" not in cmd

def test_stream_copy_cut_keyframe_granularity(tmp_path):
    """Quantify stream-copy seek granularity on a real file.

    build_cut_command seeks with ``-ss`` BEFORE ``-i`` and copies the stream,
    so the slice really starts at the keyframe at/before the requested start
    (up to ~one GOP early). This measures that offset instead of theorizing:
    it runs the command's seek flags WITHOUT the ``-avoid_negative_ts``
    rebase and reads the first packet's DTS via ffprobe, so
    ``requested_start - actual_start`` IS the keyframe offset. Skips when
    ffmpeg/ffprobe are unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe not available")
    src = str(tmp_path / "src.mp4")
    # 30fps, GOP=60 -> keyframes every 2s: at 0, 2, 4, 6, 8.
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi", "-i",
         "testsrc=duration=10:size=320x240:rate=30",
         "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p", src],
        check=True, capture_output=True)
    start, length = 3.5, 4.0  # 3.5s is not a keyframe
    probe_out = str(tmp_path / "probe.mp4")
    probe_cmd = build_cut_command(src, start, start + length, probe_out)
    # Drop the timestamp rebase so ffprobe reveals the true seek point.
    probe_cmd = [a for a in probe_cmd
                 if a not in ("-avoid_negative_ts", "make_zero")]
    subprocess.run(probe_cmd, check=True, capture_output=True)
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "packet=dts_time", "-of", "csv=p=0",
         "-read_intervals", "%+#1", probe_out],
        check=True, capture_output=True, text=True).stdout.strip().splitlines()
    first_dts = float(out[0])
    actual_start = start + first_dts  # first_dts <= 0: keyframe at/before start
    offset = start - actual_start
    print(f"\nkeyframe-granularity: requested={start}, "
          f"actual={actual_start:.3f}, offset={offset:.3f}s (GOP=2.0s)")
    assert 0.0 <= offset < 2.0, f"cut started {offset:.3f}s before request"
    # And the real command (with rebase): the slice ENDS at the requested end
    # (seek target + length), so the persisted duration is the requested
    # length PLUS the head offset -- the clip is longer than asked, starting
    # early. This is the documented stream-copy contract, not a bug in the
    # test: word timings mapped against the requested start therefore run up
    # to ~one GOP early relative to the slice's actual audio.
    real_out = str(tmp_path / "cut.mp4")
    subprocess.run(build_cut_command(src, start, start + length, real_out),
                   check=True, capture_output=True)
    dur = float(subprocess.run(
        [ffprobe, "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "format=duration", "-of", "csv=p=0", real_out],
        check=True, capture_output=True, text=True).stdout.strip())
    assert abs(dur - (length + offset)) < 0.15, (
        f"duration {dur} != requested {length} + offset {offset:.3f}")


def test_vfr_normalization_command_uses_shared_encode_policy(monkeypatch):
    monkeypatch.setenv("CLIPPYME_X264_CRF", "17")
    monkeypatch.setenv("CLIPPYME_X264_PRESET", "slow")

    command = build_vfr_normalization_command("input.mp4", "normalized.mp4")

    assert command == [
        "ffmpeg", "-y", "-i", "input.mp4",
        "-vsync", "cfr",
        "-c:v", "libx264", "-preset", "slow", "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy", "normalized.mp4",
    ]


# --- clip_output_basename -----------------------------------------------------

def test_basename_uses_sanitized_title():
    assert clip_output_basename("My Viral Clip", 0, "source") == "My Viral Clip_clip_1"


def test_forbidden_chars_stripped():
    assert clip_output_basename('a<b>c:d"e/f\\g|h?i*j#k%l', 0, "source") == "abcdefghijkl_clip_1"


def test_control_chars_stripped():
    assert clip_output_basename("hello\x00\x1fworld", 0, "source") == "helloworld_clip_1"


def test_leading_trailing_dots_and_spaces_stripped():
    assert clip_output_basename("  ..title..  ", 0, "source") == "title_clip_1"


def test_whitespace_runs_collapsed_but_spaces_kept():
    # tabs/newlines are ASCII control chars (0-31) and get stripped outright;
    # only literal space runs collapse to a single space.
    assert clip_output_basename("a   b    c", 0, "source") == "a b c_clip_1"


def test_reserved_names_fall_back():
    for name in ("CON", "con", "PRN", "AUX", "NUL", "COM1", "com9", "LPT1", "lpt9"):
        assert clip_output_basename(name, 2, "source") == "source_clip_3"


def test_empty_or_none_falls_back():
    assert clip_output_basename(None, 4, "source") == "source_clip_5"
    assert clip_output_basename("", 4, "source") == "source_clip_5"
    assert clip_output_basename("   ", 4, "source") == "source_clip_5"


def test_only_forbidden_chars_falls_back():
    assert clip_output_basename('<>:"/\\|?*', 0, "source") == "source_clip_1"


def test_long_title_truncated_at_boundary():
    title = "word " * 40  # 200 chars, well over max_len
    result = clip_output_basename(title, 1, "source", max_len=80)
    stem = result.removesuffix("_clip_2")
    assert len(stem) <= 80
    assert result.endswith("_clip_2")
    # cut on a word boundary, not mid-word
    assert stem == stem.strip()
    assert "word" in stem and not stem.endswith("wor")


def test_suffix_always_matches_index():
    assert clip_output_basename("Title", 0, "source").endswith("_clip_1")
    assert clip_output_basename("Title", 9, "source").endswith("_clip_10")
    assert clip_output_basename(None, 9, "source").endswith("_clip_10")


def test_unicode_preserved():
    assert clip_output_basename("Café Émoji 🎬 Clip", 0, "source") == "Café Émoji 🎬 Clip_clip_1"


# --- is_clip_artifact & find_source_video_candidate --------------------------

def test_is_clip_artifact():
    assert is_clip_artifact("source_clip.mp4")
    assert is_clip_artifact("clip_1.mp4")
    assert is_clip_artifact("clip_10.mp4")
    assert is_clip_artifact("Epstein's brother reveals the truth_clip_1.mp4")
    assert is_clip_artifact("source_Epstein's brother reveals the truth_clip_1.mp4")
    assert is_clip_artifact("composed_clip_1.mp4")
    assert is_clip_artifact("composed_Viral Title_clip_2.mp4")
    assert is_clip_artifact("temp.render.tmp.mp4")
    assert is_clip_artifact("test.tmp.mp4")
    assert is_clip_artifact("file.tmp")

    # The actual source videos must NOT be considered clip artifacts
    assert not is_clip_artifact("Original Long Video.mp4")
    assert not is_clip_artifact("trimmed_Original Long Video.mp4")
    assert not is_clip_artifact("＂Would I Run For President？＂ ｜ Q&A With An Ex-CIA Agent.mp4")


def test_find_source_video_candidate_ignores_viral_clips(tmp_path):
    # Simulate a job directory with rendered viral clips and the true source
    clip_1 = tmp_path / "Epstein's brother reveals the truth_clip_1.mp4"
    clip_1.write_bytes(b"A" * 50_000)

    src_clip_1 = tmp_path / "source_Epstein's brother reveals the truth_clip_1.mp4"
    src_clip_1.write_bytes(b"B" * 60_000)

    clip_2 = tmp_path / "How the CIA secretly hides operations_clip_2.mp4"
    clip_2.write_bytes(b"C" * 40_000)

    true_source = tmp_path / "＂Would I Run For President？＂ ｜ Q&A.mp4"
    true_source.write_bytes(b"S" * 200_000)

    candidate = find_source_video_candidate(str(tmp_path))
    assert candidate == str(true_source)


def test_find_source_video_candidate_prefers_trimmed_source(tmp_path):
    untrimmed = tmp_path / "source_video.mp4"
    untrimmed.write_bytes(b"U" * 500_000)

    trimmed = tmp_path / "trimmed_source_video.mp4"
    trimmed.write_bytes(b"T" * 450_000)

    clip = tmp_path / "Viral Title_clip_1.mp4"
    clip.write_bytes(b"C" * 30_000)

    candidate = find_source_video_candidate(str(tmp_path))
    assert candidate == str(trimmed)


def test_find_source_video_candidate_returns_none_if_only_clips(tmp_path):
    clip = tmp_path / "Viral Title_clip_1.mp4"
    clip.write_bytes(b"C" * 30_000)
    assert find_source_video_candidate(str(tmp_path)) is None


# --- extract_youtube_video_id & is_same_video_source -------------------------

def test_extract_youtube_video_id():
    assert extract_youtube_video_id("https://youtu.be/j09ZaBP3Bu8?is=VaH-WbQFJJqfwKav") == "j09ZaBP3Bu8"
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=j09ZaBP3Bu8") == "j09ZaBP3Bu8"
    assert extract_youtube_video_id("https://www.youtube.com/live/bEMsJi2P6PA?si=123") == "bEMsJi2P6PA"
    assert extract_youtube_video_id("https://www.youtube.com/shorts/Qtl8lJwbd4g") == "Qtl8lJwbd4g"
    assert extract_youtube_video_id("https://m.youtube.com/watch?v=j09ZaBP3Bu8") == "j09ZaBP3Bu8"
    assert extract_youtube_video_id("https://notyoutube.com/video") is None
    assert extract_youtube_video_id(None) is None


def test_is_same_video_source():
    # Exactly same string
    assert is_same_video_source("https://example.com/stream", "https://example.com/stream")

    # YouTube URL variant with query params vs canonical
    url1 = "https://youtu.be/j09ZaBP3Bu8?is=VaH-WbQFJJqfwKav"
    url2 = "https://www.youtube.com/watch?v=j09ZaBP3Bu8"
    assert is_same_video_source(url1, url2)

    # Different videos
    url3 = "https://www.youtube.com/watch?v=A0MbO-zsN1w"
    assert not is_same_video_source(url1, url3)

    # None handling
    assert not is_same_video_source(None, url1)
    assert not is_same_video_source(url1, None)
