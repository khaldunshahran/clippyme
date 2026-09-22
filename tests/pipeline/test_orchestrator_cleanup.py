from pathlib import Path

from clippyme.pipeline.orchestrator import _cleanup_completed


class _State:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = str(checkpoint_dir)


def test_completed_cleanup_keeps_reframe_source_slice(tmp_path, monkeypatch):
    source_slice = tmp_path / "source_clip.mp4"
    source_slice.write_bytes(b"source")
    checkpoint = tmp_path / ".clippyme_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "transcript.json").write_text("{}", encoding="utf-8")
    input_video = tmp_path / "upload.mp4"
    input_video.write_bytes(b"input")
    monkeypatch.delenv("CLIPPYME_KEEP_CHECKPOINTS", raising=False)

    _cleanup_completed(
        output_dir=str(tmp_path),
        state=_State(checkpoint),
        input_video=str(input_video),
        is_url=False,
        keep_original=False,
        all_clips_ready=True,
    )

    assert source_slice.exists()
    assert input_video.exists()
    assert not checkpoint.exists()


def test_url_source_removed_only_after_completed_cleanup(tmp_path, monkeypatch):
    input_video = tmp_path / "download.mp4"
    input_video.write_bytes(b"input")
    checkpoint = tmp_path / ".clippyme_checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("CLIPPYME_KEEP_CHECKPOINTS", "1")

    _cleanup_completed(
        output_dir=str(tmp_path),
        state=_State(checkpoint),
        input_video=str(input_video),
        is_url=True,
        keep_original=False,
        all_clips_ready=True,
    )

    assert not input_video.exists()
    assert checkpoint.exists()


def test_partial_success_keeps_checkpoint_for_resume(tmp_path, monkeypatch):
    checkpoint = tmp_path / ".clippyme_checkpoint"
    checkpoint.mkdir()
    input_video = tmp_path / "upload.mp4"
    input_video.write_bytes(b"input")
    monkeypatch.delenv("CLIPPYME_KEEP_CHECKPOINTS", raising=False)

    _cleanup_completed(
        output_dir=str(tmp_path),
        state=_State(checkpoint),
        input_video=str(input_video),
        is_url=False,
        keep_original=False,
        all_clips_ready=False,
    )

    assert Path(checkpoint).exists()


def test_relink_source_ignores_viral_clips(tmp_path):
    from clippyme.pipeline.run_ops import find_source_video_candidate

    # Viral clip from a prior pass (does not start with 'clip_')
    viral_clip = tmp_path / "Epstein's brother reveals the truth_clip_1.mp4"
    viral_clip.write_bytes(b"V" * 40_000)

    # Viral clip's preserved source slice
    source_slice = tmp_path / "source_Epstein's brother reveals the truth_clip_1.mp4"
    source_slice.write_bytes(b"S" * 45_000)

    # The actual full source video
    true_source = tmp_path / "Q&A With An Ex-CIA Agent.mp4"
    true_source.write_bytes(b"ORIGINAL_LONG_VIDEO" * 5000)

    candidate = find_source_video_candidate(str(tmp_path))
    assert candidate == str(true_source)
