"""Phase 3C: diarization-driven multi-speaker layouts.

Covers: word->turn merging, talk-time shares, layout selection, the
DiarizationGuide correlation, the switch schedule, and the SpeakerTracker
guide integration with synthetic two-face candidates (no CV/ML needed).
"""
import pytest

from clippyme.pipeline.layouts.speaker_switch import (
    DiarizationGuide,
    build_switch_schedule,
    select_multi_speaker_layout,
    speaker_turns_from_words,
    talk_time_shares,
)
# NOTE: SpeakerTracker (reframe_track) needs cv2/MediaPipe; imported lazily
# inside _run_tracker so the pure diarization tests run anywhere.


def _word(start, end, speaker, text="w"):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


def _two_speaker_words(total_s=20.0, a_share=0.5):
    words, t = [], 0.0
    a_end = total_s * a_share
    while t < total_s - 0.5:
        words.append(_word(t, t + 0.5, "A" if t < a_end else "B"))
        t += 0.5
    return words


# --- word -> turns -----------------------------------------------------------

def test_turns_merge_same_speaker():
    words = [_word(0, 1, "A"), _word(1, 2, "A"), _word(2.1, 3.2, "B")]
    turns = speaker_turns_from_words(words)
    assert [(t["speaker"], t["start"], t["end"]) for t in turns] == [
        ("A", 0, 2), ("B", 2.1, 3.2)]


def test_sub_second_turn_absorbed_into_previous():
    words = [_word(0, 1, "A"), _word(1, 2, "A"), _word(2.1, 3.0, "B")]
    turns = speaker_turns_from_words(words, min_turn_s=1.0)
    assert [(t["speaker"], t["start"], t["end"]) for t in turns] == [
        ("A", 0, 3.0)]


def test_micro_turn_absorbed():
    words = [_word(0, 4, "A"), _word(4, 4.5, "B"), _word(4.5, 8, "A")]
    turns = speaker_turns_from_words(words, min_turn_s=1.0)
    assert len(turns) == 1 and turns[0]["speaker"] == "A"


def test_turns_offset():
    words = [_word(10, 12, "A")]
    turns = speaker_turns_from_words(words, offset=-10.0)
    assert turns[0]["start"] == pytest.approx(0.0)
    assert turns[0]["end"] == pytest.approx(2.0)


def test_no_speakers_no_turns():
    words = [{"start": 0.0, "end": 1.0, "text": "hi"}]
    assert speaker_turns_from_words(words) == []
    assert talk_time_shares(words) == {}


# --- shares + selection ------------------------------------------------------

def test_balanced_dialogue_selects_speaker_switch():
    words = _two_speaker_words(20.0, 0.5)
    shares = talk_time_shares(words)
    assert shares["A"]["share"] == pytest.approx(0.5, abs=0.05)
    assert shares["B"]["share"] == pytest.approx(0.5, abs=0.05)
    decision = select_multi_speaker_layout(words)
    assert decision["layout"] == "speaker_switch"
    assert decision["diarization_confident"] is True
    assert set(decision["speakers"]) == {"A", "B"}


def test_single_speaker_stays_single():
    words = [_word(i, i + 0.5, "A") for i in range(20)]
    assert select_multi_speaker_layout(words)["layout"] == "single"


def test_dominant_speaker_stays_single():
    words = _two_speaker_words(20.0, 0.9)
    decision = select_multi_speaker_layout(words)
    assert decision["layout"] == "single"
    assert decision["reason"] == "below_significance"


def test_visual_split_hint_respected():
    words = _two_speaker_words(20.0, 0.5)
    decision = select_multi_speaker_layout(words, visual_hint="split")
    assert decision["layout"] == "split"
    # but the visual veto keeps the default when diarization is weak
    weak = [_word(i, i + 0.5, "A") for i in range(20)]
    assert select_multi_speaker_layout(weak, visual_hint="split")["layout"] == "single"


def test_too_little_speech_stays_single():
    words = [_word(0, 0.5, "A"), _word(1, 1.5, "B")]
    assert select_multi_speaker_layout(words, min_total_speech_s=4.0)["layout"] == "single"


# --- DiarizationGuide --------------------------------------------------------

def _guide_two_turns(fps=10.0):
    turns = [{"speaker": "A", "start": 0.0, "end": 3.0},
             {"speaker": "B", "start": 3.0, "end": 7.0}]
    return DiarizationGuide(turns, fps)


def test_guide_no_evidence_no_hint():
    guide = _guide_two_turns()
    assert guide.suggest(5) is None
    assert guide.face_for_label("A") is None


def test_guide_correlates_label_to_face():
    guide = _guide_two_turns()
    # during A's turn the tracker reports face 7 moving
    for f in range(20):
        guide.observe(7, 2.5, f)
    assert guide.face_for_label("A") == 7
    assert guide.suggest(10) == 7       # inside A's turn
    assert guide.suggest(50) is None    # inside B's turn: no evidence for B yet


def test_guide_requires_confident_motion():
    guide = _guide_two_turns()
    for f in range(20):
        guide.observe(7, 0.3, f)  # barely moving -> not confident
    assert guide.face_for_label("A") is None


def test_guide_hysteresis():
    guide = DiarizationGuide(
        [{"speaker": "A", "start": 0.0, "end": 1.0},
         {"speaker": "B", "start": 1.0, "end": 6.0}], fps=10.0)
    for f in range(10):
        guide.observe(7, 2.5, f)          # frames 0-9  -> t 0.0-0.9 (A's turn)
    for f in range(10, 30):
        guide.observe(9, 2.5, f)          # frames 10-29 -> t 1.0-2.9 (B's turn)
    # A's turn is 1.0s < hysteresis floor (2.5s) -> no hint even with evidence
    assert guide.suggest(5) is None
    # B's turn is long enough -> hint
    assert guide.suggest(25) == 9


# --- switch schedule ---------------------------------------------------------

def test_switch_schedule_no_flicker():
    turns = [{"speaker": "A", "start": 0.0, "end": 2.0},
             {"speaker": "B", "start": 2.0, "end": 2.4},   # micro turn
             {"speaker": "A", "start": 2.4, "end": 6.0}]
    sched = build_switch_schedule(turns)
    speakers = [seg["speaker"] for seg in sched]
    assert speakers == ["A"]  # micro B-turn absorbed, no flicker


def test_switch_schedule_keeps_real_turns():
    turns = [{"speaker": "A", "start": 0.0, "end": 3.0},
             {"speaker": "B", "start": 3.0, "end": 6.0}]
    sched = build_switch_schedule(turns)
    assert [seg["speaker"] for seg in sched] == ["A", "B"]
    assert sched[0]["switch_at_frame"] == 0
    assert sched[1]["switch_at_frame"] == 90  # 3s * 30fps


# --- SpeakerTracker + guide integration --------------------------------------

_FACE_A = [100.0, 100.0, 80.0, 80.0]
_FACE_B = [400.0, 100.0, 80.0, 80.0]


def _frame_candidates(mar_a, mar_b):
    return [
        {"box": list(_FACE_A), "score": 1.0, "mar": mar_a},
        {"box": list(_FACE_B), "score": 1.0, "mar": mar_b},
    ]


def _run_tracker(frames_mars, guide=None):
    from clippyme.pipeline.reframe_track import SpeakerTracker
    tracker = SpeakerTracker(cooldown_frames=5)
    boxes = []
    for f, (mar_a, mar_b) in enumerate(frames_mars):
        box = tracker.get_target(_frame_candidates(mar_a, mar_b), f, 640, guide=guide)
        boxes.append(tuple(box) if box else None)
    return boxes


def test_tracker_guide_follows_diarization_not_lips():
    """A keeps moving (laughing) while diarization says B talks: the guide
    must follow B, the legacy lip-motion path must follow A."""
    fps = 10.0
    frames = []
    for f in range(70):
        t = f / fps
        # A: brief laugh bursts during B's turn (motion 3.0 for 3 of 10
        #     frames), otherwise sustained talk during A's own turn
        if t < 3.0:
            mar_a = 0.3 if f % 2 == 0 else 0.7          # A talking, motion 3.0
            mar_b = 0.5                                 # B still
        else:
            burst = (f % 10) < 3
            mar_a = (0.3 if f % 2 == 0 else 0.7) if burst else 0.5
            mar_b = 0.4 if f % 2 == 0 else 0.6          # B talking, motion 2.0
        frames.append((mar_a, mar_b))

    guide = _guide_two_turns(fps)
    guided = _run_tracker(frames, guide=guide)
    legacy = _run_tracker(frames, guide=None)

    # late in B's turn the guide has enough evidence -> follows B's face
    assert all(b == tuple(_FACE_B) for b in guided[55:65])
    # legacy lip-motion path follows the louder mover (A, the laugher)
    assert all(b == tuple(_FACE_A) for b in legacy[55:65])


def test_tracker_default_path_unchanged_without_guide():
    """Single speaker, no guide: legacy behaviour picks the moving face."""
    frames = [(0.3 if f % 2 == 0 else 0.7, 0.5) for f in range(40)]
    boxes = _run_tracker(frames)
    assert all(b == tuple(_FACE_A) for b in boxes[20:30])
