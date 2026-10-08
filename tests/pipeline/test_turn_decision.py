"""Host tests for SpeakerTracker.decide() — the additive Task 12 decision output.

reframe_track is pure (no cv2/torch/mediapipe), so these run on the dev host.
Pins: decide() exposes chosen subject id(s), box, evidence components,
confidence, ambiguity and reason; decide().box == get_target() on identical
streams (the legacy API is preserved exactly); size alone cannot switch;
switching requires sustained mouth-motion evidence plus a margin; weak
evidence yields group/wide or held-safe framing.
"""
import pytest

from clippyme.pipeline.reframe_track import (
    SpeakerTracker,
    SubjectDecision,
    TrackEvidence,
)


def _face(x, mar, score=None):
    return {"box": [x, 100, 100, 100], "score": score or 100 * 100, "mar": mar}


def _stream(tracker, method, frames, faces_fn, width=1920):
    out = []
    for frame in frames:
        cands = faces_fn(frame)
        if method == "decide":
            out.append(tracker.decide([dict(c) for c in cands], frame, width))
        else:
            out.append(tracker.get_target([dict(c) for c in cands], frame, width))
    return out


class TestDecideBasics:
    def test_decision_shape(self):
        st = SpeakerTracker(cooldown_frames=5)
        d = st.decide([_face(200, 0.5)], 0, 1920)
        assert isinstance(d, SubjectDecision)
        assert len(d.subject_ids) == 1
        assert d.box == [200, 100, 100, 100]
        assert set(d.evidence) == {"size", "mouth", "sustained",
                                   "runner_up_sustained", "margin"}
        assert all(v >= 0 and v == v for v in d.evidence.values())  # finite
        assert 0.0 <= d.confidence <= 1.0
        assert 0.0 <= d.ambiguity <= 1.0
        assert d.reason == "initial-acquire"
        assert len(d.tracks) == 1 and isinstance(d.tracks[0], TrackEvidence)

    def test_no_candidates_held_safe(self):
        st = SpeakerTracker(cooldown_frames=5)
        d = st.decide([], 10, 1920)
        assert d.subject_ids == () and d.box is None
        assert d.confidence == 0.0 and d.ambiguity == 1.0
        assert d.reason == "no-candidates"

    def test_parity_with_get_target(self):
        # decide().box equals get_target() on identical candidate streams
        # wherever the weak-evidence guard does not trigger (the guard is an
        # intentional, documented divergence: size alone never crowns).
        def faces(frame):
            m1 = 0.1 if frame % 2 == 0 else 0.6
            m2 = 0.6 if frame % 3 == 0 else 0.1
            return [_face(200, m1), _face(1500, m2)]
        a = SpeakerTracker(cooldown_frames=5)
        b = SpeakerTracker(cooldown_frames=5)
        # Warm up the MAR history so mouth evidence is real, not the default.
        for frame in range(10):
            a.get_target([dict(c) for c in faces(frame)], frame, 1920)
            b.decide([dict(c) for c in faces(frame)], frame, 1920)
        for frame in range(10, 80):
            tbox = a.get_target([dict(c) for c in faces(frame)], frame, 1920)
            dec = b.decide([dict(c) for c in faces(frame)], frame, 1920)
            if dec.reason not in ("weak-evidence-group", "weak-evidence-hold"):
                assert (list(tbox) if tbox else None) == \
                       (list(dec.box) if dec.box else None), frame

    def test_track_ids_anonymous_ints(self):
        st = SpeakerTracker(cooldown_frames=5)
        d = st.decide([_face(200, 0.5), _face(1500, 0.5)], 0, 1920)
        for tid in d.subject_ids:
            assert isinstance(tid, int)
        for tr in d.tracks:
            assert isinstance(tr.track_id, int)


class TestSizeAloneCannotSwitch:
    def test_bigger_silent_face_cannot_steal(self):
        # Equal mouth evidence (both silent), challenger 4x the area: no
        # single face may be crowned on size alone — the decision must stay
        # grouped (or held-safe), never a lone size-winner.
        st = SpeakerTracker(cooldown_frames=5)
        def faces(frame):
            return [_face(200, 0.25, score=100 * 100),
                    _face(1500, 0.25, score=200 * 200)]
        for frame in range(60):
            d = st.decide(faces(frame), frame, 1920)
            assert len(d.subject_ids) != 1, \
                f"frame {frame}: size alone crowned {d.subject_ids}"
            assert d.reason in ("ambiguous-dialogue-two-subject",
                               "weak-evidence-group", "weak-evidence-hold",
                               "no-candidates")

    def test_switch_requires_sustained_mouth_evidence_plus_margin(self):
        st = SpeakerTracker(cooldown_frames=5)
        def faces(frame, talker):
            m = lambda x: (0.15 if frame % 2 == 0 else 0.65) if x == talker else 0.25
            return [_face(200, m(200)), _face(1500, m(1500))]
        # A talks first and locks.
        for frame in range(30):
            d = st.decide(faces(frame, 200), frame, 1920)
        assert d.box[0] == 200
        # B takes over with strong sustained mouth motion -> switch accepted.
        switched = None
        for frame in range(30, 90):
            d = st.decide(faces(frame, 1500), frame, 1920)
            if d.box[0] == 1500:
                switched = d
                break
        assert switched is not None, "sustained mouth evidence must switch"
        assert switched.reason == "turn-switch-sustained-evidence"
        assert switched.evidence["margin"] >= 1.0
        assert switched.confidence > 0.5

    def test_brief_mouth_blip_does_not_switch(self):
        st = SpeakerTracker(cooldown_frames=5)
        for frame in range(30):
            st.decide([_face(200, 0.6 if frame % 2 else 0.15),
                       _face(1500, 0.25)], frame, 1920)
        locked = st.active_speaker_id
        # Two frames of challenger mouth motion: not sustained -> hold. (The
        # reason is "speaker-locked" while the incumbent still ranks first.)
        for frame in range(30, 32):
            d = st.decide([_face(200, 0.25), _face(1500, 0.9)], frame, 1920)
            assert d.subject_ids == (locked,)
            assert d.reason in ("incumbent-held-weak-challenger", "speaker-locked")


class TestAmbiguityAndGroup:
    def test_ambiguous_dialogue_group_crop(self):
        # Both faces talk with near-equal sustained evidence -> the pair crop
        # wins with two subject ids and high ambiguity.
        st = SpeakerTracker(cooldown_frames=5)
        decided = None
        for frame in range(40):
            m = 0.15 if frame % 2 == 0 else 0.65
            d = st.decide([_face(200, m), _face(1500, m)], frame, 1920)
            if len(d.subject_ids) == 2:
                decided = d
        assert decided is not None
        assert decided.reason == "ambiguous-dialogue-two-subject"
        assert decided.ambiguity > 0.3
        # The union box spans both faces.
        x, y, w, h = decided.box
        assert x <= 200 and x + w >= 1600

    def test_weak_evidence_high_ambiguity(self):
        st = SpeakerTracker(cooldown_frames=5)
        d = st.decide([_face(200, 0.25), _face(1500, 0.25)], 0, 1920)
        # First frame acquires, but confidence stays low on weak evidence.
        assert d.confidence < 0.7 or d.ambiguity > 0.3


class TestCooldown:
    def test_switch_inside_cooldown_held(self):
        st = SpeakerTracker(cooldown_frames=1000)
        for frame in range(10):
            st.decide([_face(200, 0.6 if frame % 2 else 0.15),
                       _face(1500, 0.25)], frame, 1920)
        locked = st.active_speaker_id
        d = st.decide([_face(200, 0.25), _face(1500, 0.9)], 11, 1920)
        assert d.subject_ids == (locked,)
        assert d.reason in ("incumbent-held-weak-challenger", "speaker-locked")

    def test_no_active_during_cooldown_held_safe(self):
        st = SpeakerTracker(cooldown_frames=1000)
        st.decide([_face(200, 0.5)], 0, 1920)
        st.active_speaker_id = 999  # simulate a lost active speaker
        st.last_switch_frame = 0
        d = st.decide([_face(1500, 0.9)], 10, 1920)
        assert d.box is None and d.subject_ids == ()
        assert d.reason == "cooldown-hold-no-active"
