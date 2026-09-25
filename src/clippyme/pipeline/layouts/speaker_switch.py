"""Phase 3C: diarization-driven multi-speaker layouts.

Turns word-level diarization (``speaker`` labels preserved by
``cut_ops.flatten_words``) into layout decisions and a per-frame steering
signal for the existing ``SpeakerTracker``:

- :func:`speaker_turns_from_words` — merge words into speaker turns
  (micro-turns absorbed, small gaps bridged).
- :func:`talk_time_shares` — per-speaker talk-time share.
- :func:`select_multi_speaker_layout` — 'single' | 'split' | 'speaker_switch'.
  The default single-speaker path is untouched: multi-speaker is selected
  only when >= 2 speakers each hold significant talk time.
- :class:`DiarizationGuide` — correlates diarization speaker labels to
  visual face-track ids through mouth-motion evidence, then suggests the
  face to frame per video frame. The SpeakerTracker still eases the actual
  pan through ``SmoothedCameraman`` (no jump cuts); the guide only steers
  *which* face is the target. When unsure it returns None and the legacy
  lip-motion logic runs unchanged.
- :func:`build_switch_schedule` — deterministic, flicker-free switch plan.

The existing split-screen renderer (``layouts/split_layout.py``) is NOT
replaced: 'split' still requires suitable two-face geometry downstream;
'speaker_switch' is the active-speaker crop alternative for dialogue where
a stacked split would look wrong.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MIN_TURN_S = 1.0            # turns shorter than this are absorbed
TURN_GAP_TOLERANCE_S = 1.0  # same-speaker gap bridged below this
MIN_TOTAL_SPEECH_S = 4.0    # less than this -> not enough signal, 'single'
MIN_SPEAKER_SHARE = 0.25    # each significant speaker needs >= 25% talk time

MIN_FACE_EVIDENCE_FRAMES = 15   # frames of top-face evidence before a label maps
CONFIDENT_MEAN_MOTION = 1.0     # mean mouth-motion floor for a confident mapping
EVIDENCE_MOTION_FLOOR = 1.0     # per-frame top-face motion floor to record evidence
TURN_HYSTERESIS_S = 2.5         # turns shorter than this never steer (no flicker)


# --- words -> turns ------------------------------------------------------------

def speaker_turns_from_words(
    words: List[Dict[str, Any]],
    min_turn_s: float = MIN_TURN_S,
    gap_tolerance_s: float = TURN_GAP_TOLERANCE_S,
    offset: float = 0.0,
) -> List[Dict[str, Any]]:
    """Merge diarized words into speaker turns.

    Same-speaker words separated by <= ``gap_tolerance_s`` merge; turns
    shorter than ``min_turn_s`` are absorbed into the previous turn (a
    backchannel "yeah" must not yank the camera). ``offset`` shifts all
    times (e.g. clip_start subtraction for clip-relative turns).
    """
    turns: List[Dict[str, Any]] = []
    for w in words or []:
        speaker = w.get("speaker")
        if not speaker:
            continue
        try:
            start = float(w.get("start", 0) or 0) + offset
            end = float(w.get("end", 0) or 0) + offset
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if (turns and turns[-1]["speaker"] == speaker
                and start - turns[-1]["end"] <= gap_tolerance_s):
            turns[-1]["end"] = max(turns[-1]["end"], end)
        else:
            turns.append({"speaker": speaker, "start": start, "end": end})

    # absorb micro-turns into the previous turn (a backchannel "yeah" must
    # not become its own turn); a leading micro-turn has nothing to absorb
    # into, so it is dropped and the next turn anchors the timeline.
    out: List[Dict[str, Any]] = []
    for turn in turns:
        dur = turn["end"] - turn["start"]
        if out and dur < min_turn_s:
            out[-1]["end"] = max(out[-1]["end"], turn["end"])
        elif not out and dur < min_turn_s and len(turns) > 1:
            continue  # leading micro-turn: drop, next turn anchors
        else:
            out.append(dict(turn))
    # absorbing a micro-turn can leave adjacent same-speaker turns behind
    # (A ... "yeah" ... A) — merge them so the timeline stays clean
    merged_out: List[Dict[str, Any]] = []
    for turn in out:
        if (merged_out and merged_out[-1]["speaker"] == turn["speaker"]
                and turn["start"] - merged_out[-1]["end"] <= gap_tolerance_s):
            merged_out[-1]["end"] = max(merged_out[-1]["end"], turn["end"])
        else:
            merged_out.append(turn)
    return merged_out


# --- shares + layout selection ----------------------------------------------------

def talk_time_shares(words: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Per-speaker talk time and share of total speech."""
    totals: Dict[str, float] = defaultdict(float)
    for w in words or []:
        speaker = w.get("speaker")
        if not speaker:
            continue
        try:
            dur = float(w.get("end", 0) or 0) - float(w.get("start", 0) or 0)
        except (TypeError, ValueError):
            continue
        if dur > 0:
            totals[speaker] += dur
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {s: {"seconds": secs, "share": secs / total}
            for s, secs in sorted(totals.items(),
                                  key=lambda kv: kv[1], reverse=True)}


def select_multi_speaker_layout(
    words: List[Dict[str, Any]],
    visual_hint: Optional[str] = None,
    min_total_speech_s: float = MIN_TOTAL_SPEECH_S,
    min_speaker_share: float = MIN_SPEAKER_SHARE,
) -> Dict[str, Any]:
    """Decide the clip layout from diarization.

    Returns ``layout`` in {'single', 'split', 'speaker_switch'} plus the
    speakers, their shares, and the reason. Default (no/weak diarization)
    is ALWAYS 'single' — the existing path is preserved.

    ``visual_hint='split'``: the scene analysis already found suitable
    two-face geometry for the stacked split-screen renderer; diarization
    agreeing (>= 2 significant speakers) confirms it, otherwise the hint
    is vetoed and the layout stays 'single'.
    """
    shares = talk_time_shares(words)
    total = sum(v["seconds"] for v in shares.values())
    significant = [s for s, v in shares.items() if v["share"] >= min_speaker_share]

    def _single(reason: str) -> Dict[str, Any]:
        return {"layout": "single", "speakers": list(shares),
                "shares": shares, "diarization_confident": False,
                "reason": reason}

    if total < min_total_speech_s:
        return _single("too_little_speech")
    if len(significant) < 2:
        return _single("below_significance")

    decision = {
        "layout": "speaker_switch",
        "speakers": significant,
        "shares": shares,
        "diarization_confident": True,
        "reason": "two_significant_speakers",
    }
    if visual_hint == "split":
        decision["layout"] = "split"
        decision["reason"] = "two_significant_speakers+visual_split_geometry"
    return decision


# --- DiarizationGuide ---------------------------------------------------------------

class DiarizationGuide:
    """Map diarization speaker labels -> visual face-track ids, per frame.

    The tracker feeds ONE evidence point per frame: the face id with the
    strongest mouth-motion that frame (``observe``). Per label we keep the
    voted face id; a label maps only after ``MIN_FACE_EVIDENCE_FRAMES``
    frames of evidence with confident mean motion. ``suggest(frame)``
    returns the face id to frame, or None when unsure — the caller then
    falls back to legacy lip-motion tracking.

    Short turns (< ``TURN_HYSTERESIS_S``) never steer: a 1s interjection
    must not yank the crop.
    """

    def __init__(self, speaker_turns: List[Dict[str, Any]], fps: float):
        self.fps = float(fps) if fps else 30.0
        self.turns = [
            {"speaker": t["speaker"],
             "start": float(t["start"]),
             "end": float(t["end"])}
            for t in speaker_turns or []
            if t.get("speaker") and float(t.get("end", 0)) > float(t.get("start", 0))
        ]
        # label -> list of (face_id, motion)
        self._evidence: Dict[str, List] = defaultdict(list)
        self._last_hint: Optional[int] = None

    # -- evidence ---------------------------------------------------------
    def observe(self, face_id: int, mouth_motion: float, frame_number: int) -> None:
        """Record one frame of evidence: ``face_id`` had the strongest mouth
        motion this frame while ``_turn_at(frame_number)`` was speaking."""
        turn = self._turn_at(frame_number)
        if turn is None:
            return
        if mouth_motion < EVIDENCE_MOTION_FLOOR:
            return
        self._evidence[turn["speaker"]].append((face_id, float(mouth_motion)))

    # -- mapping ----------------------------------------------------------
    def face_for_label(self, label: str) -> Optional[int]:
        """Voted face id for a speaker label, or None when not confident."""
        ev = self._evidence.get(label, [])
        if len(ev) < MIN_FACE_EVIDENCE_FRAMES:
            return None
        votes: Dict[int, int] = defaultdict(int)
        motion_sum: Dict[int, float] = defaultdict(float)
        for face_id, motion in ev:
            votes[face_id] += 1
            motion_sum[face_id] += motion
        best = max(votes, key=lambda fid: (votes[fid],
                                           motion_sum[fid] / votes[fid]))
        mean_motion = motion_sum[best] / votes[best]
        if mean_motion < CONFIDENT_MEAN_MOTION:
            return None
        return best

    # -- per-frame steering ------------------------------------------------
    def _turn_at(self, frame_number: int) -> Optional[Dict[str, Any]]:
        t = frame_number / self.fps
        for turn in self.turns:
            if turn["start"] <= t < turn["end"]:
                return turn
        return None

    def suggest(self, frame_number: int) -> Optional[int]:
        """Face id to frame at this frame, or None (caller: legacy path)."""
        turn = self._turn_at(frame_number)
        if turn is None:
            return None
        if (turn["end"] - turn["start"]) < TURN_HYSTERESIS_S:
            return None  # too short to steer — no flicker
        return self.face_for_label(turn["speaker"])


# --- switch schedule ------------------------------------------------------------------

def build_switch_schedule(
    speaker_turns: List[Dict[str, Any]],
    fps: float = 30.0,
    min_turn_s: float = MIN_TURN_S,
) -> List[Dict[str, Any]]:
    """Deterministic per-turn switch plan: [{speaker, start_frame,
    switch_at_frame}]. Micro-turns are absorbed so the plan never flickers."""
    fps = float(fps) if fps else 30.0
    # re-absorb micro-turns defensively (callers may pass raw turns)
    turns: List[Dict[str, Any]] = []
    for t in speaker_turns or []:
        dur = float(t.get("end", 0)) - float(t.get("start", 0))
        if turns and dur < min_turn_s:
            turns[-1]["end"] = max(turns[-1]["end"], float(t.get("end", 0)))
        else:
            turns.append({"speaker": t.get("speaker"),
                          "start": float(t.get("start", 0)),
                          "end": float(t.get("end", 0))})
    # merge adjacent same-speaker segments left by micro-turn absorption
    final: List[Dict[str, Any]] = []
    for t in turns:
        if final and final[-1]["speaker"] == t["speaker"]:
            final[-1]["end"] = max(final[-1]["end"], t["end"])
        else:
            final.append(t)
    schedule = []
    for t in final:
        schedule.append({
            "speaker": t["speaker"],
            "start_frame": int(round(t["start"] * fps)),
            "end_frame": int(round(t["end"] * fps)),
            "switch_at_frame": int(round(t["start"] * fps)),
        })
    return schedule
