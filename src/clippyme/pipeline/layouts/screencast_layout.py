"""SCREENCAST layout: full-width content on top, the speaker underneath.

Used for screen recordings, presentations, and tutorials.
Preserves full-width content on top and stacks the speaker face below.
"""
import math
import os

ENABLED = os.environ.get("SCREENCAST_LAYOUT", "0") == "1"

MIN_WIDTH_FRACTION = 0.5
STACK_MAX_WIDTH_FRACTION = 0.85
MIN_OVERLAP_SECONDS = 0.25
MIN_FACE_WIDTH = 0.05


def content_bands(orig_w, orig_h, out_w, out_h):
    """(content_height, speaker_height) for the stacked screencast frame."""
    content_h = int(round(out_w * orig_h / float(orig_w)))
    content_h -= content_h % 2
    content_h = max(2, min(content_h, out_h - 2))
    speaker_h = out_h - content_h
    return content_h, speaker_h


def speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre):
    """Crop box (w, h, x, y) for the speaker band, framed on the face."""
    aspect = out_w / float(speaker_h)

    crop_h = orig_h
    crop_w = int(round(crop_h * aspect))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / aspect))

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    cx, cy = face_centre
    x = int(round(cx - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))

    y = int(round(cy - crop_h * 0.42))
    y = max(0, min(y, orig_h - crop_h))

    return crop_w, crop_h, x - (x % 2), y - (y % 2)


def screencast_filtergraph(orig_w, orig_h, out_w, out_h, face_centre):
    """Generate FFmpeg filtergraph stacking full-width content above speaker face."""
    content_h, speaker_h = content_bands(orig_w, orig_h, out_w, out_h)
    sp_w, sp_h, sp_x, sp_y = speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre)

    return (
        f"[0:v]split=2[c_in][s_in];"
        f"[c_in]scale={out_w}:{content_h}[content];"
        f"[s_in]crop=w={sp_w}:h={sp_h}:x={sp_x}:y={sp_y},"
        f"scale={out_w}:{speaker_h}[speaker];"
        f"[content][speaker]vstack=inputs=2,"
        f"pad={out_w}:{out_h}:0:0,setsar=1[v]"
    )

def create_screencast_frame(frame, out_w, out_h, face_centre):
    import cv2
    import numpy as np
    orig_h, orig_w = frame.shape[:2]
    
    content_h, speaker_h = content_bands(orig_w, orig_h, out_w, out_h)
    
    # top: content
    content_scaled = cv2.resize(frame, (out_w, content_h))
    
    # bottom: speaker
    sp_w, sp_h, sp_x, sp_y = speaker_crop(orig_w, orig_h, out_w, speaker_h, face_centre)
    speaker_crop_img = frame[sp_y:sp_y+sp_h, sp_x:sp_x+sp_w]
    speaker_scaled = cv2.resize(speaker_crop_img, (out_w, speaker_h))
    
    return np.vstack((content_scaled, speaker_scaled))

def detect_presenter_geometry(video_path, scenes, samples=10,
                              min_coverage=0.60, min_detections=3,
                              max_spread_fraction=0.10, _budget=None):
    """Optional presenter geometry per scene. No fallback.

    Returns dict {scene_idx: (cx, cy)} ONLY for scenes with reliable real
    presenter detections. A detection counts only when:
    - the frame was read successfully (failed reads never inflate coverage),
    - the box is complete and finite [x, y, w, h] with w > 0, h > 0,
    - the box lies wholly in bounds: 0 <= x, 0 <= y, x + w <= fw, y + h <= fh,
    - frame dimensions are finite and positive.

    Coverage = valid detections / attempted samples (not successful reads).
    Requires >= min_coverage, at least min_detections, and center spread
    <= max_spread_fraction of frame dimensions (static pane assumption).

    Scenes without reliable geometry are ABSENT (not center-fallback).
    Failures are local to the sample/scene; a malformed scene never aborts
    later scenes. Never raises; returns {} on unreadable video.
    """
    import cv2
    import math
    import numpy as np

    found = {}
    try:
        from clippyme.pipeline.reframe_detect import detect_face_candidates
    except Exception:
        return found

    def _valid_box(box, fw, fh):
        try:
            if box is None or len(box) != 4:
                return None
            x, y, w, h = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        for v in (x, y, w, h, fw, fh):
            if not math.isfinite(v):
                return None
        if not (fw > 0 and fh > 0):
            return None
        if not (w > 0 and h > 0):
            return None
        # Wholly in bounds.
        if not (x >= 0 and y >= 0 and x + w <= fw and y + h <= fh):
            return None
        return (x + w / 2.0, y + h / 2.0)

    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
    except Exception:
        return found
    if cap is None:
        return found
    try:
        if not cap.isOpened():
            return found
    except Exception:
        return found
    try:
        fw = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        fh = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if not (math.isfinite(fw) and math.isfinite(fh) and fw > 0 and fh > 0):
            return found
        for i, (start, end) in enumerate(scenes):
            # Cooperative budget check: stop starting new scenes after exhaustion.
            if _budget is not None and _budget.expired():
                break
            try:
                s_f = start.get_frames() if hasattr(start, "get_frames") else int(start)
                e_f = end.get_frames() if hasattr(end, "get_frames") else int(end)
            except Exception:
                continue
            if e_f <= s_f:
                continue
            try:
                n = min(int(samples), e_f - s_f)
                if n <= 0:
                    continue
                idxs = sorted(set(int(round(x)) for x in np.linspace(s_f, e_f - 1, n)))
            except Exception:
                continue
            centers = []
            attempted = 0
            for f_idx in idxs:
                # Cooperative check before each sample; do not start new
                # native reads after deadline. Already-running reads are
                # not preempted (no false hard-preemption claims).
                if _budget is not None and _budget.expired():
                    break
                attempted += 1
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ok, frame = cap.read()
                except Exception:
                    continue
                if not ok or frame is None:
                    continue
                try:
                    cands = detect_face_candidates(frame)
                except Exception:
                    continue
                if not cands:
                    continue
                try:
                    best = max(cands, key=lambda c: c["box"][2] * c["box"][3])
                    center = _valid_box(best.get("box"), fw, fh)
                except Exception:
                    continue
                if center is not None:
                    centers.append(center)
            # Coverage over ATTEMPTED samples; failed reads do not inflate.
            if attempted == 0 or len(centers) < min_detections:
                continue
            if len(centers) / attempted < min_coverage:
                continue
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            if max((max(xs) - min(xs)) / fw, (max(ys) - min(ys)) / fh) > max_spread_fraction:
                continue
            found[i] = (float(np.median(xs)), float(np.median(ys)))
    except Exception:
        pass
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
    return found



def detect_screencast_scenes(video_path, scenes, samples=10):
    """Legacy geometry helper for EXPLICIT screencast mode only.

    Returns a face center or frame-center fallback for every scene. The
    fallback is NOT evidence of a screen and must NOT drive automatic
    promotion. For auto layout, use detect_presenter_geometry (no fallback)
    combined with scene-local semantic eligibility.
    """
    import cv2
    import numpy as np
    from clippyme.pipeline.reframe_detect import detect_face_candidates
    
    cap = cv2.VideoCapture(video_path)
    found = {}
    if not cap.isOpened(): return found
    
    try:
        for i, (start, end) in enumerate(scenes):
            s_f = start.get_frames() if hasattr(start, 'get_frames') else int(start)
            e_f = end.get_frames() if hasattr(end, 'get_frames') else int(end)
            if e_f <= s_f: continue
            
            centers = []
            for f_idx in np.linspace(s_f, e_f - 1, samples):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
                ok, frame = cap.read()
                if not ok: continue
                cands = detect_face_candidates(frame)
                if cands:
                    px, py, pw, ph = max(cands, key=lambda c: c['box'][2]*c['box'][3])['box']
                    centers.append((px + pw/2.0, py + ph/2.0))
            if centers:
                cx = float(np.median([c[0] for c in centers]))
                cy = float(np.median([c[1] for c in centers]))
                found[i] = (cx, cy)
            else:
                # Fallback to center
                orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                found[i] = (orig_w/2, orig_h/2)
    finally:
        cap.release()
    return found


class _InvocationBudget:
    """Shared budget across geometry, sampling, provider attempts, backoff.

    Tracks remaining requests and deadline. All consumers check before
    expensive work. take_attempt() consumes one actual-call allowance per
    provider attempt (including retries); take_request() is for scene-level
    gating only when no per-attempt accounting exists. No background threads.
    """
    def __init__(self, max_requests, deadline_s, clock):
        # Validate finite inputs; nonfinite -> zero (conservative, no raise).
        try:
            mr = float(max_requests)
            self.max_requests = int(mr) if math.isfinite(mr) else 0
        except (TypeError, ValueError):
            self.max_requests = 0
        try:
            dl = float(deadline_s)
            self.deadline_s = dl if math.isfinite(dl) else 0.0
        except (TypeError, ValueError):
            self.deadline_s = 0.0
        self.max_requests = max(0, self.max_requests)
        self.deadline_s = max(0.0, self.deadline_s)
        self._clock = clock
        self._t0 = clock()
        self.requests_used = 0
        self.attempts_used = 0

    def remaining_s(self):
        try:
            return max(0.0, self.deadline_s - (self._clock() - self._t0))
        except Exception:
            return 0.0

    def expired(self):
        return self.remaining_s() <= 0

    def _has_allowance(self):
        return not self.expired() and self.attempts_used < self.max_requests

    def has_allowance(self):
        """Peek: True if at least one attempt allowance remains (no consume)."""
        return self._has_allowance()

    def take_attempt(self):
        """Consume one actual provider-call allowance. Returns False if empty."""
        if not self._has_allowance():
            return False
        self.attempts_used += 1
        self.requests_used += 1
        return True

    def take_request(self):
        """Scene-level gating (legacy). Prefer take_attempt per actual call."""
        return self.take_attempt()

    def note_attempt(self):
        # Deprecated: use take_attempt() which enforces allowance.
        self.attempts_used += 1



def maybe_promote_screencast(input_video, scenes, scene_strategies, screen_metadata,
                             provider=None, api_key=None,
                             max_scenes=None, max_requests=None,
                             deadline_s=None, _clock=None):
    """Conservative auto-promotion: GENERAL/TRACK -> SCREENCAST only on evidence.

    Accepts documented "screencast" and "inset" (screen/webcam) labels through
    the existing SCREENCAST composition, under the same strict evidence and
    geometry gate.

    For each eligible scene:
      1. Scene-local semantic classification (is there a screen?).
         Requires layout in ("screencast", "inset"), confidence >= 0.85,
         explicit positive screen evidence, nonempty reason.
      2. Reliable presenter geometry (stable real speaker?).
      3. Screen + presenter -> SCREENCAST. Otherwise keep previous.

    Budgets: one shared _InvocationBudget covers geometry scan, sampling,
    provider attempts and backoff. Eligible scenes are selected (up to
    max_scenes) BEFORE geometry runs, preserving original IDs. The provider
    receives remaining time/request allowance; per-call SDK timeout is
    capped by remaining time; retries stop when budget is exhausted.

    Only runs when AUTO_LAYOUT is enabled. Never raises. Mutates inputs.
    Returns promoted scene indices (original IDs).
    """
    from clippyme.pipeline.layouts import layout_classifier as lc
    import time as _time

    if not lc.is_auto_layout_enabled():
        return []
    clock = _clock or _time.monotonic

    # Validate budget inputs independently; finite only. Nonfinite or
    # invalid -> conservative defaults (no raise). Zero is valid and means
    # "no work".
    def _finite_int(v, default):
        try:
            f = float(v)
            return int(f) if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default
    def _finite_float(v, default):
        try:
            f = float(v)
            return f if math.isfinite(f) else default
        except (TypeError, ValueError):
            return default
    max_scenes_v = _finite_int(max_scenes, lc.MAX_SCENES_PER_INVOCATION)
    max_requests_v = _finite_int(max_requests, lc.MAX_REQUESTS_PER_INVOCATION)
    deadline_v = _finite_float(deadline_s, lc.INVOCATION_DEADLINE_S)
    max_scenes_v = max(0, max_scenes_v)
    max_requests_v = max(0, max_requests_v)
    deadline_v = max(0.0, deadline_v)

    budget = _InvocationBudget(max_requests_v, deadline_v, clock)

    # Zero scene allowance: no geometry, no work.
    if max_scenes_v == 0 or budget.expired():
        return []

    # Select bounded eligible scenes FIRST (preserve original IDs).
    # SPLIT and explicit strategies are never auto-promoted.
    # Check limit BEFORE append to respect max_scenes=0 (handled above)
    # and avoid off-by-one.
    eligible = []
    for idx, strat in enumerate(scene_strategies):
        if len(eligible) >= max_scenes_v:
            break
        if strat in ("GENERAL", "TRACK"):
            eligible.append(idx)
    if not eligible or budget.expired():
        return []

    # Geometry on the bounded subset only, with budget for cooperative stops.
    # Map subset positions back to original IDs.
    subset_scenes = [scenes[i] for i in eligible]
    try:
        subset_geometry = detect_presenter_geometry(
            input_video, subset_scenes, _budget=budget)
    except Exception:
        subset_geometry = {}
    geometry = {eligible[pos]: center for pos, center in subset_geometry.items()
                if pos < len(eligible)}

    promoted = []
    for idx in eligible:
        if budget.expired():
            break
        # Skip paid classification when geometry already rules the scene out.
        if idx not in geometry:
            continue
        # Peek: stop when no attempt allowance remains. Actual consumption
        # happens per-attempt inside classify_scene/_gemini_classify (no double).
        if not budget.has_allowance():
            break
        try:
            start, end = scenes[idx]
            s_f = start.get_frames() if hasattr(start, "get_frames") else int(start)
            e_f = end.get_frames() if hasattr(end, "get_frames") else int(end)
            evidence = lc.classify_scene(
                input_video, idx, s_f, e_f,
                provider=provider, api_key=api_key, _budget=budget)
        except Exception:
            continue
        if evidence is None:
            continue
        if evidence.get("layout") not in ("screencast", "inset"):
            continue
        if not evidence.get("screen_evidence"):
            continue
        center = geometry.get(idx)
        if center is None:
            continue
        scene_strategies[idx] = "SCREENCAST"
        screen_metadata[idx] = center
        promoted.append(idx)
        print("      -> Upgraded scene %d to SCREENCAST" % idx)
    return promoted
