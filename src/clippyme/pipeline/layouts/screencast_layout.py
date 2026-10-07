"""SCREENCAST layout: full-width content on top, the speaker underneath.

Used for screen recordings, presentations, and tutorials.
Preserves full-width content on top and stacks the speaker face below.
"""
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
                              max_spread_fraction=0.10):
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


def maybe_promote_screencast(input_video, scenes, scene_strategies, screen_metadata,
                             provider=None, api_key=None,
                             max_scenes=None, max_requests=None,
                             deadline_s=None, _clock=None):
    """Conservative auto-promotion: GENERAL/TRACK -> SCREENCAST only on evidence.

    For each eligible GENERAL or TRACK scene (SPLIT is left alone):
      1. Scene-local semantic classification (is there a screen?).
         Requires layout 'screencast', confidence >= 0.85, explicit positive
         screen evidence, nonempty reason.
      2. Reliable presenter geometry (is there a stable real speaker?).
         Uses detect_presenter_geometry (no center fallback).
      3. Screen + presenter -> SCREENCAST (existing composition).
         Screen without presenter -> keep previous (no fabricated pane).
         Anything unknown/failed/low-confidence -> keep previous.

    Budgets (invocation-level): at most max_scenes scenes evaluated, at most
    max_requests provider calls, deadline_s seconds total. Exhaustion stops
    evaluation; evaluated scenes keep their results, the rest keep the
    previous strategy. Defaults from layout_classifier.

    Only runs when AUTO_LAYOUT is enabled (opt-in); otherwise no-op.
    Never raises. Mutates scene_strategies and screen_metadata in place.
    Returns the list of promoted scene indices.
    """
    from clippyme.pipeline.layouts import layout_classifier as lc
    import time as _time

    if not lc.is_auto_layout_enabled():
        return []
    clock = _clock or _time.monotonic
    t0 = clock()
    max_scenes = lc.MAX_SCENES_PER_INVOCATION if max_scenes is None else max_scenes
    max_requests = lc.MAX_REQUESTS_PER_INVOCATION if max_requests is None else max_requests
    deadline_s = lc.INVOCATION_DEADLINE_S if deadline_s is None else deadline_s

    # Geometry first (local CV, no provider cost); scenes without geometry
    # are skipped before any paid classification.
    try:
        geometry = detect_presenter_geometry(input_video, scenes)
    except Exception:
        geometry = {}

    promoted = []
    evaluated = 0
    requests = 0
    for idx, strat in enumerate(scene_strategies):
        # Eligible: GENERAL and TRACK (a visible webcam often yields TRACK).
        # SPLIT (two-person) and explicit strategies are never auto-promoted.
        if strat not in ("GENERAL", "TRACK"):
            continue
        if evaluated >= max_scenes:
            break
        if requests >= max_requests:
            break
        if clock() - t0 >= deadline_s:
            break
        # Skip paid classification when geometry already rules the scene out.
        if idx not in geometry:
            continue
        evaluated += 1
        try:
            start, end = scenes[idx]
            s_f = start.get_frames() if hasattr(start, "get_frames") else int(start)
            e_f = end.get_frames() if hasattr(end, "get_frames") else int(end)
            requests += 1
            evidence = lc.classify_scene(
                input_video, idx, s_f, e_f,
                provider=provider, api_key=api_key)
        except Exception:
            continue
        if evidence is None:
            continue
        if evidence.get("layout") != "screencast":
            continue
        if not evidence.get("screen_evidence"):
            continue
        # Geometry was precomputed; re-check (cheap).
        center = geometry.get(idx)
        if center is None:
            continue
        scene_strategies[idx] = "SCREENCAST"
        screen_metadata[idx] = center
        promoted.append(idx)
        print("      -> Upgraded scene %d to SCREENCAST" % idx)
    return promoted
