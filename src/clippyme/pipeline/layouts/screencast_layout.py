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
    presenter detections: >= min_coverage of samples with a face, at least
    min_detections total, and center spread <= max_spread_fraction of frame
    dimensions (the existing static pane assumes a stationary presenter).
    Scenes without reliable geometry are ABSENT (not center-fallback).
    Absence means "no evidence", not "use center".
    Never raises; returns {} on unreadable video.
    """
    import cv2
    import numpy as np

    found = {}
    try:
        from clippyme.pipeline.reframe_detect import detect_face_candidates
    except Exception:
        return found
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return found
    try:
        fw = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1.0
        fh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1.0
        for i, (start, end) in enumerate(scenes):
            s_f = start.get_frames() if hasattr(start, 'get_frames') else int(start)
            e_f = end.get_frames() if hasattr(end, 'get_frames') else int(end)
            if e_f <= s_f:
                continue
            n = min(samples, e_f - s_f)
            idxs = sorted(set(int(round(x)) for x in np.linspace(s_f, e_f - 1, n)))
            centers = []
            tried = 0
            for f_idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                tried += 1
                try:
                    cands = detect_face_candidates(frame)
                except Exception:
                    continue
                if cands:
                    px, py, pw, ph = max(cands, key=lambda c: c['box'][2] * c['box'][3])['box']
                    # Finite, positive, in-bounds geometry only.
                    if pw > 0 and ph > 0 and 0 <= px + pw / 2 <= fw and 0 <= py + ph / 2 <= fh:
                        centers.append((px + pw / 2.0, py + ph / 2.0))
            if tried == 0 or len(centers) < min_detections:
                continue
            if len(centers) / tried < min_coverage:
                continue
            xs = [c[0] for c in centers]
            ys = [c[1] for c in centers]
            spread_x = (max(xs) - min(xs)) / fw
            spread_y = (max(ys) - min(ys)) / fh
            if max(spread_x, spread_y) > max_spread_fraction:
                continue
            import numpy as _np
            found[i] = (float(_np.median(xs)), float(_np.median(ys)))
    except Exception:
        pass
    finally:
        try:
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
                             provider=None, api_key=None):
    """Conservative auto-promotion: GENERAL -> SCREENCAST only on evidence.

    For each GENERAL scene (TRACK and SPLIT are left alone):
      1. Scene-local semantic classification (is there a screen?).
         Requires layout 'screencast', confidence >= 0.85, positive
         screen evidence.
      2. Reliable presenter geometry (is there a stable real speaker?).
         Uses detect_presenter_geometry (no center fallback).
      3. Screen + presenter -> SCREENCAST (existing composition).
         Screen without presenter -> keep GENERAL (no fabricated pane).
         Anything unknown/failed/low-confidence -> keep GENERAL.

    Only runs when AUTO_LAYOUT is enabled (opt-in); otherwise no-op.
    Never raises. Mutates scene_strategies and screen_metadata in place.
    Returns the list of promoted scene indices.
    """
    from clippyme.pipeline.layouts import layout_classifier as lc

    if not lc.is_auto_layout_enabled():
        return []
    promoted = []
    try:
        geometry = detect_presenter_geometry(input_video, scenes)
    except Exception:
        geometry = {}
    for idx, strat in enumerate(scene_strategies):
        if strat != "GENERAL":
            continue
        try:
            start, end = scenes[idx]
            s_f = start.get_frames() if hasattr(start, "get_frames") else int(start)
            e_f = end.get_frames() if hasattr(end, "get_frames") else int(end)
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
        center = geometry.get(idx)
        if center is None:
            print("      -> Scene %d: screen detected but no reliable "
                  "presenter geometry; keeping GENERAL" % idx)
            continue
        scene_strategies[idx] = "SCREENCAST"
        screen_metadata[idx] = center
        promoted.append(idx)
        print("      -> Upgraded scene %d to SCREENCAST" % idx)
    return promoted
