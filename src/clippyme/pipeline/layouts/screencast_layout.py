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

def detect_screencast_scenes(video_path, scenes, samples=10):
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
