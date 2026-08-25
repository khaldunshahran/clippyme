"""Find the webcam inset in a screen recording or game stream, and frame the two apart.

Targeted at stream VODs (Twitch/Kick/OBS) where the streamer's camera is in a corner.
"""
import os
import numpy as np

CORNER_MARGIN = 0.20
MAX_SUBJECT_HEIGHT = 0.35
MIN_OFFSET = 0.18
INSET_PADDING = float(os.environ.get("INSET_PADDING", "1.45"))
MIN_INSET_HEIGHT = 0.10
MAX_INSET_HEIGHT = 0.38
MAX_CENTRE_SPREAD = 0.05


def nearest_corner(box, frame_w, frame_h):
    """Which corner the subject sits in: (horizontal, vertical)."""
    cx = box[0] + box[2] / 2.0
    cy = box[1] + box[3] / 2.0
    return ("left" if cx < frame_w / 2 else "right",
            "top" if cy < frame_h / 2 else "bottom")


def is_cornered(box, frame_w, frame_h):
    """True when the subject looks like a webcam inset rather than a main talking head."""
    x, y, w, h = box
    if h > frame_h * MAX_SUBJECT_HEIGHT:
        return False

    cx = (x + w / 2.0) / float(frame_w)
    off_centre = abs(cx - 0.5) >= MIN_OFFSET
    if not off_centre:
        return False

    cy = (y + h / 2.0) / float(frame_h)
    near_v_edge = cy < CORNER_MARGIN or cy > (1.0 - CORNER_MARGIN)
    near_h_edge = cx < CORNER_MARGIN or cx > (1.0 - CORNER_MARGIN)

    return near_v_edge or near_h_edge


def compute_inset_box(box, frame_w, frame_h, padding=INSET_PADDING):
    """Grow box to cover the full webcam frame anchored to the corner."""
    x, y, w, h = box
    h_corner, v_corner = nearest_corner(box, frame_w, frame_h)

    inset_h = int(round(h * padding))
    inset_h = max(int(frame_h * MIN_INSET_HEIGHT),
                  min(int(frame_h * MAX_INSET_HEIGHT), inset_h))
    
    # Standard 16:9 or 4:3 webcam aspect
    inset_w = int(round(inset_h * (16.0 / 9.0)))
    inset_w = min(frame_w, inset_w)

    inset_x = 0 if h_corner == "left" else frame_w - inset_w
    inset_y = 0 if v_corner == "top" else frame_h - inset_h

    inset_w -= inset_w % 2
    inset_h -= inset_h % 2
    inset_x -= inset_x % 2
    inset_y -= inset_y % 2

    return inset_x, inset_y, inset_w, inset_h


def find_inset_box(video_path, samples=12):
    """Scan video to find a consistent corner webcam inset."""
    import cv2
    from clippyme.pipeline.reframe_detect import detect_face_candidates

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames <= 0 or frame_w <= 0 or frame_h <= 0:
        cap.release()
        return None

    corner_boxes = []
    try:
        for f_idx in np.linspace(0, total_frames - 1, samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
            ok, frame = cap.read()
            if not ok or frame.mean() < 16:
                continue
            candidates = detect_face_candidates(frame)
            for c in candidates:
                if 'box' in c and is_cornered(c['box'], frame_w, frame_h):
                    corner_boxes.append(c['box'])
                    break
    finally:
        cap.release()

    if len(corner_boxes) < max(3, samples // 3):
        return None

    med_x = float(np.median([b[0] for b in corner_boxes]))
    med_y = float(np.median([b[1] for b in corner_boxes]))
    med_w = float(np.median([b[2] for b in corner_boxes]))
    med_h = float(np.median([b[3] for b in corner_boxes]))

    return compute_inset_box((med_x, med_y, med_w, med_h), frame_w, frame_h)


def inset_filtergraph(orig_w, orig_h, out_w, out_h, inset_box):
    """FFmpeg filtergraph for Top (Full Screen) + Bottom (Enlarged Webcam Inset)."""
    ix, iy, iw, ih = inset_box
    # Content keeps full width on top
    content_h = int(round(out_w * orig_h / float(orig_w)))
    content_h -= content_h % 2
    content_h = max(2, min(content_h, out_h - 2))
    webcam_h = out_h - content_h

    return (
        f"[0:v]split=2[screen][cam];"
        f"[screen]scale={out_w}:{content_h}[top];"
        f"[cam]crop=w={iw}:h={ih}:x={ix}:y={iy},"
        f"scale={out_w}:{webcam_h}[bot];"
        f"[top][bot]vstack=inputs=2,"
        f"pad={out_w}:{out_h}:0:0,setsar=1[v]"
    )
