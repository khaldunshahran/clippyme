"""SPLIT layout: two speakers stacked one above the other in the 9:16 frame.

Designed for 2-person podcasts and dialogue scenes sharing a wide shot.
Stacking gives each speaker a half-frame crop, so a 1920x1080 source feeds each
half a 1215x1080 region that scales to 1080x960.
"""
import os
import numpy as np

ENABLED = os.environ.get("SPLIT_LAYOUT", "0") == "1"

MIN_COEXISTENCE = 0.5
MIN_SEPARATION = 0.20
MIN_FACE_WIDTH = 0.045
MIN_SCENE_SECONDS = 2.5
SPLIT_TIGHTNESS = float(os.environ.get("SPLIT_TIGHTNESS", "0.8"))
MAX_FACE_OVERLAP = 0.30

SECONDS_PER_SAMPLE = 1.5
MIN_SAMPLES = 8
MAX_SAMPLES = 24


def _iou(a, b):
    """Intersection over union of two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def dedupe_faces(candidates, threshold=MAX_FACE_OVERLAP):
    """Remove overlapping face boxes, keeping the larger one."""
    boxes = [c['box'] for c in candidates if 'box' in c]
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if not any(_iou(b, k) > threshold for k in kept):
            kept.append(b)
    return kept


def split_candidates(candidates, frame_w):
    """Filter to genuine talking heads on left vs right sides."""
    boxes = dedupe_faces(candidates)
    min_w = frame_w * MIN_FACE_WIDTH
    return [b for b in boxes if b[2] >= min_w]


def pair_in_frame(candidates, frame_w):
    """(left_box, right_box) if two distinct faces sit separated, else None."""
    valid = split_candidates(candidates, frame_w)
    if len(valid) < 2:
        return None

    # Pick the two largest faces
    valid = sorted(valid, key=lambda b: b[2] * b[3], reverse=True)[:2]
    first, second = valid[0], valid[1]

    cx1 = first[0] + first[2] / 2.0
    cx2 = second[0] + second[2] / 2.0

    sep = abs(cx1 - cx2) / float(frame_w)
    if sep < MIN_SEPARATION:
        return None

    left, right = (first, second) if cx1 < cx2 else (second, first)
    return left, right


def analyze_scene(sampled_frames, frame_w):
    """Check if sampled frames consistently contain two separated faces."""
    if not sampled_frames:
        return None

    pairs = [pair_in_frame(c, frame_w) for c in sampled_frames]
    coexistent = [p for p in pairs if p is not None]

    if len(coexistent) / float(len(sampled_frames)) < MIN_COEXISTENCE:
        return None

    left_centers = [(p[0][0] + p[0][2] / 2.0, p[0][1] + p[0][3] / 2.0) for p in coexistent]
    right_centers = [(p[1][0] + p[1][2] / 2.0, p[1][1] + p[1][3] / 2.0) for p in coexistent]

    left_med = (float(np.median([c[0] for c in left_centers])), float(np.median([c[1] for c in left_centers])))
    right_med = (float(np.median([c[0] for c in right_centers])), float(np.median([c[1] for c in right_centers])))

    return left_med, right_med


def split_geometry(orig_w, orig_h, out_w, out_h, centre):
    """Crop box for one half of the stacked frame."""
    half_h = out_h // 2
    half_h -= half_h % 2
    aspect = out_w / float(half_h)

    crop_h = int(round(orig_h * max(0.3, min(SPLIT_TIGHTNESS, 1.0))))
    crop_w = int(round(crop_h * aspect))
    if crop_w > orig_w:
        crop_w = orig_w
        crop_h = int(round(crop_w / aspect))

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    cx, cy = centre[0], centre[1]
    x = int(round(cx - crop_w / 2.0))
    x = max(0, min(x, orig_w - crop_w))

    y = int(round(cy - crop_h * 0.42))
    y = max(0, min(y, orig_h - crop_h))

    return crop_w, crop_h, x - (x % 2), y - (y % 2), half_h


def split_filtergraph(orig_w, orig_h, out_w, out_h, left_centre, right_centre):
    """Generate FFmpeg filtergraph for vertical split stack (left on top, right on bottom)."""
    top_w, top_h, top_x, top_y, half_h = split_geometry(orig_w, orig_h, out_w, out_h, left_centre)
    bot_w, bot_h, bot_x, bot_y, _ = split_geometry(orig_w, orig_h, out_w, out_h, right_centre)

    return (
        f"[0:v]split=2[ta][ba];"
        f"[ta]crop=w={top_w}:h={top_h}:x={top_x}:y={top_y},"
        f"scale={out_w}:{half_h}[top];"
        f"[ba]crop=w={bot_w}:h={bot_h}:x={bot_x}:y={bot_y},"
        f"scale={out_w}:{half_h}[bot];"
        f"[top][bot]vstack=inputs=2,"
        f"pad={out_w}:{out_h}:0:0,setsar=1[v]"
    )


def detect_split_scenes(video_path, scenes, strategies=None, samples=None):
    """Detect and upgrade scenes eligible for SPLIT layout."""
    import cv2
    from clippyme.pipeline.reframe_detect import detect_face_candidates

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    found = {}

    try:
        for i, (start, end) in enumerate(scenes):
            if strategies and i < len(strategies) and strategies[i] != 'GENERAL' and strategies[i] != 'WIDE':
                continue
            s_f = start.get_frames() if hasattr(start, 'get_frames') else int(start)
            e_f = end.get_frames() if hasattr(end, 'get_frames') else int(end)
            duration = (e_f - s_f) / fps
            if duration < MIN_SCENE_SECONDS:
                continue

            n = samples or int(min(max(duration / SECONDS_PER_SAMPLE, MIN_SAMPLES), MAX_SAMPLES))
            last_f = e_f - 1
            if total_frames:
                last_f = min(last_f, total_frames - 1)
            if last_f < s_f:
                continue

            sampled = []
            for f_idx in np.linspace(s_f, last_f, n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(f_idx)))
                ok, frame = cap.read()
                if not ok:
                    continue
                if frame.mean() < 16:
                    continue
                sampled.append(detect_face_candidates(frame))

            if len(sampled) < max(4, n // 2):
                continue

            pair = analyze_scene(sampled, frame_w)
            if pair:
                found[i] = pair
    finally:
        cap.release()

    return found

def create_split_frame(frame, out_w, out_h, left_centre, right_centre):
    import cv2
    import numpy as np
    orig_h, orig_w = frame.shape[:2]
    # left
    crop_w_l, crop_h_l, x_l, y_l, half_h = split_geometry(orig_w, orig_h, out_w, out_h, left_centre)
    left_crop = frame[y_l:y_l+crop_h_l, x_l:x_l+crop_w_l]
    left_scaled = cv2.resize(left_crop, (out_w, half_h))
    
    # right
    crop_w_r, crop_h_r, x_r, y_r, _ = split_geometry(orig_w, orig_h, out_w, out_h, right_centre)
    right_crop = frame[y_r:y_r+crop_h_r, x_r:x_r+crop_w_r]
    right_scaled = cv2.resize(right_crop, (out_w, out_h - half_h))
    
    return np.vstack((left_scaled, right_scaled))
