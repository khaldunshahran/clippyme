
from clippyme.pipeline.layouts.split_layout import (
    _iou,
    dedupe_faces,
    pair_in_frame,
    split_geometry,
    split_filtergraph,
)
from clippyme.pipeline.layouts.screencast_layout import (
    content_bands,
    speaker_crop,
    screencast_filtergraph,
)
from clippyme.pipeline.layouts.camera_inset import (
    nearest_corner,
    is_cornered,
    compute_inset_box,
    inset_filtergraph,
)
from clippyme.pipeline.layouts.punch_in import zoom_curve


def test_iou_and_dedupe_faces():
    b1 = [100, 100, 50, 50]
    b2 = [105, 105, 50, 50]
    assert _iou(b1, b2) > 0.5

    candidates = [{'box': b1}, {'box': b2}]
    deduped = dedupe_faces(candidates, threshold=0.3)
    assert len(deduped) == 1


def test_pair_in_frame_and_split_geometry():
    # Two separated faces in a 1920-wide frame
    left_face = {'box': [200, 300, 150, 150]}
    right_face = {'box': [1400, 300, 150, 150]}
    pair = pair_in_frame([left_face, right_face], frame_w=1920)
    assert pair is not None
    left, right = pair
    assert left[0] < right[0]

    # Split geometry check
    crop_w, crop_h, x, y, half_h = split_geometry(1920, 1080, 1080, 1920, (300, 400))
    assert crop_w % 2 == 0
    assert crop_h % 2 == 0
    assert half_h == 960

    fg = split_filtergraph(1920, 1080, 1080, 1920, (300, 400), (1400, 400))
    assert "vstack=inputs=2" in fg
    assert "pad=1080:1920:0:0" in fg


def test_split_geometry_outer_third_centering_and_adaptive():
    # Outer-third speakers (typical podcast host at cx=350, guest at cx=1550)
    crop_w_l, crop_h_l, xl, yl, _ = split_geometry(1920, 1080, 1080, 1920, (350, 400))
    # Must not clamp to 0 border
    assert xl > 0
    # Center of crop box must be exactly the face center cx=350
    assert (xl + crop_w_l // 2) == 350

    crop_w_r, crop_h_r, xr, yr, _ = split_geometry(1920, 1080, 1080, 1920, (1550, 400))
    # Must not clamp to 1920 - crop_w border
    assert xr < (1920 - crop_w_r)
    # Center of crop box must be exactly cx=1550
    assert (xr + crop_w_r // 2) == 1550

    # Face-adaptive geometry
    cw_adapt, ch_adapt, xa, ya, _ = split_geometry(1920, 1080, 1080, 1920, (400, 400), face_dim=(120, 160))
    # Adaptive height scales with face height (160 * 3.0 = 480)
    assert ch_adapt == 486 or ch_adapt >= 480
    assert cw_adapt % 2 == 0
    assert ch_adapt % 2 == 0
    # Headroom check: eye-line comfortably in upper third (~35%)
    assert ya <= 400

    # Filtergraph with face dimensions
    fg = split_filtergraph(1920, 1080, 1080, 1920, (350, 400), (1550, 400), left_dim=(120, 160), right_dim=(130, 170))
    assert "vstack=inputs=2" in fg


def test_screencast_layout():
    content_h, speaker_h = content_bands(1920, 1080, 1080, 1920)
    assert content_h + speaker_h == 1920
    assert content_h % 2 == 0

    sp_w, sp_h, sp_x, sp_y = speaker_crop(1920, 1080, 1080, speaker_h, (960, 540))
    assert sp_w % 2 == 0
    assert sp_h % 2 == 0

    fg = screencast_filtergraph(1920, 1080, 1080, 1920, (960, 540))
    assert "scale=1080" in fg
    assert "vstack=inputs=2" in fg


def test_camera_inset():
    # Corner webcam box
    webcam_box = [1600, 800, 250, 200]
    assert is_cornered(webcam_box, 1920, 1080) is True

    corner = nearest_corner(webcam_box, 1920, 1080)
    assert corner == ("right", "bottom")

    ix, iy, iw, ih = compute_inset_box(webcam_box, 1920, 1080)
    assert iw % 2 == 0
    assert ih % 2 == 0

    fg = inset_filtergraph(1920, 1080, 1080, 1920, (ix, iy, iw, ih))
    assert "vstack=inputs=2" in fg


def test_punch_in_zoom():
    zooms = zoom_curve(n_frames=100, fps=30, emphasis_times=[1.0], max_zoom=1.12)
    assert len(zooms) == 100
    assert max(zooms) > 1.05
