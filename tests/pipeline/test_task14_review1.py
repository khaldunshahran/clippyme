"""Task 14 review-01: promoted coordinator probes + load-bearing mutations.

Three production contracts the submitted suite missed:

1. Mixed rollout flags (action ON, turn planning OFF) must emit a valid
   reusable v4 plan: TRACK/WIDE scenes keep their accepted renderer
   through the non-applicable contract (scene_kind="none").
2. Box-area occupancy must not manufacture action evidence: visibility
   means actual visibility/occlusion, never size.
3. Malformed detector items must be validated before saliency/tracking
   and isolated per GENERAL scene, never aborting pass 1.

Each mutation test disables one fix and asserts the original failure
mode reproduces.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="CV runtime unavailable")


def _stub_heavy_modules():
    """Import-time stubs for the missing ML wheels (host has no
    mediapipe/ultralytics/torch); mirrors test_action_framing_render.py."""
    import sys
    import types
    if "mediapipe" not in sys.modules:
        mp = types.ModuleType("mediapipe")
        mp_solutions = types.ModuleType("mediapipe.solutions")
        mp_face = types.ModuleType("mediapipe.solutions.face_detection")
        mp_face.FaceDetection = object
        mp_solutions.face_detection = mp_face
        mp.solutions = mp_solutions
        sys.modules["mediapipe"] = mp
        sys.modules["mediapipe.solutions"] = mp_solutions
        sys.modules["mediapipe.solutions.face_detection"] = mp_face
    if "ultralytics" not in sys.modules:
        ultra = types.ModuleType("ultralytics")
        ultra.YOLO = object
        sys.modules["ultralytics"] = ultra
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: False
        torch.cuda = cuda
        torch.device = lambda value: value
        sys.modules["torch"] = torch
        sys.modules["torch.cuda"] = cuda
    if "psutil" not in sys.modules:
        psutil = types.ModuleType("psutil")

        class _VM:
            total = 16 * 1024 ** 3
            available = 8 * 1024 ** 3
            percent = 50.0

        psutil.virtual_memory = lambda: _VM()
        psutil.cpu_count = lambda logical=True: 8
        sys.modules["psutil"] = psutil


_stub_heavy_modules()

from clippyme.pipeline import action_framing as af
from clippyme.pipeline import framing_plan as fp
from clippyme.pipeline import reframe


# --- Promoted probe 1: mixed flags round-trip -------------------------------

def test_action_plan_survives_when_turn_planning_is_disabled():
    """The two rollout flags are independent.

    With action framing on and turn framing off, TRACK/WIDE scenes keep their
    accepted renderer while GENERAL action scenes still need a valid reusable
    v4 sidecar. The production builder's output must round-trip through the
    production parser.
    """
    face_log = fp.ObservationLog(640, 360)
    action_log = af.ActionObservationLog(640, 360)
    plan = fp.build_framing_plan(
        log=face_log,
        scenes=[(0, 29), (30, 59)],
        strategies=["TRACK", "GENERAL"],
        config=fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=60,
        source_fingerprint="source",
        aspect=9 / 16,
        tracker_digest="tracker",
        action_log=action_log,
        action_config=af.ActionFramingConfig(),
        plan_face=False,
    )
    reparsed = fp.plan_from_dict(fp.plan_to_dict(plan))
    assert reparsed.schema_version == 4
    assert reparsed.scenes[0].strategy == "TRACK"
    assert reparsed.scenes[0].applies is False
    assert reparsed.scenes[1].scene_kind == "action"


def test_mutation_face_kind_with_applies_false_is_rejected():
    """The pre-fix builder output must stay invalid.

    With turn planning off the old code emitted scene_kind="face" with
    applies=False for TRACK/WIDE; the v4 parser rejects that combination,
    discarding the whole plan. Reintroducing it must fail loudly here.
    """
    face_log = fp.ObservationLog(640, 360)
    action_log = af.ActionObservationLog(640, 360)
    plan = fp.build_framing_plan(
        log=face_log,
        scenes=[(0, 29), (30, 59)],
        strategies=["TRACK", "GENERAL"],
        config=fp.TurnFramingConfig(),
        fps=30.0,
        frame_w=640,
        frame_h=360,
        frame_count=60,
        source_fingerprint="source",
        aspect=9 / 16,
        tracker_digest="tracker",
        action_log=action_log,
        action_config=af.ActionFramingConfig(),
        plan_face=False,
    )
    d = fp.plan_to_dict(plan)
    d["scenes"][0]["scene_kind"] = "face"  # the old invalid emission
    with pytest.raises(ValueError, match="malformed-plan"):
        fp.plan_from_dict(d)


# --- Promoted probe 2: area must not manufacture evidence -------------------

def _run_steal_scenario():
    """Production path for the large-static-object vs small-moving-person
    scenario. Returns (tight_ids, log)."""
    width, height, fps = 1920, 1080, 30.0
    log = af.ActionObservationLog(width, height)
    tracker = af.ActionTracker()
    camera_motion = {}
    fallbacks = {}
    previous = None

    def detect(frame):
        f = int(frame[0, 0, 0])
        return [
            {"box": (100.0, 100.0, 650.0, 650.0),
             "confidence": 0.75, "category": "object"},
            {"box": (1200.0 + f, 420.0, 80.0, 100.0),
             "confidence": 0.99, "category": "person"},
        ]

    def observe(frame, detections, prev, frame_w, frame_h):
        # Strong support belongs to the moving person, not the static object.
        return [0.0, 1.0], 0.0, prev

    for frame_number in range(30):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0, 0] = frame_number
        previous = reframe._analyze_action_frame(
            frame, frame_number, fps, 0, log, tracker, camera_motion,
            fallbacks, detect, observe, lambda: True, width, height, previous,
        )

    segments, _ = af.segment_actions(
        log,
        scene_id=0,
        scene_start=0,
        scene_end=29,
        config=af.ActionFramingConfig(
            fast_travel_frac=1.0,
            fast_motion_widen=1.0,
            uncertain_motion_range=1.0,
            # Temporal gate compatible with the 30-frame scenario
            # (test is about visibility weighting, not acquisition).
            acquisition_min_span_s=0.2,
            acquisition_max_gap_s=0.2,
        ),
        fps=fps,
        frame_w=width,
        frame_h=height,
        aspect=9 / 16,
        camera_motion=camera_motion,
    )
    tight_ids = [sid for seg in segments if seg.kind == "single"
                 for sid in seg.subject_ids]
    return tight_ids, log


def test_production_visibility_weight_does_not_let_large_static_object_steal():
    """A larger box must not become the action subject merely through area.

    The small person has stronger detector confidence, motion, and saliency.
    The large object is static and has lower confidence. Production observation
    construction must not turn box area into the decisive action evidence.
    """
    tight_ids, _ = _run_steal_scenario()
    # Confidence ordering assigns t-0 to the stronger person and t-1 to the
    # lower-confidence large object on the first frame.
    assert "t-0" in tight_ids
    assert "t-1" not in tight_ids


def test_mutation_area_derived_visibility_lets_large_object_steal(monkeypatch):
    """Restoring area-derived visibility reproduces the original failure.

    With visibility = box-area occupancy, the static 650x650 object at 0.75
    confidence defeats the moving 80x100 person at 0.99 confidence and
    steals the tight crop.
    """
    def area_visibility(box, frame_w, frame_h):
        _x, _y, w, h = box
        return min(1.0, max(0.0, w * h)
                   / max(1.0, frame_w * frame_h * 0.02))

    monkeypatch.setattr(reframe, "_action_detection_visibility",
                        area_visibility)
    tight_ids, _ = _run_steal_scenario()
    assert "t-1" in tight_ids, \
        "area-derived visibility must let the large static object steal"
    assert "t-0" not in tight_ids


def test_production_observations_record_full_visibility():
    """Validated detector observations carry visibility=1.0, not occupancy."""
    width, height, fps = 1920, 1080, 30.0
    log = af.ActionObservationLog(width, height)
    tracker = af.ActionTracker()

    def detect(frame):
        return [{"box": (100.0, 100.0, 650.0, 650.0),
                 "confidence": 0.75, "category": "object"}]

    reframe._analyze_action_frame(
        np.zeros((2, 2, 3), dtype=np.uint8), 0, fps, 0, log, tracker, {},
        {}, detect, lambda *a: ([0.0], 0.0, None), lambda: True,
        width, height, None)
    obs = log.observations_for_scene(0)
    assert obs, "the valid detection must be recorded"
    assert all(o.visibility == 1.0 for o in obs)


# --- Promoted probe 3: malformed detector items are scene-isolated -----------

def test_malformed_detector_item_isolated_to_general_scene():
    """Strict validation must fail soft per scene, not abort pass 1."""
    log = af.ActionObservationLog(640, 360)
    tracker = af.ActionTracker()
    fallbacks = {}

    def bad_detect(frame):
        return [{"box": "not-a-box", "confidence": 0.9,
                 "category": "person"}]

    result = reframe._analyze_action_frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        0, 30.0, 0, log, tracker, {}, fallbacks,
        bad_detect, lambda *args: ([], 0.0, None), lambda: True,
        640, 360, None,
    )
    assert result is None
    assert 0 in fallbacks
    assert fallbacks[0][0] in {"detector-failed", "invalid-detection"}


def test_mutation_disabled_item_validation_aborts_pass1(monkeypatch):
    """Skipping item validation reproduces the original abort.

    With validation disabled, the malformed box reaches
    ActionTracker.update() and raises, aborting the render pass.
    """
    monkeypatch.setattr(reframe, "validate_action_detection",
                        lambda det, fw, fh: det)
    log = af.ActionObservationLog(640, 360)
    tracker = af.ActionTracker()
    fallbacks = {}

    def bad_detect(frame):
        return [{"box": "not-a-box", "confidence": 0.9,
                 "category": "person"}]

    with pytest.raises(Exception):
        reframe._analyze_action_frame(
            np.zeros((4, 4, 3), dtype=np.uint8),
            0, 30.0, 0, log, tracker, {}, fallbacks,
            bad_detect, lambda *args: ([], 0.0, None), lambda: True,
            640, 360, None,
        )


def test_mixed_valid_and_malformed_items_keep_the_scene():
    """Malformed items are dropped individually; valid ones are kept."""
    log = af.ActionObservationLog(640, 360)
    tracker = af.ActionTracker()
    fallbacks = {}

    def mixed_detect(frame):
        return [
            {"box": "not-a-box", "confidence": 0.9, "category": "person"},
            {"box": (100.0, 100.0, 80.0, 100.0),
             "confidence": 0.9, "category": "person"},
        ]

    reframe._analyze_action_frame(
        np.zeros((4, 4, 3), dtype=np.uint8),
        0, 30.0, 0, log, tracker, {}, fallbacks,
        mixed_detect, lambda *a: ([0.0, 0.0], 0.0, None), lambda: True,
        640, 360, None,
    )
    obs = log.observations_for_scene(0)
    assert len(obs) == 1, "the valid detection must survive"
    assert 0 not in fallbacks, \
        "a scene with usable detections must not be force-fallbacked"


def test_validate_action_detection_unit():
    """Pure validator contract: structural checks, never raises."""
    good = af.validate_action_detection(
        {"box": (10.0, 10.0, 80.0, 100.0), "confidence": 0.9,
         "category": "person"}, 640, 360)
    assert good is not None
    assert good["box"] == (10.0, 10.0, 80.0, 100.0)
    assert good["confidence"] == 0.9
    assert good["category"] == "person"

    bad_items = [
        None,
        "not-a-dict",
        {},
        {"box": "not-a-box", "confidence": 0.9, "category": "person"},
        {"box": (10.0, 10.0, 80.0, 100.0), "confidence": float("nan"),
         "category": "person"},
        {"box": (10.0, 10.0, 80.0, 100.0), "confidence": True,
         "category": "person"},
        {"box": (10.0, 10.0, 0.0, 100.0), "confidence": 0.9,
         "category": "person"},
        {"box": (10.0, 10.0, -5.0, 100.0), "confidence": 0.9,
         "category": "person"},
        {"box": (-1.0, 10.0, 80.0, 100.0), "confidence": 0.9,
         "category": "person"},
        {"box": (600.0, 10.0, 80.0, 100.0), "confidence": 0.9,
         "category": "person"},
        {"box": (10.0, 10.0, 80.0, 100.0), "confidence": 0.9,
         "category": ""},
        {"box": (10.0, 10.0, 80.0), "confidence": 0.9,
         "category": "person"},
        {"box": (10.0, 10.0, float("inf"), 100.0), "confidence": 0.9,
         "category": "person"},
    ]
    for item in bad_items:
        assert af.validate_action_detection(item, 640, 360) is None, \
            f"malformed item must be rejected: {item!r}"
