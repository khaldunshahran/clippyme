"""Task 14 review-02 probes, promoted to durable repository tests.

The four failing probes from coordinator review 02 (requirements 4 and 6):
- isolated detections cannot authorize a tight crop (temporal support gate)
- weak/ambiguous evidence retains both edge landmarks (conservative fit)

Originally: work/muse-handoff/task-14/coordinator-review2-remaining.py
"""
"""Task14 review02: sustained acquisition and conservative GENERAL output."""
import importlib.util
from pathlib import Path
import pytest
from clippyme.pipeline import action_framing as af


def observation(frame, confidence=.9, track_id='t-0', box=(20., 120., 60., 100.)):
    return af.ActionObservation(frame, frame / 30., 0, track_id, box,
                                confidence, 'object', .1, .5, 1.,
                                'injected-test', 'test')


@pytest.mark.parametrize('frames', [[0], [0, 1]])
def test_isolated_detections_cannot_authorize_tight_crop(frames):
    log = af.ActionObservationLog(640., 480.)
    for f in frames:
        log.add(observation(f))
    segs, _ = af.segment_actions(log, scene_id=0, scene_start=0,
                                scene_end=59, config=af.ActionFramingConfig(),
                                fps=30., frame_w=640., frame_h=480.)
    assert all(s.kind == 'wide' for s in segs), [(s.kind, s.start_frame, s.end_frame) for s in segs]


@pytest.mark.parametrize('confidence', [.1, .9], ids=['weak-evidence', 'ambiguous-wide-union'])
def test_weak_evidence_retains_both_edge_landmarks_in_real_render(tmp_path, monkeypatch, confidence):
    # Use the submitted fixture/import seams; exercise production analysis,
    # planning, persisted sidecar, pass2 and FFmpeg with an AV source.
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('_r2_render_fixture', repo / 'tests/pipeline/test_action_framing_render.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    def edge_frame(i):
        image = fixture.np.zeros((480, 640, 3), dtype=fixture.np.uint8)
        image[180:300, 20:80] = (0, 255, 0)
        image[180:300, 560:620] = (255, 0, 0)
        return image
    monkeypatch.setattr(fixture, '_frame', edge_frame)
    monkeypatch.delenv('REFRAME_OBJECT_WEIGHTS', raising=False)
    monkeypatch.delenv('REFRAME_SALIENT_GENERAL', raising=False)
    control = fixture.reframe.create_general_frame(edge_frame(0), 270, 480)
    assert fixture._green_present(control, thresh=100)
    assert fixture._blue_present(control, thresh=100)
    src = fixture._build_src(tmp_path / 'src.mp4', tmp_path)
    before = fixture._sha(src)
    def weak(frame):
        return [{'box': [20., 180., 60., 120.], 'confidence': confidence, 'category': 'object'},
                {'box': [560., 180., 60., 120.], 'confidence': confidence, 'category': 'object'}]
    fixture._install(monkeypatch, detect_fn=weak,
                     observe_fn=lambda *args: ([.1, .1], [.5, .5], 0.))
    out = str(tmp_path / 'out.mp4')
    assert fixture.reframe.process_video_to_vertical(src, out, reframe_mode='auto') is True
    assert fixture._sha(src) == before
    assert fixture._tone_hz(out) == pytest.approx(880, abs=40)
    for index, frame in fixture._read_frames(out, [5, 30, 59, 60, 90, 119]).items():
        assert fixture._green_present(frame, thresh=100), f'left participant lost at frame {index}'
        assert fixture._blue_present(frame, thresh=100), f'right participant lost at frame {index}'
