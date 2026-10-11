import pytest
from clippyme.pipeline.layouts import split_layout as split


@pytest.mark.parametrize('flag', [None, '0', 'false', 'no'])
def test_auto_split_off_does_not_analyze_or_change_scenes(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv('SPLIT_LAYOUT', raising=False)
    else:
        monkeypatch.setenv('SPLIT_LAYOUT', flag)
    def unexpected(*args, **kwargs):
        raise AssertionError('disabled automatic split performed detection')
    monkeypatch.setattr(split, 'detect_split_scenes', unexpected)
    strategies = ['WIDE', 'GENERAL', 'TRACK']; metadata = {}
    split.maybe_promote_split('source', [], strategies, metadata)
    assert strategies == ['WIDE', 'GENERAL', 'TRACK']
    assert metadata == {}


def test_opt_in_keeps_detector_evidence_and_other_scene_routes(monkeypatch):
    monkeypatch.setenv('SPLIT_LAYOUT', 'true')
    entry = ((30, 40), (200, 40), (20, 25), (20, 25))
    calls = []
    def detect(path, scenes, strategies):
        calls.append((path, scenes, list(strategies))); return {0: entry}
    monkeypatch.setattr(split, 'detect_split_scenes', detect)
    strategies = ['WIDE', 'SCREENCAST', 'TRACK']; metadata = {}
    split.maybe_promote_split('source', [(0, 30)], strategies, metadata)
    assert calls == [('source', [(0, 30)], ['WIDE', 'SCREENCAST', 'TRACK'])]
    assert strategies == ['SPLIT', 'SCREENCAST', 'TRACK']
    assert metadata == {0: entry}
