"""Phase 2 (2C): kinetic hook-card reveal sequence.

- frame counts follow fps/interval/hold defaults (12 fps, 0.28 s, 1.0 s hold)
- card canvas is constant across the whole sequence
- reveal progresses word by word; the newest word pops at 1.18x
  (single-word card: popped ink area ~= 1.18^2 of the static card)
- empty hooks never render blank: transcript-aware generation, hard fallback
"""
import os

import pytest
from PIL import Image

from clippyme.domain.hooks import (
    KINETIC_FPS,
    KINETIC_HOLD_SEC,
    KINETIC_WORD_INTERVAL,
    generate_hook_text,
    render_hook_reveal_frames,
)


def _ink_area(path):
    with Image.open(path) as im:
        g = im.convert("L")
    px = g.load()
    w, h = g.size
    return sum(1 for y in range(h) for x in range(w) if px[x, y] > 40)


def test_reveal_frame_counts_and_constant_canvas(tmp_path):
    pattern, fps, total = render_hook_reveal_frames(
        "alpha beta gamma", 540, str(tmp_path))
    assert fps == KINETIC_FPS == 12
    assert KINETIC_WORD_INTERVAL == pytest.approx(0.28)
    assert KINETIC_HOLD_SEC == pytest.approx(1.0)
    repeat = round(0.28 * 12)   # 3 frames per revealed stage
    hold = round(1.0 * 12)      # 12 hold frames
    assert total == 3 * repeat + hold == 21
    sizes = set()
    for i in range(1, total + 1):
        with Image.open(pattern % i) as im:
            sizes.add(im.size)
    assert len(sizes) == 1  # card never resizes mid-reveal


def test_reveal_progresses_word_by_word(tmp_path):
    pattern, _, total = render_hook_reveal_frames(
        "alpha beta gamma", 540, str(tmp_path))
    f1, f4, f7 = pattern % 1, pattern % 4, pattern % 7
    assert _ink_area(f1) > 0
    assert _ink_area(f4) != _ink_area(f1)   # second word revealed
    assert _ink_area(f7) != _ink_area(f4)   # third word revealed
    # hold tail == static full card (no pop once complete)
    assert _ink_area(pattern % total) == _ink_area(pattern % (total - 1))


def test_newest_word_pops_at_118x(tmp_path):
    pattern, _, _ = render_hook_reveal_frames("MONEY", 540, str(tmp_path))
    popped = _ink_area(pattern % 1)   # reveal stage: newest word at 1.18x
    static = _ink_area(pattern % 4)   # hold frame: full card, no pop
    ratio = popped / static
    assert 1.25 < ratio < 1.55, ratio  # ~= 1.18^2 = 1.3924


def test_empty_text_renders_single_frame(tmp_path):
    pattern, _, total = render_hook_reveal_frames("", 540, str(tmp_path))
    assert total == 1
    assert os.path.exists(pattern % 1)


def test_empty_hook_falls_back_without_network(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "clippyme.storage.config_store.load_persistent_config", lambda: {})
    assert generate_hook_text("") == "You need to see this"
    assert generate_hook_text("   ") == "You need to see this"
    # transcript present but no key -> hard fallback, no API call
    assert generate_hook_text("some transcript words here") == \
        "You need to see this"
