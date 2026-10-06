"""Shared drop-mapping fixture, run in pytest.

The same JSON drives the frontend suite
(dashboard-chat/src/editor-timeline/timebase.test.js) -- one fixture, no
drift between the Python render path and the JS preview path.
"""

import json
import os

from clippyme.domain.timebase import (
    derive_drops,
    invert_drops,
    merge_spans,
    output_to_source,
    source_to_output,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "fixtures", "drop_mappings.json")


def _cases():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)["cases"]


def _segments(case):
    return invert_drops(case["drops"], case["source_start"], case["source_end"])


def test_merge_spans_matches_fixture():
    for case in _cases():
        if "merged_drops" in case:
            assert merge_spans(case["drops"]) == case["merged_drops"], case["name"]


def test_source_to_output_matches_fixture():
    for case in _cases():
        segs = _segments(case)
        for pt in case.get("points", []):
            got = source_to_output(pt["t"], segs)
            assert got == pt["out"], (
                f"{case['name']}: source_to_output({pt['t']}) = {got}, "
                f"expected {pt['out']}")


def test_output_to_source_roundtrip():
    for case in _cases():
        segs = _segments(case)
        for pt in case.get("roundtrip", []):
            got = output_to_source(pt["out"], segs)
            assert got == pt["t"], (
                f"{case['name']}: output_to_source({pt['out']}) = {got}, "
                f"expected {pt['t']}")


def test_derive_drops_roundtrip():
    for case in _cases():
        segs = _segments(case)
        back = derive_drops(
            segs, case["source_start"], case["source_end"])
        assert back == merge_spans(case["drops"]), (
            f"{case['name']}: derive_drops mismatch: {back}")
