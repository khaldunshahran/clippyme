"""Actual FFmpeg regression for long VFR frame durations crossing a cut."""
import asyncio
import subprocess

import pytest

from clippyme.domain.clip_project import validate_project
from clippyme.domain.project_render import _materialize_timeline
from clippyme.pipeline.media_qa import probe_media


@pytest.mark.parametrize('end', [2.0, 3.5])
def test_cut_does_not_extend_to_original_frame_duration(tmp_path, end):
    source = tmp_path / 'source.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'testsrc2=s=160x90:r=10:d=4',
        '-f', 'lavfi', '-i', 'sine=frequency=880:duration=4',
        '-vf', "select='lt(t,1)+gte(t,3)'", '-fps_mode', 'vfr',
        '-c:v', 'libx264', '-c:a', 'aac', str(source)], check=True)
    project = validate_project({
        'schema': 'nugget.clip-project/1', 'job_id': 'vfr-fixture',
        'clip_index': 0,
        'source': {'file': source.name, 'width': 160, 'height': 90, 'duration': 4.},
        'segments': [{'id': 'cut', 'start': 0., 'end': end,
            'crop': {'x': 0., 'y': 0., 'w': 160., 'h': 90.}}],
        'render': {'width': 160, 'height': 90, 'fps': 30, 'encoder': 'libx264'}})
    out, expected = asyncio.run(_materialize_timeline(project, str(tmp_path), 0))
    media = probe_media(out)
    assert media['has_audio'] is True and media['has_video'] is True
    assert abs(expected-end) < .001
    assert abs(media['duration']-end) < .1
    assert media['fps'] == 30.
    video = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(end-.2),
        '-i', out, '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        check=True, capture_output=True).stdout
    assert len(video) == 160*90*3 and any(video)
