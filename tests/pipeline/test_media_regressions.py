"""Reference regression gates for Task01; production code is not modified.

Run with PYTHONPATH=work/clippyme/src from the orchestration workspace.
The hardware parameter uses the real environment alias and a live encode probe.
"""
import array
from functools import lru_cache
import json
import pathlib
import re
import shutil
import subprocess

import pytest

from clippyme.pipeline.run_ops import build_cut_command, validate_cut_output

FFMPEG = shutil.which('ffmpeg')
FFPROBE = shutil.which('ffprobe')
pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE), reason='ffmpeg/ffprobe unavailable')


def run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, **kwargs)


@lru_cache(maxsize=1)
def nvenc_unavailable_reason():
    try:
        result = subprocess.run(
            [FFMPEG, '-v', 'error', '-f', 'lavfi', '-i',
             'color=c=black:s=320x240:r=30:d=0.1', '-an', '-c:v', 'h264_nvenc',
             '-frames:v', '1', '-f', 'null', '-'],
            capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f'NVENC live capability probe unavailable: {exc}'
    if result.returncode:
        return 'NVENC live capability probe failed: ' + result.stderr.strip()[-700:]
    return None


@pytest.fixture(params=['libx264', 'nvenc'], ids=['software-libx264', 'hardware-nvenc'])
def expected_codec(request, monkeypatch):
    if request.param == 'nvenc':
        reason = nvenc_unavailable_reason()
        if reason:
            pytest.skip(reason)
    monkeypatch.setenv('CLIPPYME_VIDEO_ENCODER', request.param)
    return 'h264_nvenc' if request.param == 'nvenc' else 'libx264'


def make_source(path, beep=4.0):
    run([FFMPEG, '-v', 'error', '-y', '-f', 'lavfi', '-i',
         'testsrc=duration=12:size=320x240:rate=30', '-f', 'lavfi', '-i',
         f"aevalsrc='if(between(t,{beep},{beep + 0.02}),0.9*sin(2*PI*1000*t),0)':s=48000:d=12",
         '-c:v', 'libx264', '-g', '60', '-pix_fmt', 'yuv420p',
         '-c:a', 'pcm_s16le', '-shortest', str(path)])


def frame_pts(path):
    return [float(frame['best_effort_timestamp_time']) for frame in json.loads(run(
        [FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
         'frame=best_effort_timestamp_time', '-of', 'json', str(path)]
    ).stdout)['frames']]


def media_probe(path):
    return json.loads(run([FFPROBE, '-v', 'error', '-show_entries',
        'stream=index,codec_type,start_time:format=start_time,duration',
        '-of', 'json', str(path)]).stdout)


def audio_marker(path):
    pcm = run([FFMPEG, '-v', 'error', '-copyts', '-i', str(path),
               '-map', '0:a:0', '-af', 'aresample=async=1:first_pts=0',
               '-ar', '48000', '-ac', '1', '-f', 's16le', '-']).stdout
    samples = array.array('h')
    samples.frombytes(pcm)
    return next((i / 48000 for i, sample in enumerate(samples) if abs(sample) > 5000), None)


def frame_ssim(source, cut, source_index, cut_index):
    filters = (
        f'[0:v]select=eq(n\\,{source_index}),setpts=PTS-STARTPTS[a];'
        f'[1:v]select=eq(n\\,{cut_index}),setpts=PTS-STARTPTS[b];[a][b]ssim'
    )
    result = run([FFMPEG, '-hide_banner', '-i', str(source), '-i', str(cut),
                  '-filter_complex', filters, '-an', '-f', 'null', '-'], text=True)
    match = re.search(r'All:([0-9.]+)', result.stderr)
    assert match, result.stderr[-1000:]
    return float(match.group(1))


def cut_and_validate(source, start, end, destination, expected_codec):
    command = build_cut_command(str(source), start, end, str(destination))
    assert command[command.index('-c:v') + 1] == expected_codec, command
    run(command)
    ok, reason = validate_cut_output(str(destination), end - start)
    assert ok, reason
    return frame_pts(destination)


def assert_frame_identity(source, cut, cut_index, correct, wrong):
    actual = frame_ssim(source, cut, correct, cut_index)
    alternative = frame_ssim(source, cut, wrong, cut_index)
    assert actual > 0.95, (cut_index, correct, actual)
    assert actual > alternative, (cut_index, correct, wrong, actual, alternative)


def test_late_source_decoded_edges_and_marker(tmp_path, expected_codec):
    source, cut = tmp_path / 'late.mkv', tmp_path / 'late-cut.mp4'
    make_source(source, beep=8.0)
    pts = cut_and_validate(source, 6.0, 9.0, cut, expected_codec)
    assert len(pts) == 90
    assert pts == pytest.approx([i / 30 for i in range(90)], abs=0.0001)
    assert_frame_identity(source, cut, 0, correct=180, wrong=210)
    assert_frame_identity(source, cut, 89, correct=269, wrong=270)
    assert audio_marker(cut) == pytest.approx(2.0, abs=0.005)


def test_common_container_origin_decoded_edges_and_marker(tmp_path, expected_codec):
    original = tmp_path / 'original.mkv'
    source, cut = tmp_path / 'common.mkv', tmp_path / 'common-cut.mp4'
    make_source(original, beep=4.0)
    run([FFMPEG, '-v', 'error', '-y', '-itsoffset', '5', '-i', str(original),
         '-map', '0', '-c', 'copy', str(source)])
    probe = media_probe(source)
    assert float(probe['format']['start_time']) == pytest.approx(5.0, abs=0.001)
    assert {s['codec_type']: float(s['start_time']) for s in probe['streams']} == {
        'video': pytest.approx(5.0, abs=0.001), 'audio': pytest.approx(5.0, abs=0.001)
    }
    pts = cut_and_validate(source, 3.5, 5.5, cut, expected_codec)
    assert len(pts) == 60
    assert pts == pytest.approx([i / 30 for i in range(60)], abs=0.0001)
    assert_frame_identity(source, cut, 0, correct=105, wrong=120)
    assert_frame_identity(source, cut, 59, correct=164, wrong=165)
    assert audio_marker(cut) == pytest.approx(0.5, abs=0.005)


def test_cross_density_vfr_source_pts_and_decoded_edges(tmp_path, expected_codec):
    source, cut = tmp_path / 'vfr.mp4', tmp_path / 'vfr-cut.mp4'
    run([FFMPEG, '-v', 'error', '-y', '-f', 'lavfi', '-i',
         'testsrc=duration=8:size=320x240:rate=60', '-vf',
         "select='if(lt(t,4),not(mod(n,6)),1)'", '-fps_mode', 'vfr',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)])
    source_pts = frame_pts(source)
    expected = [timestamp - 3.0 for timestamp in source_pts if 3.0 <= timestamp < 5.0]
    assert len(expected) == 70, 'fixture must contain both sparse and dense portions'
    pts = cut_and_validate(source, 3.0, 5.0, cut, expected_codec)
    assert pts == pytest.approx(expected, abs=0.0001)
    assert_frame_identity(source, cut, 0, correct=30, wrong=31)
    assert_frame_identity(source, cut, 69, correct=99, wrong=100)

