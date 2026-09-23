import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from campus_safety_ai.adapters.timed_file_source import OpenCvTimedFileFrames, video_timestamps

START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def vfr_video(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg and ffprobe required for the real VFR decoder check')
    pytest.importorskip('cv2')
    path = tmp_path / 'variable-rate.mkv'
    # Five real encoded frames with a 300 ms interval after the second frame.
    subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=32x32:rate=10:duration=0.5',
        '-vf', r'setpts=PTS+gte(N\,2)*0.2/TB', '-fps_mode', 'vfr', '-c:v', 'ffv1', str(path),
    ], check=True, capture_output=True, timeout=30)
    return path


def test_real_vfr_frames_preserve_presentation_times_instead_of_average_fps(vfr_video):
    expected = [0, .1, .4, .5, .6]
    assert video_timestamps(vfr_video) == expected
    with OpenCvTimedFileFrames(vfr_video, 'vfr-camera', 2, START) as source:
        frames = list(source)
    assert [(frame.captured_at - START).total_seconds() for frame in frames] == expected
    assert [frame.sequence for frame in frames] == [1, 2, 3, 4, 5]
    assert source.decoded_frames == 5
    assert all((frame.width, frame.height, frame.source_epoch) == (32, 32, 2) for frame in frames)


@pytest.mark.parametrize('times, message', [([0, .1, .4, .5], 'exceed'),
                                           ([0, .1, .4, .5, .6, .7], 'fewer')])
def test_real_decoder_rejects_sidecar_frame_count_mismatch(vfr_video, times, message):
    source = OpenCvTimedFileFrames(vfr_video, 'vfr-camera', 1, START, timestamps=times)
    with pytest.raises(ValueError, match=message):
        list(source)
    assert not source._capture.isOpened()


def test_real_file_context_releases_decoder_after_early_break(vfr_video):
    with OpenCvTimedFileFrames(vfr_video, 'vfr-camera', 1, START) as source:
        for frame in source:
            assert frame.sequence == 1
            break
    assert source.decoded_frames == 1
    assert not source._capture.isOpened()
    source.close()


class Capture:
    def __init__(self, *, opened=True):
        self.opened = opened
        self.releases = 0
        self.reads = 0

    def isOpened(self):
        if isinstance(self.opened, BaseException):
            raise self.opened
        return self.opened

    def read(self):
        assert not self.releases, 'read called on a released decoder'
        self.reads += 1
        return True, SimpleNamespace(shape=(8, 8, 3))

    def release(self):
        self.releases += 1


def inject_capture(monkeypatch, tmp_path, capture):
    path = tmp_path / 'fixture.mp4'
    path.write_bytes(b'placeholder; decoding is injected')
    monkeypatch.setitem(sys.modules, 'cv2', SimpleNamespace(VideoCapture=lambda _: capture))
    return path


def test_explicit_close_stops_an_existing_iterator_without_reading_released_capture(monkeypatch, tmp_path):
    capture = Capture()
    path = inject_capture(monkeypatch, tmp_path, capture)
    source = OpenCvTimedFileFrames(path, 'camera', 1, START, timestamps=[0, .1])
    frames = iter(source)
    next(frames)
    source.close()
    with pytest.raises(StopIteration):
        next(frames)
    assert capture.reads == capture.releases == 1


@pytest.mark.parametrize('opened', [False, OSError('decoder query failed'), KeyboardInterrupt()])
def test_failed_open_or_interrupt_releases_constructed_decoder(monkeypatch, tmp_path, opened):
    capture = Capture(opened=opened)
    path = inject_capture(monkeypatch, tmp_path, capture)
    expected = type(opened) if isinstance(opened, BaseException) else ValueError
    with pytest.raises(expected):
        OpenCvTimedFileFrames(path, 'camera', 1, START, timestamps=[0, .1])
    assert capture.releases == 1


@pytest.mark.parametrize('raw', [[], ['nan'], ['inf'], ['1', '1'], ['2', '1'], ['2', '2.0000001']])
def test_probe_rejects_missing_nonfinite_or_nonincreasing_microsecond_times(monkeypatch, tmp_path, raw):
    result = SimpleNamespace(stdout=json.dumps({'frames': [{'best_effort_timestamp_time': t} for t in raw]}))
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: result)
    with pytest.raises(ValueError):
        video_timestamps(tmp_path / 'fixture.mp4')
