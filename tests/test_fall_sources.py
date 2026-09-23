import json
from datetime import datetime, UTC
from types import SimpleNamespace
from threading import Event

import pytest

from campus_safety_ai.adapters.fall_sources import (
    OpenCvRtspFrames, RtspOpenError, RtspReadError, cached_pose_frames,
)
from campus_safety_ai.adapters.runtimes.pose_fall import sha256


class Capture:
    def __init__(self, reads=(), opened=True):
        self.reads = iter(reads)
        self.opened = opened
        self.releases = 0

    def isOpened(self):
        if isinstance(self.opened, BaseException):
            raise self.opened
        return self.opened

    def read(self):
        result = next(self.reads, (False, None))
        if isinstance(result, BaseException):
            raise result
        return result

    def release(self):
        self.releases += 1


URI = 'rtsp://alice:do-not-log@camera.example/live?token=secret-query'


@pytest.mark.parametrize('opened', [False, RuntimeError(URI), KeyboardInterrupt()])
def test_rtsp_open_failures_release_capture_and_never_echo_uri(opened):
    capture = Capture(opened=opened)
    expected = KeyboardInterrupt if isinstance(opened, KeyboardInterrupt) else RtspOpenError
    with pytest.raises(expected) as caught:
        OpenCvRtspFrames(URI, 'camera', 3, capture_factory=lambda *args: capture)
    assert capture.releases == 1
    assert 'alice' not in str(caught.value) and 'secret-query' not in str(caught.value)


def test_rtsp_factory_exception_is_redacted():
    def factory(*args):
        raise RuntimeError(URI)

    with pytest.raises(RtspOpenError, match='cannot open RTSP source') as caught:
        OpenCvRtspFrames(URI, 'camera', 1, capture_factory=factory)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize('failure', [(False, None), RuntimeError(URI), (True, None)])
def test_rtsp_read_failures_release_capture_and_raise_safe_error(failure):
    capture = Capture([failure])
    source = OpenCvRtspFrames(URI, 'camera', 1, capture_factory=lambda *args: capture)
    with pytest.raises(RtspReadError) as caught:
        list(source)
    source.close()
    assert capture.releases == 1
    assert 'alice' not in str(caught.value) and 'secret-query' not in str(caught.value)


def test_rtsp_receive_timestamps_epoch_sequence_timeouts_and_explicit_stop():
    stop = Event()
    now = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = iter([40.0, 40.25, 40.75])
    capture = Capture([(True, SimpleNamespace(shape=(80, 100, 3)))] * 2)
    calls = []

    def factory(*args):
        calls.append(args)
        return capture

    source = OpenCvRtspFrames(URI, 'camera', 7, capture_factory=factory, stop_event=stop,
                             monotonic=lambda: next(ticks), utcnow=lambda: now,
                             open_timeout_ms=12, read_timeout_ms=34)
    frames = iter(source)
    first, second = next(frames), next(frames)
    stop.set()
    with pytest.raises(StopIteration):
        next(frames)
    assert calls == [(URI, 12, 34)]
    assert [first.sequence, second.sequence] == [1, 2]
    assert first.source_epoch == second.source_epoch == 7
    assert (second.captured_at - first.captured_at).total_seconds() == .5
    assert (first.width, first.height) == (100, 80)
    assert capture.releases == 1


def test_rtsp_close_stops_iteration_and_double_iteration_is_rejected():
    capture = Capture([(True, SimpleNamespace(shape=(8, 8, 3)))] * 2)
    source = OpenCvRtspFrames(URI, 'camera', 1, capture_factory=lambda *args: capture)
    frames = iter(source)
    next(frames)
    source.close()
    assert list(frames) == []
    with pytest.raises(RuntimeError, match='only be consumed once'):
        list(source)
    assert capture.releases == 1


def test_cached_multi_person_frames_are_unknown_and_reset_identity(tmp_path):
    identity = {'pose_checkpoint_sha256': 'test', 'settings': {}, 'source_sha256': 'test'}
    identity_path = tmp_path / 'cache-identity.json'
    identity_path.write_text(json.dumps(identity))
    identity_hash = sha256(identity_path)
    record = {'sequence_id': 'sample', 'cache_identity_sha256': identity_hash,
              'timestamps_ms': [0, 100, 200], 'keypoints': [[], [], []], 'boxes': [[], [], []],
              'tracking': [{'association_valid': True, 'detection_count': count, 'track_id': 1,
                            'status': 'associated'} for count in [1, 2, 1]]}
    path = tmp_path / 'sample.json'
    path.write_text(json.dumps(record))
    (tmp_path / 'prepared.json').write_text(json.dumps({'cache_identity_sha256': identity_hash,
                                                       'record_sha256': {'sample': sha256(path)}}))
    frames = list(cached_pose_frames(path, SimpleNamespace(extraction_identity=identity),
                                    'camera', 1, datetime(2026, 1, 1, tzinfo=UTC)))
    assert frames[0].valid and frames[2].valid
    assert not frames[1].valid
    assert frames[1].reason == 'multiple_people'
    assert frames[1].track_key is None
    assert frames[0].track_key != frames[2].track_key
