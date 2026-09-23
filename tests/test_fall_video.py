import json
from datetime import datetime, timedelta, UTC
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from campus_safety_ai.apps.fall_video import run
from campus_safety_ai.adapters.fall_sources import OpenCvRtspFrames, RtspSourceError
from campus_safety_ai.contracts import FramePacket
from campus_safety_ai.core.fall_pipeline import PoseFrame, WindowPolicy
from campus_safety_ai.core.source_retry import ReconnectPolicy
from campus_safety_ai.core.event_delivery import EventDelivery, OutboxCapacityError

START = datetime(2026, 1, 1, tzinfo=UTC)


class Head:
    checkpoint_sha256 = 'test-sha'
    model_version = 'test-head'

    def __init__(self, *args):
        pass

    def predict(self, frames):
        # normal -> falling -> motion cleared -> second falling motion
        t = (frames[-1].observed_at - START).total_seconds()
        return (0.9 if 2 <= t < 4 or 7 <= t < 9 else 0.1), {}


def poses(*args, fail=False):
    for index in range(101):
        yield PoseFrame('camera', 1, index + 1, START + timedelta(seconds=index / 10),
                        (), (), 'person-1', True, 'associated')
        if fail and index == 35:
            raise OSError('simulated decode failure')


def execute(tmp_path, source=poses):
    config = tmp_path / 'policy.toml'
    config.write_text('schema_version="1.0"\nedge_id="test"\nstart_score=0.5\nend_score=0.3\n'
                      'confirm_seconds=0.2\nclear_seconds=0.3\ncooldown_seconds=0.5\n')
    with patch('campus_safety_ai.apps.fall_video.FrozenPoseFallHead', Head), \
            patch('campus_safety_ai.apps.fall_video.cached_pose_frames', source):
        return run(pose_record=Path('fixture.json'), checkpoint=Path('fixture.pt'), detector=Path('pose.pt'),
                   output_dir=tmp_path / 'run', camera_id='camera', source_epoch=1,
                   started_at=START, event_config=config,
                   window=WindowPolicy(sample_count=4), evidence=False)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_two_events_delivered_and_epoch_aware_audit(tmp_path):
    report = execute(tmp_path)
    assert report['status'] == 'completed'
    assert report['starts'] == report['ends'] == 2
    assert report['open_events_at_exit'] == 0
    audit = rows(tmp_path / 'run/events.jsonl')
    delivered = rows(tmp_path / 'run/delivered-events.jsonl')
    assert len(audit) == len(delivered) == 4
    assert all(row['sourceEpoch'] == 1 for row in audit)
    assert [row['phase'] for row in audit] == ['START', 'END', 'START', 'END']
    assert audit[0]['eventId'] != audit[2]['eventId']
    assert audit[1]['reason'] == 'motion_cleared'
    with pytest.raises(ValueError, match='new or empty'):
        execute(tmp_path)


def test_source_failure_closes_event_and_saves_failure_report(tmp_path):
    with pytest.raises(OSError, match='simulated'):
        execute(tmp_path, lambda *args: poses(fail=True))
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['status'] == 'failed'
    assert report['starts'] == report['ends'] == 1
    assert report['open_events_at_exit'] == 0
    assert rows(tmp_path / 'run/events.jsonl')[-1]['reason'] == 'source_error'


URI = 'rtsp://alice:do-not-log@camera.example/live?token=secret-query'


class FakeCapture:
    def __init__(self, frames=17, *, opened=True, stop=None, failure=None):
        self.frames = frames
        self.opened = opened
        self.stop = stop
        self.failure = failure
        self.releases = 0

    def isOpened(self):
        if isinstance(self.opened, BaseException):
            raise self.opened
        return self.opened

    def read(self):
        if self.frames:
            self.frames -= 1
            return True, SimpleNamespace(shape=(8, 8, 3))
        if self.stop is not None:
            self.stop.set()
        if self.failure is not None:
            raise self.failure
        return False, None

    def release(self):
        self.releases += 1


class Clock:
    def __init__(self):
        self.tick = 0

    def __call__(self):
        self.tick += .1
        return self.tick


class AlwaysFallingHead(Head):
    def predict(self, frames):
        return .9, {}


class FakePose:
    instances = []

    def __init__(self, *args):
        self.frames = []
        self.instances.append(self)

    def predict(self, frame):
        self.frames.append(frame)
        return PoseFrame(frame.camera_id, frame.source_epoch, frame.sequence, frame.captured_at,
                         (), (), f'{frame.camera_id}:{frame.source_epoch}:person', True, 'associated')


def execute_rtsp(tmp_path, captures, *, stop=None, reconnect=None, **kwargs):
    config = tmp_path / 'policy.toml'
    config.write_text('schema_version="1.0"\nedge_id="test"\nstart_score=0.5\nend_score=0.3\n'
                      'confirm_seconds=0.2\nclear_seconds=0.3\ncooldown_seconds=0.5\n')
    sources = iter(captures)
    # The production OpenCvRtspFrames adapter is exercised; only its backend
    # decoder/clock and the model inference are injected for deterministic tests.
    clock = Clock()

    def source_factory(*args, **kwargs):
        kwargs['timestamp_anchor'] = (START, 0)
        return OpenCvRtspFrames(*args, **kwargs, capture_factory=lambda *args: next(sources),
                               monotonic=clock, utcnow=lambda: START)
    FakePose.instances = []
    with patch('campus_safety_ai.apps.fall_video.FrozenPoseFallHead', AlwaysFallingHead), \
            patch('campus_safety_ai.apps.fall_video.SinglePersonPose', FakePose), \
            patch('campus_safety_ai.apps.fall_video.OpenCvRtspFrames', source_factory):
        return run(video=URI, checkpoint=Path('fixture.pt'), detector=Path('pose.pt'),
                   output_dir=tmp_path / 'run', camera_id='camera', source_epoch=3,
                   started_at=START, event_config=config, window=WindowPolicy(sample_count=4),
                   evidence=False, stop_event=stop, reconnect=reconnect or ReconnectPolicy(1, 0, 0), **kwargs)


def test_rtsp_read_failure_finalizes_before_wait_and_reconnect_starts_new_event(tmp_path):
    class Stop(Event):
        def wait(self, delay):
            audit = rows(tmp_path / 'run/events.jsonl')
            assert [record['phase'] for record in audit] == ['START', 'END']
            assert audit[-1]['reason'] == 'source_disconnected'
            assert first.releases == 1
            return super().wait(delay)

    stop = Stop()
    first, second = FakeCapture(failure=RuntimeError(URI)), FakeCapture(stop=stop)
    report = execute_rtsp(tmp_path, [first, second], stop=stop)
    assert report['status'] == 'interrupted'
    assert report['starts'] == report['ends'] == 2
    assert report['open_events_at_exit'] == 0
    assert report['reconnect']['attempts_used'] == report['reconnect']['successful_reconnects'] == 1
    assert report['reconnect']['read_failures'] == 1
    assert report['last_source_epoch'] == 4
    audit = rows(tmp_path / 'run/events.jsonl')
    assert [record['sourceEpoch'] for record in audit] == [3, 3, 4, 4]
    assert audit[0]['eventId'] != audit[2]['eventId']
    assert audit[3]['reason'] == 'interrupted'
    observations = rows(tmp_path / 'run/observations.jsonl')
    second_epoch = [row for row in observations if row['sourceEpoch'] == 4]
    assert second_epoch[0]['reason'] == 'warming_up'
    assert second_epoch[0]['score'] is None
    assert len(FakePose.instances) == 2
    assert [instance.frames[0].sequence for instance in FakePose.instances] == [1, 1]
    assert first.releases == second.releases == 1
    report_text = (tmp_path / 'run/run.json').read_text()
    assert all(secret not in report_text for secret in ('alice', 'do-not-log', 'secret-query'))
    assert report['source'] == 'rtsp://camera.example/live'
    assert 'monotonic receive time' in report['timestamp_basis']


def test_rtsp_success_does_not_reset_total_retry_budget(tmp_path):
    first, second = FakeCapture(), FakeCapture()
    with pytest.raises(RtspSourceError, match='budget exhausted'):
        execute_rtsp(tmp_path, [first, second])
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['status'] == 'failed'
    assert report['starts'] == report['ends'] == 2
    assert report['reconnect']['exhausted']
    assert report['reconnect']['attempts_used'] == 1
    assert report['reconnect']['read_failures'] == 2
    assert first.releases == second.releases == 1


def test_rtsp_open_failures_release_all_attempts_and_save_safe_report(tmp_path):
    captures = [FakeCapture(opened=RuntimeError(URI)), FakeCapture(opened=False)]
    with pytest.raises(RtspSourceError, match='budget exhausted'):
        execute_rtsp(tmp_path, captures)
    report_text = (tmp_path / 'run/run.json').read_text()
    report = json.loads(report_text)
    assert report['reconnect']['open_failures'] == 2
    assert report['reconnect']['successful_reconnects'] == 0
    assert report['processed_frames'] == 0
    assert all(capture.releases == 1 for capture in captures)
    assert all(secret not in report_text for secret in ('alice', 'do-not-log', 'secret-query'))


def test_rtsp_initial_open_failure_can_recover_into_a_new_epoch(tmp_path):
    stop = Event()
    first, second = FakeCapture(opened=False), FakeCapture(stop=stop)
    report = execute_rtsp(tmp_path, [first, second], stop=stop)
    assert report['reconnect']['open_failures'] == 1
    assert report['reconnect']['successful_reconnects'] == 1
    assert report['starts'] == report['ends'] == 1
    assert all(row['sourceEpoch'] == 4 for row in rows(tmp_path / 'run/events.jsonl'))
    assert first.releases == second.releases == 1


def test_ctrl_c_during_reconnect_wait_closes_event_without_opening_next_capture(tmp_path):
    class InterruptWait(Event):
        def wait(self, delay):
            raise KeyboardInterrupt()

    capture = FakeCapture()
    with pytest.raises(KeyboardInterrupt):
        execute_rtsp(tmp_path, [capture], stop=InterruptWait(), reconnect=ReconnectPolicy(2, 30, 30))
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['status'] == 'interrupted'
    assert report['starts'] == report['ends'] == 1
    assert report['reconnect']['attempts_used'] == 0
    assert report['last_source_epoch'] == 3
    assert capture.releases == 1


def test_external_stop_interrupts_backoff_and_no_further_decoder_is_opened(tmp_path):
    class StopInWait(Event):
        def wait(self, delay):
            self.set()
            return super().wait(delay)

    capture = FakeCapture(opened=False)
    report = execute_rtsp(tmp_path, [capture], stop=StopInWait(), reconnect=ReconnectPolicy(2, 30, 30))
    assert report['status'] == 'interrupted'
    assert report['reconnect']['attempts_used'] == 0
    assert report['elapsed_seconds'] < 2
    assert capture.releases == 1


def test_zero_rtsp_retry_budget_does_not_reopen(tmp_path):
    capture = FakeCapture(opened=False)
    with pytest.raises(RtspSourceError):
        execute_rtsp(tmp_path, [capture], reconnect=ReconnectPolicy(0))
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['reconnect']['attempts_used'] == 0
    assert report['reconnect']['exhausted']
    assert capture.releases == 1


def test_full_outbox_stops_rtsp_and_preserves_unsubmitted_lifecycle(tmp_path):
    class OfflineProducer(EventDelivery):
        # Model an offline background consumer: actual SQLite enqueue/capacity
        # logic runs, and accepted records remain pending rather than publishing.
        def submit(self, records):
            return self.enqueue(records)

    first, second = FakeCapture(), FakeCapture()
    with patch('campus_safety_ai.apps.fall_video.EventDelivery', OfflineProducer), \
            pytest.raises(OutboxCapacityError):
        execute_rtsp(tmp_path, [first, second], max_pending_records=1)
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['status'] == 'blocked'
    assert report['reconnect']['attempts_used'] == 0
    assert report['unsubmitted_records'] == 1
    assert report['outstanding_event_ids']
    assert report['delivery_status']['pending'] == 1
    rejected = rows(tmp_path / 'run/unsubmitted-events.jsonl')
    assert [row['phase'] for row in rejected] == ['END']
    assert first.releases == 1 and second.releases == 0


def test_rejected_start_and_end_are_both_preserved_without_claiming_delivery(tmp_path):
    capture = FakeCapture()
    with pytest.raises(OutboxCapacityError):
        execute_rtsp(tmp_path, [capture], max_pending_bytes=1)
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['status'] == 'blocked'
    assert report['unsubmitted_records'] == 2
    assert len(report['outstanding_event_ids']) == 1
    assert report['delivery_status']['pending'] == 0
    assert report['finalize_error'].startswith('OutboxCapacityError:')
    assert [row['phase'] for row in rows(tmp_path / 'run/unsubmitted-events.jsonl')] == ['START', 'END']
    assert capture.releases == 1


def test_unconfirmed_candidate_is_not_carried_across_reconnect(tmp_path):
    stop = Event()
    # Each source epoch reaches a positive window but ends before confirmation.
    first, second = FakeCapture(frames=12), FakeCapture(frames=12, stop=stop)
    report = execute_rtsp(tmp_path, [first, second], stop=stop)
    assert report['reconnect']['successful_reconnects'] == 1
    assert report['starts'] == report['ends'] == 0
    assert not rows(tmp_path / 'run/events.jsonl')


def test_processing_errors_do_not_trigger_reconnect_or_expose_rtsp_uri(tmp_path):
    capture = FakeCapture()
    with patch.object(FakePose, 'predict', side_effect=RuntimeError(URI)), \
            pytest.raises(RuntimeError, match='sanitized run report') as caught:
        execute_rtsp(tmp_path, [capture])
    report_text = (tmp_path / 'run/run.json').read_text()
    report = json.loads(report_text)
    assert report['status'] == 'failed'
    assert report['reconnect']['attempts_used'] == 0
    assert all(secret not in report_text + str(caught.value) for secret in ('alice', 'do-not-log', 'secret-query'))
    assert capture.releases == 1


@pytest.mark.parametrize('failure', [None, OSError('file decoder failed')])
def test_finite_file_eof_or_failure_never_reconnects(tmp_path, failure):
    config = tmp_path / 'policy.toml'
    config.write_text('schema_version="1.0"\nedge_id="test"\n')

    class FileSource:
        closes = 0

        def __init__(self, *args):
            pass

        def __iter__(self):
            yield FramePacket('camera', 1, 1, START, START, 8, 8)
            if failure:
                raise failure

        def __enter__(self):
            return self

        def __exit__(self, *args):
            type(self).closes += 1

    with patch('campus_safety_ai.apps.fall_video.FrozenPoseFallHead', AlwaysFallingHead), \
            patch('campus_safety_ai.apps.fall_video.SinglePersonPose', FakePose), \
            patch('campus_safety_ai.apps.fall_video.OpenCvTimedFileFrames', FileSource), \
            patch('campus_safety_ai.apps.fall_video.OpenCvRtspFrames') as rtsp:
        def invoke():
            return run(video='fixture.mp4', checkpoint=Path('fixture.pt'), detector=Path('pose.pt'),
                       output_dir=tmp_path / 'run', camera_id='camera', source_epoch=1,
                       started_at=START, event_config=config, evidence=False)

        if failure:
            with pytest.raises(OSError, match='file decoder failed'):
                invoke()
        else:
            assert invoke()['status'] == 'completed'
        rtsp.assert_not_called()
    report = json.loads((tmp_path / 'run/run.json').read_text())
    assert report['reconnect']['enabled'] is False
    assert report['reconnect']['attempts_used'] == 0
    assert 'presentation timestamps' in report['timestamp_basis']
    assert FileSource.closes == 1
