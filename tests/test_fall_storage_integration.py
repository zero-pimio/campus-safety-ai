import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_fall_video as fixtures

from campus_safety_ai.apps.fall_video import run
from campus_safety_ai.core.fall_pipeline import WindowPolicy
from campus_safety_ai.core.storage_guard import StorageCapacityError, StorageInspectionError, StoragePolicy


class CountingHead(fixtures.Head):
    inferred_sequences = []

    def predict(self, frames):
        self.inferred_sequences.append(frames[-1].sequence)
        return super().predict(frames)


def execute(tmp_path, *, policy=None, grow_on_frame=None):
    config = tmp_path / "policy.toml"
    config.write_text('schema_version="1.0"\nedge_id="test"\nstart_score=0.5\nend_score=0.3\n'
                      'confirm_seconds=0.2\nclear_seconds=0.3\ncooldown_seconds=0.5\n')
    yielded = []
    CountingHead.inferred_sequences = []

    def source(*args):
        for frame in fixtures.poses():
            yielded.append(frame.sequence)
            if frame.sequence == grow_on_frame:
                # A tiny sparse test file crosses the configured per-run cap;
                # this does not fill the actual filesystem or alter real data.
                with (tmp_path / "run/growth.bin").open("wb") as handle:
                    handle.truncate(policy.max_run_bytes)
            yield frame

    with patch("campus_safety_ai.apps.fall_video.FrozenPoseFallHead", CountingHead), \
            patch("campus_safety_ai.apps.fall_video.cached_pose_frames", source):
        try:
            return run(pose_record=Path("fixture.json"), checkpoint=Path("fixture.pt"), detector=Path("pose.pt"),
                       output_dir=tmp_path / "run", camera_id="camera", source_epoch=1,
                       started_at=fixtures.START, event_config=config,
                       window=WindowPolicy(sample_count=4), evidence=False, storage_policy=policy)
        finally:
            # Preserve only deterministic test observations for failure-path assertions.
            (tmp_path / "yielded.json").write_text(json.dumps(yielded))


def assert_closed_after_storage_failure(tmp_path, error):
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "blocked"
    assert report["processed_frames"] == 30
    assert report["starts"] == report["ends"] == 1
    assert report["open_events_at_exit"] == 0
    assert report["unsubmitted_records"] == 0
    assert report["outstanding_event_ids"] == []
    assert report["delivery_status"]["pending"] == 0
    assert report["storage"] == error.report
    assert report["storage"]["checks"] == 2  # Finalization does not recheck/block END.
    assert report["storage"]["processed_frames_checked"] == 30
    assert report["error"].startswith(f"{type(error).__name__}:")
    assert json.loads((tmp_path / "yielded.json").read_text()) == list(range(1, 31))
    assert CountingHead.inferred_sequences
    assert max(CountingHead.inferred_sequences) < 30  # No inference on the blocking frame.
    audit = fixtures.rows(tmp_path / "run/events.jsonl")
    delivered = fixtures.rows(tmp_path / "run/delivered-events.jsonl")
    assert [record["phase"] for record in audit] == ["START", "END"]
    assert [record["phase"] for record in delivered] == ["START", "END"]
    assert audit[0]["eventId"] == audit[1]["eventId"]
    assert [record["revision"] for record in audit] == [1, 2]
    assert audit[-1]["reason"] == "source_error"
    assert fixtures.rows(tmp_path / "run/unsubmitted-events.jsonl") == []
    with closing(sqlite3.connect(tmp_path / "run/outbox.sqlite3")) as connection:
        assert connection.execute("SELECT COUNT(*), SUM(delivered) FROM outbox").fetchone() == (2, 2)
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    return report


def test_low_space_at_start_stops_before_source_or_event_delivery(tmp_path):
    policy = StoragePolicy(min_free_bytes=100, max_run_bytes=1024 * 1024, check_every_frames=30)
    with patch("campus_safety_ai.core.storage_guard.os.fstatvfs",
               return_value=SimpleNamespace(f_bavail=99, f_frsize=1)), \
            pytest.raises(StorageCapacityError) as caught:
        execute(tmp_path, policy=policy)
    report = json.loads((tmp_path / "run/run.json").read_text())
    assert report["status"] == "blocked"
    assert report["processed_frames"] == 0
    assert report["starts"] == report["ends"] == 0
    assert report["storage"] == caught.value.report
    assert report["storage"]["disk_free_bytes"] == 99
    assert report["storage"]["processed_frames_checked"] == 0
    assert CountingHead.inferred_sequences == []
    assert json.loads((tmp_path / "yielded.json").read_text()) == []
    assert not (tmp_path / "run/outbox.sqlite3").exists()


def test_low_space_after_start_stops_inference_and_still_delivers_end(tmp_path):
    policy = StoragePolicy(min_free_bytes=100, max_run_bytes=1024 * 1024, check_every_frames=30)
    with patch("campus_safety_ai.core.storage_guard.os.fstatvfs", side_effect=[
            SimpleNamespace(f_bavail=10**9, f_frsize=1), SimpleNamespace(f_bavail=99, f_frsize=1)]), \
            pytest.raises(StorageCapacityError) as caught:
        execute(tmp_path, policy=policy)
    report = assert_closed_after_storage_failure(tmp_path, caught.value)
    assert report["storage"]["violations"] == ["disk_free_bytes_at_or_below_reserve"]


def test_actual_run_size_cap_after_start_preserves_files_and_delivers_end(tmp_path):
    policy = StoragePolicy(min_free_bytes=100, max_run_bytes=512 * 1024, check_every_frames=30)
    with patch("campus_safety_ai.core.storage_guard.os.fstatvfs",
               return_value=SimpleNamespace(f_bavail=10**9, f_frsize=1)), \
            pytest.raises(StorageCapacityError) as caught:
        execute(tmp_path, policy=policy, grow_on_frame=30)
    report = assert_closed_after_storage_failure(tmp_path, caught.value)
    assert report["storage"]["violations"] == ["run_bytes_at_or_above_limit"]
    assert report["storage"]["run_bytes"] >= policy.max_run_bytes
    assert (tmp_path / "run/growth.bin").stat().st_size == policy.max_run_bytes


def test_inspection_failure_after_start_is_not_swallowed_and_still_delivers_end(tmp_path):
    policy = StoragePolicy(min_free_bytes=100, max_run_bytes=1024 * 1024, check_every_frames=30)
    with patch("campus_safety_ai.core.storage_guard.os.fstatvfs", side_effect=[
            SimpleNamespace(f_bavail=10**9, f_frsize=1), OSError("simulated inaccessible filesystem")]), \
            pytest.raises(StorageInspectionError) as caught:
        execute(tmp_path, policy=policy)
    report = assert_closed_after_storage_failure(tmp_path, caught.value)
    assert report["storage"]["status"] == "failed"
    assert report["storage"]["violations"] == ["storage_inspection_failed"]
    assert report["storage"]["disk_free_bytes"] is None
    assert isinstance(caught.value.__cause__, OSError)


def test_default_limits_preserve_normal_finite_event_flow(tmp_path):
    with patch("campus_safety_ai.core.storage_guard.os.fstatvfs",
               return_value=SimpleNamespace(f_bavail=10**10, f_frsize=1)):
        report = execute(tmp_path)
    assert report["status"] == "completed"
    assert report["processed_frames"] == 101
    assert report["starts"] == report["ends"] == 2
    assert report["open_events_at_exit"] == report["delivery_status"]["pending"] == 0
    assert report["storage"]["policy"] == asdict(StoragePolicy())
    assert report["storage"]["status"] == "ok"
    assert report["storage"]["checks"] == 4
    assert report["storage"]["processed_frames_checked"] == 90
    assert [record["phase"] for record in fixtures.rows(tmp_path / "run/delivered-events.jsonl")] == [
        "START", "END", "START", "END"]
