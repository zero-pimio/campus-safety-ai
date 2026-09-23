import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from campus_safety_ai.core.storage_guard import (
    StorageCapacityError, StorageGuard, StorageInspectionError, StoragePolicy,
)


class StorageGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name).resolve()
        self.root = self.parent / "current-run"
        self.root.mkdir()
        self.policy = StoragePolicy(min_free_bytes=100, max_run_bytes=1000, check_every_frames=3)

    def free(self, size):
        return patch("campus_safety_ai.core.storage_guard.os.fstatvfs",
                     return_value=SimpleNamespace(f_bavail=size, f_frsize=1))

    def test_defaults_are_finite_and_values_are_strict_positive_integers(self):
        policy = StoragePolicy()
        self.assertEqual(policy.min_free_bytes, 256 * 1024 * 1024)
        self.assertEqual(policy.max_run_bytes, 2 * 1024 * 1024 * 1024)
        self.assertEqual(policy.check_every_frames, 30)
        for field in ("min_free_bytes", "max_run_bytes", "check_every_frames"):
            for value in (True, 0, -1, .5, float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    StoragePolicy(**{field: value})

    def test_first_check_and_fixed_frame_intervals_do_not_scan_on_each_frame(self):
        path = self.root / "observations.jsonl"
        path.write_bytes(b"a" * 10)
        guard = StorageGuard(self.root, self.policy)
        self.assertEqual(guard.report()["status"], "not_checked")
        with self.free(10000) as probe:
            self.assertEqual(guard.check(0)["run_bytes"], 10)
            path.write_bytes(b"a" * 40)
            self.assertEqual(guard.check(1)["run_bytes"], 10)
            self.assertEqual(guard.check(2)["run_bytes"], 10)
            self.assertEqual(probe.call_count, 1)
            self.assertEqual(guard.check(3)["run_bytes"], 40)
            self.assertEqual(guard.report()["checks"], 2)
            self.assertEqual(guard.check(3, force=True)["checks"], 3)
            self.assertEqual(probe.call_count, 3)
        with self.assertRaises(ValueError):
            guard.check(2)

    def test_low_space_is_detected_before_first_frame_and_no_files_are_modified(self):
        path = self.root / "outbox.sqlite3"
        path.write_bytes(b"preserved queued events")
        guard = StorageGuard(self.root, self.policy)
        with self.free(100), self.assertRaises(StorageCapacityError) as raised:
            guard.check(0)
        report = raised.exception.report
        self.assertEqual(report["processed_frames_checked"], 0)
        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["disk_free_bytes"], 100)
        self.assertEqual(report["violations"], ["disk_free_bytes_at_or_below_reserve"])
        self.assertEqual(path.read_bytes(), b"preserved queued events")
        self.assertEqual(list(self.root.iterdir()), [path])
        # Rechecking a blocked guard never bypasses the limit via the interval cache.
        with self.free(99), self.assertRaises(StorageCapacityError):
            guard.check(0)
        with self.free(101):
            self.assertEqual(guard.check(0)["status"], "ok")

    def test_run_limit_counts_only_files_in_explicit_run_and_does_not_follow_symlinks(self):
        originals = self.parent / "original-videos"
        originals.mkdir()
        original = originals / "input.mp4"
        original.write_bytes(b"s" * 2000)
        (self.root / "linked-video.mp4").symlink_to(original)
        (self.root / "linked-dataset").symlink_to(originals, target_is_directory=True)
        (self.root / "self-link").symlink_to(self.root, target_is_directory=True)
        nested = self.root / "evidence"
        nested.mkdir()
        snapshot = nested / "snapshot.jpg"
        snapshot.write_bytes(b"a" * 600)
        log = self.root / "events.jsonl"
        log.write_bytes(b"b" * 400)
        guard = StorageGuard(self.root, self.policy)
        with self.free(10000), self.assertRaises(StorageCapacityError) as raised:
            guard.check(0)
        self.assertEqual(raised.exception.report["run_bytes"], 1000)
        self.assertEqual(raised.exception.report["regular_files"], 2)
        self.assertEqual(raised.exception.report["skipped_symlinks"], 3)
        self.assertEqual(raised.exception.report["violations"], ["run_bytes_at_or_above_limit"])
        self.assertEqual(original.read_bytes(), b"s" * 2000)
        self.assertEqual(snapshot.read_bytes(), b"a" * 600)
        self.assertEqual(log.read_bytes(), b"b" * 400)

    def test_changed_root_or_unreadable_directory_fails_closed(self):
        guard = StorageGuard(self.root, self.policy)
        moved = self.parent / "moved-run"
        self.root.rename(moved)
        self.root.symlink_to(moved, target_is_directory=True)
        with self.assertRaises(StorageInspectionError) as raised:
            guard.check(0)
        self.assertEqual(raised.exception.report["status"], "failed")
        self.assertEqual(raised.exception.report["violations"], ["storage_inspection_failed"])
        with self.assertRaises(OSError):
            StorageGuard(self.root, self.policy)

    def test_inspection_failure_is_not_reported_as_zero_bytes_or_healthy(self):
        guard = StorageGuard(self.root, self.policy)
        with patch("campus_safety_ai.core.storage_guard.os.fstatvfs", side_effect=OSError("disk unavailable")):
            with self.assertRaises(StorageInspectionError):
                guard.check(0)
        report = guard.report()
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["run_bytes"])
        self.assertIsNone(report["disk_free_bytes"])

    def test_replaced_directory_and_failed_refresh_do_not_reuse_healthy_snapshot(self):
        guard = StorageGuard(self.root, self.policy)
        (self.root / "events.jsonl").write_bytes(b"events")
        with self.free(10000):
            self.assertEqual(guard.check(0)["run_bytes"], 6)
        self.root.rename(self.parent / "old-current-run")
        self.root.mkdir()
        with self.assertRaises(StorageInspectionError):
            guard.check(3)
        report = guard.report()
        self.assertEqual(report["status"], "failed")
        self.assertIsNone(report["run_bytes"])
        self.assertIsNone(report["disk_free_bytes"])
        self.assertEqual((self.parent / "old-current-run/events.jsonl").read_bytes(), b"events")

    def test_reports_cannot_mutate_internal_limits_and_special_files_are_not_opened(self):
        os.mkfifo(self.root / "named-pipe")
        guard = StorageGuard(self.root, self.policy)
        with self.free(10000):
            result = guard.check(0)
        self.assertEqual(result["skipped_special_files"], 1)
        self.assertEqual(result["run_bytes"], 0)
        result["policy"]["min_free_bytes"] = 0
        result["violations"].append("invented")
        self.assertEqual(guard.report()["policy"]["min_free_bytes"], 100)
        self.assertEqual(guard.report()["violations"], [])

    def test_filesystem_block_units_are_converted_to_bytes(self):
        guard = StorageGuard(self.root, self.policy)
        with patch("campus_safety_ai.core.storage_guard.os.fstatvfs",
                   return_value=SimpleNamespace(f_bavail=20, f_frsize=4096)):
            report = guard.check(0)
        self.assertEqual(report["disk_free_bytes"], 81920)
