import json
import os
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_delivery

from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination
from campus_safety_ai.core.outbox_lease import OutboxLease
from campus_safety_ai.core.retention import REGISTRY_NAME, maintain_runtime, register_managed_evidence


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.database = self.root / "outbox.db"
        self.sender = EventDelivery(self.database, InMemoryDestination())
        self.addCleanup(self.sender.close)
        self.epoch = test_delivery.DeliveryTests().record().observed_at.timestamp()

    def artifact(self, name="event.jpg", *, register=True):
        path = self.evidence / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"generated evidence bytes")
        if register:
            with patch("campus_safety_ai.core.retention.time.time", return_value=self.epoch):
                register_managed_evidence(self.evidence, path)
        return path

    def event(self, paths, *, name="event-1", close=True, delivered=True):
        start = replace(test_delivery.DeliveryTests().record(), event_id=name, idempotency_key=f"{name}:1",
                        evidence_uris=tuple(str(path) for path in paths))
        rows = [start]
        if close:
            rows.append(replace(start, revision=2, phase="END", status="CLOSED",
                                ended_at=start.observed_at, idempotency_key=f"{name}:2"))
        self.sender.enqueue(rows)
        if delivered:
            self.sender.drain_due(now=self.epoch)
        return rows

    def run_maintenance(self, **kwargs):
        return maintain_runtime(runtime_root=self.root, database=self.database, evidence_root=self.evidence,
                                retention_seconds=100, now=self.epoch + 200, **kwargs)

    def test_dry_run_does_not_move_files_and_apply_is_recoverable_without_row_pruning(self):
        path = self.artifact()
        rows = self.event([path])
        source_video = self.root / "input.mp4"
        source_video.write_bytes(b"preserved source")
        report = self.run_maintenance()
        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual([item["path"] for item in report["candidates"]], [str(path)])
        self.assertTrue(path.exists())
        self.assertFalse((self.root / "quarantine").exists())
        report = self.run_maintenance(apply=True)
        self.assertEqual(len(report["quarantined"]), 1)
        self.assertFalse(path.exists())
        target = Path(report["quarantined"][0]["target"])
        self.assertEqual(target.read_bytes(), b"generated evidence bytes")
        receipt = [json.loads(line) for line in (target.parent / "receipt.jsonl").read_text().splitlines()]
        self.assertEqual([line["state"] for line in receipt], ["planned", "moved"])
        self.assertEqual(report["bytesFreed"], 0)
        self.assertEqual(source_video.read_bytes(), b"preserved source")
        self.assertEqual(self.sender.enqueue(rows), 0)  # Delivered idempotency history remains.
        self.assertEqual(self.sender.retry_status()["totalRecords"], 2)
        os.rename(target, path)  # Receipt's exact original path restores the evidence.
        self.assertTrue(path.exists())

    def test_pending_active_shared_and_unregistered_source_files_are_protected(self):
        shared, pending, source = self.artifact("shared.jpg"), self.artifact("pending.jpg"), self.artifact("source.mp4", register=False)
        self.event([shared, source], name="closed")
        self.event([shared], name="active", close=False)
        self.event([pending], name="pending", delivered=False)
        report = self.run_maintenance(apply=True)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(len(report["skipped"]), 3)
        for path in (shared, pending, source):
            self.assertTrue(path.exists())
        self.assertEqual(self.sender.pending_count(), 2)

    def test_recent_delivery_or_recent_generation_prevents_old_event_expiration(self):
        path = self.artifact()
        self.event([path])
        with self.sender.connection:
            self.sender.connection.execute("UPDATE outbox SET delivered_at=?", (self.epoch + 150,))
        self.assertEqual(self.run_maintenance()["candidates"], [])
        with self.sender.connection:
            self.sender.connection.execute("UPDATE outbox SET delivered_at=?", (self.epoch,))
        with patch("campus_safety_ai.core.retention.time.time", return_value=self.epoch + 150):
            register_managed_evidence(self.evidence, path)
        self.assertEqual(self.run_maintenance()["candidates"], [])

    def test_unknown_legacy_delivery_age_is_never_guessed_from_event_time(self):
        path = self.artifact()
        self.event([path])
        with self.sender.connection:
            self.sender.connection.execute("UPDATE outbox SET delivered_at=NULL")
        report = self.run_maintenance(apply=True)
        self.assertEqual(report["candidates"], [])
        self.assertTrue(path.exists())

    def test_legacy_schema_is_inspected_without_migration_and_cannot_expire_evidence(self):
        path = self.artifact()
        database = self.root / "legacy.db"
        connection = sqlite3.connect(database)
        try:
            with connection:
                connection.execute("CREATE TABLE outbox(idempotency_key TEXT PRIMARY KEY, "
                                   "payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)")
                record = test_delivery.DeliveryTests().record()
                connection.execute("INSERT INTO outbox VALUES(?, ?, 1)",
                                   (record.idempotency_key, json.dumps(record.to_dict())))
        finally:
            connection.close()
        before = database.read_bytes()
        for apply in (False, True):
            with self.subTest(apply=apply):
                report = maintain_runtime(runtime_root=self.root, database=database,
                                          evidence_root=self.evidence, retention_seconds=100,
                                          now=self.epoch + 200, apply=apply)
                self.assertEqual(report["candidates"], [])
                self.assertEqual(report["outbox"], {"pending": 0, "schema": "legacy"})
                self.assertIn("legacy outbox", report["blockedReasons"][0])
                self.assertEqual(database.read_bytes(), before)
                self.assertTrue(path.exists())
        self.assertEqual(list(self.root.glob("*.backup-*.sqlite3")), [])

    def test_changed_symlink_hardlink_escape_and_unrelated_files_are_never_moved(self):
        changed, linked = self.artifact("changed.jpg"), self.artifact("linked.jpg")
        changed.write_bytes(b"source video replaced evidence bytes")
        outside = self.root / "source.jpg"
        outside.write_bytes(b"outside source")
        linked.unlink()
        linked.symlink_to(outside)
        hardlink = self.artifact("hardlink.jpg")
        os.link(hardlink, self.root / "other-link.jpg")
        unrelated = self.evidence / "unrelated.txt"
        unrelated.write_text("keep")
        self.event([changed, linked, hardlink, outside, self.evidence / ".." / "source.jpg"])
        report = self.run_maintenance(apply=True)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(outside.read_bytes(), b"outside source")
        self.assertTrue(linked.is_symlink())
        self.assertTrue(unrelated.exists())

    def test_symlink_directory_and_registry_are_refused(self):
        path = self.artifact("nested/file.jpg")
        self.event([path])
        actual = self.root / "actual"
        (self.evidence / "nested").rename(actual)
        (self.evidence / "nested").symlink_to(actual, target_is_directory=True)
        self.assertEqual(self.run_maintenance(apply=True)["candidates"], [])
        registry = self.evidence / REGISTRY_NAME
        backup = self.root / "registry"
        registry.rename(backup)
        registry.symlink_to(backup)
        with self.assertRaises(OSError):
            self.run_maintenance(apply=True)

    def test_unsubmitted_journal_blocks_cleanup_even_when_outbox_looks_complete(self):
        path = self.artifact()
        self.event([path])
        (self.root / "unsubmitted-events.jsonl").write_text('{"pending":"unknown lifecycle"}\n')
        report = self.run_maintenance(apply=True)
        self.assertTrue(report["blockedReasons"])
        self.assertEqual(report["candidates"], [])
        self.assertTrue(path.exists())

    def test_active_sender_lease_blocks_apply_but_dry_run_remains_read_only(self):
        path = self.artifact()
        self.event([path])
        lease = OutboxLease(self.database).acquire()
        try:
            self.assertEqual(len(self.run_maintenance()["candidates"]), 1)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.run_maintenance(apply=True)
        finally:
            lease.close()
        self.assertTrue(path.exists())

    def test_corrupt_or_reopened_lifecycle_fails_closed(self):
        path = self.artifact()
        rows = self.event([path])
        reopened = replace(rows[0], revision=3, phase="UPDATE", idempotency_key="event-1:3")
        self.sender.enqueue([reopened])
        self.sender.drain_due(now=self.epoch)
        self.assertEqual(self.run_maintenance(apply=True)["candidates"], [])
        with self.sender.connection:
            self.sender.connection.execute("UPDATE outbox SET payload='invalid' WHERE idempotency_key='event-1:1'")
        with self.assertRaises(ValueError):
            self.run_maintenance(apply=True)
        self.assertTrue(path.exists())

    def test_interrupted_move_leaves_recovery_intent_and_all_outbox_history(self):
        paths = [self.artifact("first.jpg"), self.artifact("second.jpg")]
        self.event(paths)
        rename = os.rename
        calls = 0

        def interrupt_after_first(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated storage fault")
            return rename(*args, **kwargs)

        with patch("campus_safety_ai.core.retention.os.rename", side_effect=interrupt_after_first):
            with self.assertRaisesRegex(OSError, "simulated storage"):
                self.run_maintenance(apply=True)
        receipts = list((self.root / "quarantine").glob("*/receipt.jsonl"))
        self.assertEqual(len(receipts), 1)
        records = [json.loads(line) for line in receipts[0].read_text().splitlines()]
        self.assertEqual([record["state"] for record in records], ["planned", "moved", "planned"])
        self.assertEqual(sum(path.exists() for path in paths), 1)
        self.assertEqual(self.sender.retry_status()["totalRecords"], 2)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_external_outbox_and_symlink_root_are_refused(self):
        other = self.root / "other"
        other.mkdir()
        with self.assertRaises(ValueError):
            maintain_runtime(runtime_root=other, database=self.database, evidence_root=self.evidence,
                             retention_seconds=100)
        alias = self.root / "alias"
        alias.symlink_to(self.evidence, target_is_directory=True)
        with self.assertRaises(ValueError):
            maintain_runtime(runtime_root=self.root, database=self.database, evidence_root=alias,
                             retention_seconds=100)
