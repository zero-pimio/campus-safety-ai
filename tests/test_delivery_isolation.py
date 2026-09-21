import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import test_delivery

from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination


def records():
    start = test_delivery.DeliveryTests().record()
    return start, replace(start, revision=2, phase="END", status="CLOSED",
                          ended_at=start.observed_at, idempotency_key="event-1:2")


class DeliveryIsolationTests(unittest.TestCase):
    def test_sync_failure_is_counted_and_still_propagates(self):
        class Offline:
            def publish(self, record):
                raise OSError("do not persist this message")

        with tempfile.TemporaryDirectory() as directory:
            sender = EventDelivery(Path(directory) / "outbox.db", Offline())
            try:
                sender.enqueue([records()[0]])
                with self.assertRaises(OSError):
                    sender.flush()
                status = sender.retry_status()
                self.assertEqual(status["totalAttempts"], 1)
                self.assertEqual(status["head"]["errorType"], "OSError")
                self.assertEqual(status["head"]["nextAttemptAt"], 0)
                sender.destination = InMemoryDestination()
                self.assertEqual(sender.flush(), 1)
                self.assertEqual(sender.retry_status()["totalAttempts"], 2)
            finally:
                sender.close()

    def test_failed_event_does_not_block_other_events_and_keeps_its_order(self):
        start, end = records()
        other = replace(start, event_id="event-2", camera_id="other-camera", idempotency_key="event-2:1")
        other_end = replace(end, event_id="event-2", camera_id="other-camera", idempotency_key="event-2:2")
        attempted = []

        class Destination:
            def publish(self, record):
                attempted.append(record["idempotencyKey"])
                if record["eventId"] == "event-1":
                    raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            sender = EventDelivery(database, Destination())
            try:
                sender.enqueue([start, end, other, other_end])
                self.assertEqual(sender.drain_due(now=100, retry_base=2), 2)
                self.assertEqual(attempted, ["event-1:1", "event-2:1", "event-2:2"])
                self.assertEqual(sender.pending_count(), 2)
            finally:
                sender.close()
            target = InMemoryDestination()
            sender = EventDelivery(database, target)
            try:
                self.assertEqual(sender.drain_due(now=101), 0)
                self.assertEqual(sender.drain_due(now=102), 2)
                self.assertEqual([r["phase"] for r in target.records], ["START", "END"])
            finally:
                sender.close()

    def test_retry_migration_preserves_payload_and_retry_state_with_backup(self):
        start, end = records()
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE outbox (idempotency_key TEXT PRIMARY KEY, "
                                   "payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0, "
                                   "attempts INTEGER NOT NULL DEFAULT 0, "
                                   "next_attempt_at REAL NOT NULL DEFAULT 0, last_error TEXT)")
                for record in (start, end):
                    connection.execute("INSERT INTO outbox VALUES (?, ?, 0, ?, ?, ?)", (
                        record.idempotency_key, json.dumps(record.to_dict()),
                        2 if record is start else 0, 102 if record is start else 0,
                        "OSError" if record is start else None,
                    ))
            target = InMemoryDestination()
            sender = EventDelivery(database, target)
            try:
                self.assertEqual(sender.retry_status()["head"]["attempts"], 2)
                self.assertEqual(sender.drain_due(now=101), 0)
                self.assertEqual(sender.drain_due(now=102), 2)
                self.assertEqual([r["idempotencyKey"] for r in target.records], ["event-1:1", "event-1:2"])
            finally:
                sender.close()
            backups = list(Path(directory).glob("*.backup-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0]) as connection:
                self.assertNotIn("event_id", {r[1] for r in connection.execute("PRAGMA table_info(outbox)")})
                self.assertEqual(connection.execute("SELECT SUM(delivered) FROM outbox").fetchone()[0], 0)
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_corrupt_payload_does_not_kill_healthy_event_delivery(self):
        start, end = records()
        other = replace(start, event_id="event-2", idempotency_key="event-2:1")
        with tempfile.TemporaryDirectory() as directory:
            target = InMemoryDestination()
            sender = EventDelivery(Path(directory) / "outbox.db", target)
            try:
                sender.enqueue([start, end, other])
                with sender.connection:
                    sender.connection.execute("UPDATE outbox SET payload='invalid' WHERE idempotency_key=?",
                                              (start.idempotency_key,))
                self.assertEqual(sender.drain_due(now=100), 1)
                self.assertEqual([r["eventId"] for r in target.records], ["event-2"])
                self.assertEqual(sender.retry_status()["head"]["errorType"], "JSONDecodeError")
                self.assertEqual(sender.pending_count(), 2)
            finally:
                sender.close()

    def test_unreadable_legacy_payload_rolls_back_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE outbox (idempotency_key TEXT PRIMARY KEY, "
                                   "payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)")
                connection.execute("INSERT INTO outbox VALUES ('broken', 'invalid', 0)")
            with self.assertRaises(json.JSONDecodeError):
                EventDelivery(database, InMemoryDestination())
            with sqlite3.connect(database) as connection:
                self.assertEqual(len(list(connection.execute("PRAGMA table_info(outbox)"))), 3)
                self.assertEqual(connection.execute("SELECT * FROM outbox").fetchall(),
                                 [("broken", "invalid", 0)])
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(len(list(Path(directory).glob("*.backup-*.sqlite3"))), 1)

    def test_one_batch_bounds_attempts_even_when_every_event_fails(self):
        start, _ = records()
        attempted = []

        class Offline:
            def publish(self, record):
                attempted.append(record["eventId"])
                raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            sender = EventDelivery(Path(directory) / "outbox.db", Offline())
            try:
                sender.flush_chunk = 2
                sender.enqueue([replace(start, event_id=f"event-{i}", idempotency_key=f"event-{i}:1")
                                for i in range(4)])
                self.assertEqual(sender.drain_due(now=100), 0)
                self.assertEqual(attempted, ["event-0", "event-1"])
                self.assertEqual(sender.drain_due(now=100), 0)
                self.assertEqual(attempted, ["event-0", "event-1", "event-2", "event-3"])
            finally:
                sender.close()

    def test_status_distinguishes_due_deferred_and_blocked(self):
        start, end = records()
        other = replace(start, event_id="event-2", idempotency_key="event-2:1")
        with tempfile.TemporaryDirectory() as directory:
            sender = EventDelivery(Path(directory) / "outbox.db", InMemoryDestination())
            try:
                sender.enqueue([start, end, other])
                with sender.connection:
                    sender.connection.execute("UPDATE outbox SET attempts=2, next_attempt_at=200 "
                                              "WHERE idempotency_key=?", (start.idempotency_key,))
                from campus_safety_ai.core.event_delivery import read_outbox_status
                status = read_outbox_status(sender.connection, now=100)
                self.assertEqual(status["pending"], 3)
                self.assertEqual(status["readyEvents"], 1)
                self.assertEqual(status["deferredEvents"], 1)
                self.assertEqual(status["blockedRecords"], 1)
                self.assertEqual(status["nextAttemptAt"], 0)
                self.assertEqual(status["totalAttempts"], 2)
                self.assertEqual(sender.drain_due(now=100), 1)
                status = read_outbox_status(sender.connection, now=100)
                self.assertEqual(status["nextAttemptAt"], 200)
                self.assertEqual(status["totalAttempts"], 3)
            finally:
                sender.close()
