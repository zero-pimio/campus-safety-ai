import json
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

import test_delivery
from campus_safety_ai.apps.deliver_events import inspect_outbox
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination


def records():
    start = test_delivery.DeliveryTests().record()
    end = replace(start, revision=2, phase="END", status="CLOSED",
                  ended_at=start.observed_at, idempotency_key="event-1:2")
    return start, end


class BackgroundDeliveryTests(unittest.TestCase):
    def test_blocked_network_does_not_block_enqueue_and_preserves_order(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        class SlowDestination:
            def publish(self, record):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("test timeout")
                received.append(record["phase"])

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundDelivery(Path(directory) / "outbox.db", SlowDestination())
            start, end = records()
            try:
                self.assertEqual(worker.submit([start]), 1)
                self.assertTrue(entered.wait(1))
                self.assertEqual(worker.submit([end]), 1)
                self.assertEqual(worker.submit([start]), 0)
                self.assertEqual(worker.pending_count(), 2)
            finally:
                release.set()
                worker.close()
            self.assertEqual(received, ["START", "END"])

    def test_failure_backoff_survives_reopen_and_does_not_reorder(self):
        class Offline:
            def publish(self, record):
                raise OSError("do not persist credential-containing error messages")

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            sender = EventDelivery(database, Offline())
            sender.enqueue(list(records()))
            self.assertEqual(sender.drain_due(now=100, retry_base=2), 0)
            self.assertEqual(sender.retry_status()["head"]["nextAttemptAt"], 102)
            self.assertEqual(sender.retry_status()["head"]["attempts"], 1)
            self.assertEqual(sender.retry_status()["head"]["errorType"], "OSError")
            sender.close()
            destination = InMemoryDestination()
            sender = EventDelivery(database, destination)
            try:
                self.assertEqual(sender.drain_due(now=101), 0)
                self.assertEqual(destination.records, [])
                self.assertEqual(sender.drain_due(now=102), 2)
                self.assertEqual([r["phase"] for r in destination.records], ["START", "END"])
            finally:
                sender.close()

    def test_background_retries_without_another_submit(self):
        recovered = threading.Event()
        attempts = []

        class Flaky:
            def publish(self, record):
                attempts.append(record["idempotencyKey"])
                if len(attempts) == 1:
                    raise ConnectionError("offline")
                recovered.set()

        with tempfile.TemporaryDirectory() as directory:
            with BackgroundDelivery(Path(directory) / "outbox.db", Flaky(),
                                    retry_base=0.01, poll_seconds=0.01) as worker:
                worker.submit([records()[0]])
                self.assertTrue(recovered.wait(2))
            self.assertEqual(len(attempts), 2)

    def test_legacy_migration_backs_up_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            start, _ = records()
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE outbox (idempotency_key TEXT PRIMARY KEY, "
                                   "payload TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0)")
                connection.execute("INSERT INTO outbox VALUES (?, ?, 0)",
                                   (start.idempotency_key, json.dumps(start.to_dict())))
            self.assertEqual(inspect_outbox(database), {"pending": 1, "schema": "legacy"})
            sender = EventDelivery(database, InMemoryDestination())
            try:
                self.assertEqual(sender.pending_count(), 1)
                self.assertEqual(sender.drain_due(), 1)
            finally:
                sender.close()
            backups = list(Path(directory).glob("*.backup-*.sqlite3"))
            self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0]) as connection:
                self.assertEqual(connection.execute("SELECT delivered FROM outbox").fetchone()[0], 0)
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_second_worker_is_rejected_and_lock_released_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            with BackgroundDelivery(database, InMemoryDestination()):
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    BackgroundDelivery(database, InMemoryDestination())
            with BackgroundDelivery(database, InMemoryDestination()):
                pass

    def test_shutdown_is_bounded_and_pending_events_resume(self):
        class Offline:
            def publish(self, record):
                raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            before = time.monotonic()
            with BackgroundDelivery(database, Offline(), retry_base=0.01, retry_max=0.01,
                                    shutdown_seconds=0.1) as worker:
                worker.submit(list(records()))
            self.assertLess(time.monotonic() - before, 1)
            self.assertEqual(inspect_outbox(database)["pending"], 2)
            # An exiting consumer may be finishing cleanup after a bounded join.
            worker._thread.join(1)
            destination = InMemoryDestination()
            with BackgroundDelivery(database, destination, poll_seconds=0.01):
                pass
            self.assertEqual([r["phase"] for r in destination.records], ["START", "END"])
            self.assertEqual(inspect_outbox(database)["pending"], 0)
