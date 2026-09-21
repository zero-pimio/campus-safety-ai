import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_delivery
from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination
from campus_safety_ai.core.outbox_lease import OutboxLease


class DeliveryLockTests(unittest.TestCase):
    def records(self):
        start = test_delivery.DeliveryTests().record()
        end = replace(start, revision=2, phase="END", status="CLOSED",
                      ended_at=start.observed_at, idempotency_key="event-1:2")
        return start, end

    def test_background_shutdown_keeps_inflight_request_locked(self):
        entered, release = threading.Event(), threading.Event()
        received = []

        class SlowDestination:
            def publish(self, record):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test request was not released")
                received.append(record["phase"])

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            worker = BackgroundDelivery(database, SlowDestination(), shutdown_seconds=0.05)
            target = InMemoryDestination()
            sender = EventDelivery(database, target)
            start, end = self.records()
            try:
                worker.submit([start])
                self.assertTrue(entered.wait(1))
                before = time.monotonic()
                worker.close()
                self.assertLess(time.monotonic() - before, 0.75)
                self.assertTrue(worker._thread.is_alive())
                # The returning close must not let another sender overtake START.
                for send in (sender.flush, sender.drain_due):
                    with self.assertRaisesRegex(RuntimeError, "already owns"):
                        send()
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    BackgroundDelivery(database, InMemoryDestination())
                self.assertEqual(sender.enqueue([end]), 1)
                self.assertEqual(sender.pending_count(), 2)
                release.set()
                worker._thread.join(2)
                self.assertFalse(worker._thread.is_alive())
                self.assertEqual(sender.flush(), 1)
                self.assertEqual(received, ["START"])
                self.assertEqual([r["phase"] for r in target.records], ["END"])
                self.assertEqual(sender.pending_count(), 0)
            finally:
                release.set()
                worker.close()
                worker._thread.join(2)
                sender.close()

    def test_sync_flush_excludes_other_senders_but_allows_enqueue(self):
        entered, release = threading.Event(), threading.Event()
        received, errors = [], []

        class SlowDestination:
            def publish(self, record):
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("test request was not released")
                received.append(record["phase"])

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            sender = EventDelivery(database, InMemoryDestination())
            start, end = self.records()
            sender.enqueue([start])

            def send():
                consumer = None
                try:
                    consumer = EventDelivery(database, SlowDestination())
                    consumer.flush()
                except BaseException as error:
                    errors.append(error)
                finally:
                    if consumer is not None:
                        consumer.close()

            thread = threading.Thread(target=send, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(sender.enqueue([end]), 1)
                for competing_send in (sender.flush, sender.drain_due):
                    with self.assertRaisesRegex(RuntimeError, "already owns"):
                        competing_send()
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    BackgroundDelivery(database, InMemoryDestination())
                release.set()
                thread.join(2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(received, ["START", "END"])
                with BackgroundDelivery(database, InMemoryDestination()):
                    pass
            finally:
                release.set()
                thread.join(2)
                sender.close()

    def test_failed_sync_flush_releases_lease(self):
        class Offline:
            def publish(self, record):
                raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            sender = EventDelivery(database, Offline())
            try:
                sender.enqueue([self.records()[0]])
                with self.assertRaisesRegex(OSError, "offline"):
                    sender.flush()
                target = InMemoryDestination()
                with BackgroundDelivery(database, target):
                    pass
                self.assertEqual(len(target.records), 1)
            finally:
                sender.close()

    def test_thread_start_failure_releases_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            with patch.object(threading.Thread, "start", side_effect=RuntimeError("no threads")):
                with self.assertRaisesRegex(RuntimeError, "no threads"):
                    BackgroundDelivery(database, InMemoryDestination())
            with BackgroundDelivery(database, InMemoryDestination()):
                pass

    def test_lease_uses_resolved_database_path_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "outbox.db"
            database.touch()
            alias = Path(directory) / "alias.db"
            alias.symlink_to(database)
            lease = OutboxLease(database).acquire()
            try:
                self.assertTrue(lease.held)
                self.assertEqual(lease.database, database.resolve())
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    OutboxLease(alias).acquire()
            finally:
                lease.close()
                lease.close()
            self.assertFalse(lease.held)
            second = OutboxLease(alias).acquire()
            second.close()
