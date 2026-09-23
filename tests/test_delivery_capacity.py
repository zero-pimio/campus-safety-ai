import json
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import test_delivery

from campus_safety_ai.core.background_delivery import BackgroundDelivery
from campus_safety_ai.core.event_delivery import (
    EventDelivery, InMemoryDestination, OutboxCapacityError, read_outbox_status,
)


class DeliveryCapacityTests(unittest.TestCase):
    def test_atomic_rejection_preserves_existing_and_duplicate_at_capacity(self):
        record = test_delivery.DeliveryTests().record()
        other = replace(record, event_id="event-2", idempotency_key="event-2:1")
        end = replace(record, revision=2, phase="END", status="CLOSED",
                      ended_at=record.observed_at, idempotency_key="event-1:2")
        with tempfile.TemporaryDirectory() as directory:
            sender = EventDelivery(Path(directory) / "outbox.db", InMemoryDestination(), max_pending_records=2)
            try:
                sender.enqueue([record])
                with self.assertRaises(OutboxCapacityError) as raised:
                    sender.enqueue([record, other, end])
                self.assertEqual((raised.exception.pending_records, raised.exception.incoming_records), (1, 2))
                self.assertEqual(sender.pending_count(), 1)
                self.assertEqual(sender.enqueue([end, end]), 1)
                self.assertEqual(sender.enqueue([record, end]), 0)
                with self.assertRaises(OutboxCapacityError):
                    sender.enqueue([other])
                self.assertEqual(sender.flush(), 2)
                self.assertEqual([r["phase"] for r in sender.destination.records], ["START", "END"])
                self.assertEqual(sender.enqueue([record, end]), 0)
                self.assertEqual(sender.enqueue([other]), 1)
            finally:
                sender.close()

    def test_utf8_byte_limit_and_status_use_real_queue_age(self):
        record = replace(test_delivery.DeliveryTests().record(), camera_id="校园东门")
        size = len(json.dumps(record.to_dict(), ensure_ascii=False).encode("utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.db"
            sender = EventDelivery(path, InMemoryDestination(), max_pending_bytes=size - 1)
            try:
                with self.assertRaises(OutboxCapacityError) as raised:
                    sender.enqueue([record])
                self.assertEqual(raised.exception.incoming_bytes, size)
                self.assertEqual(sender.pending_count(), 0)
                sender.max_pending_bytes = size
                with patch("campus_safety_ai.core.event_delivery.time.time", return_value=100):
                    self.assertEqual(sender.enqueue([record]), 1)
                status = read_outbox_status(sender.connection, now=150)
                self.assertEqual(status["pendingBytes"], size)
                self.assertEqual(status["oldestPendingAt"], 100)
                self.assertEqual(status["oldestPendingAgeSeconds"], 50)
                self.assertEqual(status["unknownPendingAgeRecords"], 0)
                self.assertGreater(status["databaseBytes"] + status["databaseAuxiliaryBytes"], size)
            finally:
                sender.close()

    def test_multi_producer_capacity_check_cannot_oversubscribe(self):
        barrier = threading.Barrier(2)
        outcomes = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.db"
            EventDelivery(path, InMemoryDestination()).close()

            def produce(number):
                sender = EventDelivery(path, InMemoryDestination(), max_pending_records=1)
                try:
                    barrier.wait(timeout=2)
                    record = replace(test_delivery.DeliveryTests().record(), event_id=f"event-{number}",
                                     idempotency_key=f"event-{number}:1")
                    try:
                        outcomes.append(sender.enqueue([record]))
                    except OutboxCapacityError:
                        outcomes.append("full")
                finally:
                    sender.close()

            threads = [threading.Thread(target=produce, args=(i,)) for i in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(4)
                self.assertFalse(thread.is_alive())
            self.assertCountEqual(outcomes, [1, "full"])
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM outbox").fetchone()[0], 1)

    def test_old_queue_age_is_unknown_after_additive_migration(self):
        record = test_delivery.DeliveryTests().record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outbox.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE outbox(idempotency_key TEXT PRIMARY KEY, payload TEXT, "
                                   "delivered INTEGER DEFAULT 0)")
                connection.execute("INSERT INTO outbox VALUES(?, ?, 0)",
                                   (record.idempotency_key, json.dumps(record.to_dict())))
            sender = EventDelivery(path, InMemoryDestination())
            try:
                status = sender.retry_status()
                self.assertEqual(status["unknownPendingAgeRecords"], 1)
                self.assertIsNone(status["oldestPendingAt"])
                self.assertIsNone(status["oldestPendingAgeSeconds"])
            finally:
                sender.close()

    def test_background_propagates_capacity_without_discarding_queued_event(self):
        entered, release = threading.Event(), threading.Event()

        class Offline:
            def publish(self, record):
                entered.set()
                release.wait(2)
                raise OSError("offline")

        with tempfile.TemporaryDirectory() as directory:
            worker = BackgroundDelivery(Path(directory) / "outbox.db", Offline(), max_pending_records=1,
                                        shutdown_seconds=0.05)
            record = test_delivery.DeliveryTests().record()
            try:
                self.assertEqual(worker.submit([record]), 1)
                self.assertTrue(entered.wait(1))
                self.assertEqual(worker.submit([record]), 0)
                with self.assertRaises(OutboxCapacityError):
                    worker.submit([replace(record, event_id="next", idempotency_key="next:1")])
                self.assertEqual(worker.pending_count(), 1)
                self.assertEqual(worker.retry_status()["maxPendingRecords"], 1)
            finally:
                release.set()
                worker.close()
                worker._thread.join(2)

    def test_invalid_capacity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in (0, -1, 1.5, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    EventDelivery(Path(directory) / "outbox.db", InMemoryDestination(), max_pending_records=value)
