import tempfile
import unittest
from datetime import datetime, UTC
from pathlib import Path

from campus_safety_ai.contracts import EventRecord
from campus_safety_ai.core.event_delivery import EventDelivery, InMemoryDestination


class DeliveryTests(unittest.TestCase):
    def record(self) -> EventRecord:
        now = datetime(2026, 8, 25, tzinfo=UTC)
        return EventRecord(
            schema_version="1.0",
            event_id="event-1",
            revision=1,
            phase="START",
            event_type="intrusion",
            severity="warning",
            edge_id="edge-01",
            camera_id="gate-02",
            started_at=now,
            observed_at=now,
            ended_at=None,
            subject_track_keys=("gate-02:1:1:1",),
            confidence=0.9,
            model_version="fake-v1",
            config_version="intrusion-v1",
            idempotency_key="event-1:1",
            status="OPEN",
        )

    def test_duplicate_submit_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = InMemoryDestination()
            delivery = EventDelivery(Path(directory) / "outbox.sqlite3", target)
            try:
                self.assertEqual(delivery.submit([self.record()]), 1)
                self.assertEqual(delivery.submit([self.record()]), 0)
                self.assertEqual(len(target.records), 1)
                self.assertEqual(target.records[0]["evidenceUris"], [])
            finally:
                delivery.close()


if __name__ == "__main__":
    unittest.main()
